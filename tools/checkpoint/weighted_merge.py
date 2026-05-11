# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Weighted averaging utility for Megatron distributed checkpoints.

This tool is intentionally kept under ``tools/checkpoint``. It builds a model
only to obtain the sharded-state template, then streams each input checkpoint
through Megatron's distributed checkpointing APIs. Floating model tensors are
accumulated in fp32 on CPU; Transformer Engine ``_extra_state`` entries are
copied from one source checkpoint instead of averaged.
"""

import argparse
import copy
import math
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as torch_dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import create_default_global_save_plan
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    SavePlan,
    SavePlanner,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.core import maybe_load_config
from megatron.core.dist_checkpointing.mapping import (
    ShardedBase,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.dist_checkpointing.serialization import load_sharded_metadata
from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from megatron.core.dist_checkpointing.utils import (
    extract_sharded_base,
    force_all_tensors_to_non_fp8,
)
from megatron.core.dist_checkpointing.validation import (
    StrictHandling,
    determine_global_metadata,
    parse_strict_flag,
    validate_integrity_and_strict_load,
)

ITERATION_RE = re.compile(r"^iter_(\d+)$")
LATEST_CHECKPOINTED_ITERATION = "latest_checkpointed_iteration.txt"
SAVE_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_MODIFIERS = ("reverse", "scramble")
SUPPORTED_INPUT_BACKENDS = ("torch_dist",)
MERGE_EXECUTION_MODES = ("baseline", "cpu-resident", "file-backed-streaming", "direct-dcp-streaming")
MERGE_TEMPLATE_INIT_MODES = ("default", "cpu", "meta")
BYTE_ACCOUNTING_MODES = ("none", "rank0", "all")
RESOURCE_LOG_MODES = ("none", "rank0", "all")
FILE_BACKED_STAGING_LAYOUTS = ("shared-dtype", "per-tensor")


class WeightedMergeError(ValueError):
    """Raised when weighted checkpoint merge inputs are invalid."""


@dataclass(frozen=True)
class MergeTimings:
    """Wall-clock timing split for a checkpoint merge."""

    discovery: float = 0.0
    model_init: float = 0.0
    load: float = 0.0
    accumulation: float = 0.0
    save: float = 0.0
    verification: float = 0.0
    total: float = 0.0


@dataclass(frozen=True)
class MergeMemoryEstimate:
    """Local-rank tensor residency estimate for the whole-shard merge path."""

    mergeable_tensors: int = 0
    extra_state_entries: int = 0
    loaded_checkpoint_bytes: int = 0
    accumulator_bytes: int = 0
    output_tensor_bytes: int = 0
    extra_state_tensor_bytes: int = 0
    projected_cpu_peak_bytes: int = 0
    projected_gpu_peak_bytes: int = 0
    file_backed_staging_bytes: int = 0
    file_backed_staging_files: int = 0
    template_devices: tuple[str, ...] = ()


@dataclass(frozen=True)
class MergeResult:
    """Result metadata returned after a successful merge."""

    output_dir: Path
    input_dirs: tuple[Path, ...]
    weights: tuple[float, ...]
    timings: MergeTimings
    averaged_tensors: int
    copied_extra_states: int
    bytes_read: int = 0
    bytes_written: int = 0
    backend: str = ""
    verified_load: bool = False
    host_peak_bytes: int = 0
    gpu_peak_bytes: int = 0
    max_host_peak_bytes: int = 0
    max_host_peak_rank: int = 0
    max_gpu_peak_bytes: int = 0
    max_gpu_peak_rank: int = 0
    world_size: int = 1
    rank: int = 0
    implementation_mode: str = "baseline"
    memory_estimate: MergeMemoryEstimate = MergeMemoryEstimate()
    save_metadata_cache_requested: bool = False
    save_metadata_cache_reused: bool = False
    file_backed_staging_layout: str = "shared-dtype"
    preflight_only: bool = False


@dataclass
class _StreamingWorkItem:
    merge_path: tuple[Union[str, int], ...]
    load_path: tuple[Union[str, int], ...]
    tensor_index: int
    load_leaf: Any
    target_tensor: torch.Tensor
    chunk_dim: int
    chunk_start: int
    chunk_length: int
    chunk_shape: tuple[int, ...]
    accumulator_bytes: int


@dataclass(frozen=True)
class _StreamingFactoryComponentPlan:
    component_path: tuple[Union[str, int], ...]
    component_leaf: Any
    source_dtype: torch.dtype
    source_shape: tuple[int, ...]


@dataclass(frozen=True)
class _StreamingTensorPlan:
    path: tuple[Union[str, int], ...]
    tensor_index: int
    template_leaf: Any
    template_shape: tuple[int, ...]
    target_dtype: torch.dtype
    source_dtype: Optional[torch.dtype] = None
    source_shape: Optional[tuple[int, ...]] = None
    factory_components: Optional[list[_StreamingFactoryComponentPlan]] = None


@dataclass(frozen=True)
class _DirectDcpWriteSpec:
    path: tuple[Union[str, int], ...]
    template_leaf: Any
    sharded_key: str
    global_shape: tuple[int, ...]
    global_offsets: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    target_dtype: torch.dtype
    chunk_dim: int
    chunk_start: int
    chunk_length: int
    source_dtype: Optional[torch.dtype] = None
    source_shape: Optional[tuple[int, ...]] = None
    is_extra_state: bool = False


class _WeightedMergeDirectOutputSavePlanner(SavePlanner):
    """Public DCP planner that resolves exactly one merged output chunk per item."""

    def __init__(
        self,
        *,
        write_specs: list[_DirectDcpWriteSpec],
        resolved_input_dirs: list[Path],
        weights: list[float],
        load_strategies: dict[Path, TorchDistLoadShardedStrategy],
        extra_state_source_index: int,
    ) -> None:
        self.write_specs = write_specs
        self.resolved_input_dirs = resolved_input_dirs
        self.weights = weights
        self.load_strategies = load_strategies
        self.extra_state_source_index = extra_state_source_index
        self.load_time = 0.0
        self.accumulation_time = 0.0
        self.resolved_tensor_count = 0
        self.max_resolved_output_tensor_bytes = 0
        self._write_spec_by_index = {
            (spec.sharded_key, spec.global_offsets, spec.chunk_shape): spec
            for spec in write_specs
        }
        self._plan = SavePlan([])

    def set_up_planner(
        self,
        state_dict: ShardedStateDict,
        storage_meta: Any = None,
        is_coordinator: bool = False,
    ) -> None:
        self._state_dict = state_dict
        self._storage_meta = storage_meta
        self._is_coordinator = is_coordinator

    def create_local_plan(self) -> SavePlan:
        items = [
            self._write_item(spec, index)
            for index, spec in enumerate(self.write_specs)
        ]
        self._plan = SavePlan(items)
        return self._plan

    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Any]:
        return create_default_global_save_plan(all_plans)

    def finish_plan(self, new_plan: SavePlan) -> SavePlan:
        self._plan = new_plan
        return self._plan

    def resolve_data(self, write_item: WriteItem) -> torch.Tensor:
        if write_item.type != WriteItemType.SHARD or write_item.tensor_data is None:
            raise WeightedMergeError("direct-dcp-streaming emits only tensor shard write items.")
        chunk = write_item.tensor_data.chunk
        spec_key = (
            write_item.index.fqn,
            tuple(int(offset) for offset in chunk.offsets),
            tuple(int(size) for size in chunk.sizes),
        )
        try:
            spec = self._write_spec_by_index[spec_key]
        except KeyError as exc:
            raise WeightedMergeError(
                "direct-dcp-streaming could not resolve public DCP write item "
                f"for '{write_item.index.fqn}' at offsets={tuple(chunk.offsets)} "
                f"sizes={tuple(chunk.sizes)}."
            ) from exc
        if spec.is_extra_state:
            output = self._resolve_extra_state(spec)
        else:
            output = self._resolve_merged_chunk(spec)
        self.resolved_tensor_count += 1
        self.max_resolved_output_tensor_bytes = max(
            self.max_resolved_output_tensor_bytes,
            int(output.numel() * output.element_size()),
        )
        return output

    @staticmethod
    def _write_item(spec: _DirectDcpWriteSpec, index: int) -> WriteItem:
        fake_tensor = torch.empty((), dtype=spec.target_dtype, device="cpu")
        offsets = torch.Size(spec.global_offsets)
        sizes = torch.Size(spec.chunk_shape)
        return WriteItem(
            index=MetadataIndex(spec.sharded_key, offsets, index),
            type=WriteItemType.SHARD,
            tensor_data=TensorWriteData(
                chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
                properties=TensorProperties.create_from_tensor(fake_tensor),
                size=torch.Size(spec.global_shape),
            ),
        )

    def _resolve_merged_chunk(self, spec: _DirectDcpWriteSpec) -> torch.Tensor:
        if spec.source_dtype is None or spec.source_shape is None:
            raise WeightedMergeError(
                f"Direct DCP tensor plan for '{_path_label(spec.path, spec.template_leaf)}' "
                "is missing source metadata."
            )
        load_leaf = _streaming_chunk_leaf(
            spec.template_leaf,
            chunk_dim=spec.chunk_dim,
            chunk_start=spec.chunk_start,
            chunk_length=spec.chunk_length,
            dtype=spec.source_dtype,
            global_shape=spec.source_shape,
        )
        path_leaves = [(spec.path, load_leaf)]
        accumulator = torch.zeros(spec.chunk_shape, dtype=torch.float32, device="cpu")
        for checkpoint_dir, weight in zip(self.resolved_input_dirs, self.weights):
            load_start = time.perf_counter()
            loaded_by_path = _load_tensor_path_group_fast(
                checkpoint_dir,
                path_leaves,
                self.load_strategies[checkpoint_dir],
            )
            self.load_time += time.perf_counter() - load_start
            tensor = _as_tensor(loaded_by_path[spec.path])
            if tensor is None:
                raise WeightedMergeError(
                    f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                    "is not a tensor."
                )
            if not tensor.is_floating_point():
                raise WeightedMergeError(
                    f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                    f"has non-floating dtype {tensor.dtype}; weighted averaging is "
                    "only supported for floating tensors."
                )
            if tuple(tensor.shape) != spec.chunk_shape:
                if tensor.numel() != accumulator.numel():
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(spec.path)}' "
                        f"loaded chunk shape {tuple(tensor.shape)} cannot be reshaped "
                        f"to {spec.chunk_shape}."
                    )
                tensor = tensor.reshape(spec.chunk_shape)
            accumulation_start = time.perf_counter()
            accumulator.add_(tensor.detach().to(dtype=torch.float32, device="cpu"), alpha=weight)
            self.accumulation_time += time.perf_counter() - accumulation_start
        return accumulator.to(dtype=spec.target_dtype).contiguous()

    def _resolve_extra_state(self, spec: _DirectDcpWriteSpec) -> torch.Tensor:
        load_leaf = copy.deepcopy(spec.template_leaf)
        template_tensor = _as_tensor(load_leaf)
        if template_tensor is not None:
            _assign_leaf_data(load_leaf, torch.empty_like(template_tensor, device="cpu"))
        load_start = time.perf_counter()
        loaded = _load_path_group(
            self.resolved_input_dirs[self.extra_state_source_index],
            [(spec.path, load_leaf)],
            strict=StrictHandling.ASSUME_OK_UNEXPECTED,
        )
        self.load_time += time.perf_counter() - load_start
        value = _copy_loaded_value(loaded[spec.path])
        tensor = _as_tensor(value)
        if tensor is None:
            raise WeightedMergeError(
                "direct-dcp-streaming currently supports tensor _extra_state leaves only; "
                f"'{_path_label(spec.path, spec.template_leaf)}' loaded as {type(value).__name__}."
            )
        return tensor.detach().cpu().contiguous()


def is_rank_0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def print_rank_0(*args: Any, **kwargs: Any) -> None:
    if is_rank_0():
        print(*args, **kwargs)


def ensure_process_group() -> None:
    """Initialize a gloo process group when one is not already active."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(
        backend="gloo", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"])
    )


def iteration_dir_name(iteration: int) -> str:
    if iteration < 0:
        raise WeightedMergeError(f"Iteration must be non-negative, got {iteration}.")
    return f"iter_{iteration:07d}"


def _schedule_linear_decay(x: float) -> float:
    if x < 0:
        raise WeightedMergeError(f"Schedule position must be non-negative, got {x}.")
    return 1 - x


def _schedule_minus_sqrt_decay(x: float) -> float:
    if x < 0:
        raise WeightedMergeError(f"Schedule position must be non-negative, got {x}.")
    return 1 - math.sqrt(x)


SCHEDULES: dict[str, Callable[[float], float]] = {
    "linear": _schedule_linear_decay,
    "minus-sqrt": _schedule_minus_sqrt_decay,
}


def parse_schedule_style(style: str) -> tuple[str, Optional[str]]:
    """Parse ``base__modifier`` coefficient style strings."""

    if "__" not in style:
        return style, None
    base_schedule, modifier = style.split("__", 1)
    if modifier not in VALID_MODIFIERS:
        raise WeightedMergeError(
            f"Unknown coefficient modifier '{modifier}'. "
            f"Valid modifiers are: {', '.join(VALID_MODIFIERS)}."
        )
    return base_schedule, modifier


def get_valid_styles() -> list[str]:
    styles = list(SCHEDULES)
    for schedule in SCHEDULES:
        styles.extend(f"{schedule}__{modifier}" for modifier in VALID_MODIFIERS)
    return styles


def schedule_to_merge_coefficients(
    schedule_fn: Callable[[float], float], n_checkpoints: int
) -> list[float]:
    """Convert a decay schedule into discrete checkpoint merge coefficients."""

    if n_checkpoints < 0:
        raise WeightedMergeError(
            f"Number of checkpoints must be non-negative, got {n_checkpoints}."
        )
    if n_checkpoints <= 1:
        return [1.0] * n_checkpoints

    decay_schedule = [schedule_fn(index / n_checkpoints) for index in range(n_checkpoints)]
    coefficients = [0.0 for _ in range(n_checkpoints)]
    coefficients[-1] = decay_schedule[-1]
    for index in range(1, n_checkpoints - 1):
        coefficients[index] = decay_schedule[index] - decay_schedule[index + 1]
    coefficients[0] = 1 - sum(coefficients)
    return coefficients


def apply_modifier(
    coefficients: list[float], modifier: Optional[str], seed: Optional[int] = 0
) -> list[float]:
    """Apply a deterministic coefficient modifier."""

    if modifier is None:
        return coefficients
    if modifier == "reverse":
        return list(reversed(coefficients))
    if modifier == "scramble":
        shuffled = list(coefficients)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    raise WeightedMergeError(f"Unknown coefficient modifier '{modifier}'.")


def checkpoint_coefficients(
    checkpoints: list[int], schedule: str, seed: Optional[int] = 0
) -> dict[int, float]:
    """Return ``iteration -> coefficient`` for the checkpoints in input order."""

    base_schedule, modifier = parse_schedule_style(schedule)
    if base_schedule not in SCHEDULES:
        raise WeightedMergeError(
            f"Unknown coefficient schedule '{base_schedule}'. "
            f"Valid schedules are: {', '.join(SCHEDULES)}."
        )

    coefficients = schedule_to_merge_coefficients(SCHEDULES[base_schedule], len(checkpoints))
    coefficients = apply_modifier(coefficients, modifier, seed)
    return dict(zip(checkpoints, coefficients))


def normalize_weights(weights: Iterable[float]) -> list[float]:
    weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in weights):
        raise WeightedMergeError(f"Weights must be finite, got {weights}.")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise WeightedMergeError(f"Weight sum must be positive, got {total}.")
    return [weight / total for weight in weights]


def validate_weights(weights: Iterable[float]) -> list[float]:
    weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) for weight in weights):
        raise WeightedMergeError(f"Weights must be finite, got {weights}.")
    return weights


def _manual_weight_warnings(weights: Iterable[float], *, normalize: bool) -> list[str]:
    weights = [float(weight) for weight in weights]
    messages: list[str] = []
    if any(weight < 0 for weight in weights):
        messages.append(
            "WARNING: manual merge weights include negative values; this is allowed for "
            "subtractive merges but can produce outputs outside the input checkpoint range."
        )
    if not normalize:
        total = sum(weights)
        if math.isfinite(total) and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            messages.append(
                f"WARNING: manual merge weights sum to {total:.12g} without --normalize; "
                "the merged tensors will be scaled by that total."
            )
    return messages


def warn_manual_weight_policy(weights: Iterable[float], *, normalize: bool) -> None:
    for message in _manual_weight_warnings(weights, normalize=normalize):
        print_rank_0(message, flush=True)


def parse_weighted_inputs(specs: Iterable[str]) -> tuple[list[Path], list[float]]:
    """Parse manual ``PATH:WEIGHT`` input specifications."""

    paths: list[Path] = []
    weights: list[float] = []
    for spec in specs:
        if ":" not in spec:
            raise WeightedMergeError(f"Input must be PATH:WEIGHT, got '{spec}'.")
        path, weight = spec.rsplit(":", 1)
        paths.append(Path(path))
        try:
            weights.append(float(weight))
        except ValueError as exc:
            raise WeightedMergeError(f"Invalid weight in '{spec}'.") from exc
    return paths, weights


def discover_checkpoint_iterations(checkpoint_root: Union[str, Path]) -> list[int]:
    """Discover sorted ``iter_*`` checkpoint directories under ``checkpoint_root``."""

    root = Path(checkpoint_root)
    if not root.exists():
        raise WeightedMergeError(f"Checkpoint root does not exist: {root}.")
    if not root.is_dir():
        raise WeightedMergeError(f"Checkpoint root is not a directory: {root}.")

    iterations = []
    for child in root.iterdir():
        match = ITERATION_RE.match(child.name)
        if child.is_dir() and match:
            iterations.append(int(match.group(1)))
    return sorted(iterations)


def filter_checkpoints_by_interval(
    checkpoints: list[int], min_iteration_interval: Optional[int]
) -> list[int]:
    """Greedily keep checkpoints at least ``min_iteration_interval`` apart.

    Filtering walks backward from the target checkpoint, so the last checkpoint
    in the input list is always preserved.
    """

    if not checkpoints or min_iteration_interval is None or min_iteration_interval <= 0:
        return list(checkpoints)

    filtered = []
    last_selected = None
    for checkpoint in reversed(checkpoints):
        if last_selected is None or last_selected - checkpoint >= min_iteration_interval:
            filtered.append(checkpoint)
            last_selected = checkpoint
    return list(reversed(filtered))


def derive_start_iteration_from_token_window(
    end_iteration: int, token_window_btok: int, seq_length: int, global_batch_size: int
) -> int:
    tokens_per_iteration = seq_length * global_batch_size
    if tokens_per_iteration <= 0:
        raise WeightedMergeError(
            f"Tokens per iteration must be positive, got {tokens_per_iteration}."
        )
    if token_window_btok <= 0:
        raise WeightedMergeError(f"Token window must be positive, got {token_window_btok}.")

    window_tokens = token_window_btok * 1_000_000_000
    iterations = math.ceil(window_tokens / tokens_per_iteration)
    return max(end_iteration - iterations, 0)


def select_checkpoints_in_window(
    checkpoint_root: Union[str, Path],
    *,
    start_iteration: Optional[int],
    end_iteration: int,
    token_window_btok: Optional[int] = None,
    seq_length: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    min_iteration_interval: Optional[int] = None,
) -> list[int]:
    """Select sorted checkpoint iterations for range or token-window merging."""

    if token_window_btok is not None:
        if seq_length is None or global_batch_size is None:
            raise WeightedMergeError(
                "Token-window selection requires seq_length and global_batch_size."
            )
        start_iteration = derive_start_iteration_from_token_window(
            end_iteration, token_window_btok, seq_length, global_batch_size
        )
    if start_iteration is None:
        raise WeightedMergeError("start_iteration is required for checkpoint selection.")
    if start_iteration > end_iteration:
        raise WeightedMergeError(
            f"start_iteration ({start_iteration}) must be <= end_iteration ({end_iteration})."
        )

    available = discover_checkpoint_iterations(checkpoint_root)
    if end_iteration not in available:
        raise WeightedMergeError(
            f"Target iteration {end_iteration} is not present under {checkpoint_root}."
        )

    selected = [
        iteration for iteration in available if start_iteration <= iteration <= end_iteration
    ]
    selected = filter_checkpoints_by_interval(selected, min_iteration_interval)
    if not selected or selected[-1] != end_iteration:
        raise WeightedMergeError(
            f"Checkpoint selection did not preserve target iteration {end_iteration}."
        )
    return selected


def checkpoint_paths_for_iterations(
    checkpoint_root: Union[str, Path], iterations: Iterable[int]
) -> list[Path]:
    root = Path(checkpoint_root)
    return [root / iteration_dir_name(iteration) for iteration in iterations]


def validate_min_checkpoints(num_checkpoints: int, min_checkpoints: Optional[int]) -> None:
    """Validate an optional minimum input checkpoint count."""

    if min_checkpoints is None:
        return
    if min_checkpoints < 1:
        raise WeightedMergeError(f"min_checkpoints must be positive, got {min_checkpoints}.")
    if num_checkpoints < min_checkpoints:
        raise WeightedMergeError(
            f"Selected {num_checkpoints} checkpoints, but at least {min_checkpoints} are required."
        )


def _read_latest_checkpointed_iteration(path: Path) -> Union[int, str]:
    tracker = path / LATEST_CHECKPOINTED_ITERATION
    if not tracker.exists():
        raise WeightedMergeError(f"Missing {LATEST_CHECKPOINTED_ITERATION} under {path}.")
    value = tracker.read_text(encoding="utf-8").strip()
    if value == "release":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise WeightedMergeError(
            f"Invalid latest checkpoint marker '{value}' in {tracker}."
        ) from exc


def resolve_checkpoint_dir(path: Union[str, Path]) -> Path:
    """Resolve direct, release, or latest-marker checkpoint paths."""

    checkpoint = Path(path)
    if (checkpoint / "metadata.json").exists():
        return checkpoint

    if (checkpoint / LATEST_CHECKPOINTED_ITERATION).exists():
        latest = _read_latest_checkpointed_iteration(checkpoint)
        if latest == "release":
            resolved = checkpoint / "release"
        else:
            resolved = checkpoint / iteration_dir_name(latest)
        if not (resolved / "metadata.json").exists():
            raise WeightedMergeError(
                f"{checkpoint} points to {resolved}, but that is not a distributed checkpoint."
            )
        return resolved

    raise WeightedMergeError(
        f"{checkpoint} is not a distributed checkpoint and has no "
        f"{LATEST_CHECKPOINTED_ITERATION} marker."
    )


def output_checkpoint_dir(output_root: Union[str, Path], output_iteration: Optional[int]) -> Path:
    """Return the concrete directory to pass to dist_checkpointing.save."""

    root = Path(output_root)
    if output_iteration is None:
        return root

    expected_name = iteration_dir_name(output_iteration)
    if root.name == expected_name:
        return root
    if ITERATION_RE.match(root.name):
        raise WeightedMergeError(
            f"Output directory {root} does not match requested iteration {output_iteration}."
        )
    return root / expected_name


def write_latest_checkpointed_iteration(checkpoint_dir: Union[str, Path], iteration: int) -> None:
    """Write Megatron's latest-checkpoint marker for an iteration checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / "metadata.json").exists():
        raise WeightedMergeError(
            f"Refusing to write {LATEST_CHECKPOINTED_ITERATION} because {checkpoint_dir} "
            "does not contain distributed checkpoint metadata."
        )
    parent = checkpoint_dir.parent if ITERATION_RE.match(checkpoint_dir.name) else checkpoint_dir
    tracker = parent / LATEST_CHECKPOINTED_ITERATION
    tracker.parent.mkdir(parents=True, exist_ok=True)
    temporary_tracker = tracker.with_name(f".{tracker.name}.tmp.{os.getpid()}")
    temporary_tracker.write_text(f"{iteration}\n", encoding="utf-8")
    os.replace(temporary_tracker, tracker)


def _checkpoint_format(checkpoint_dir: Path) -> str:
    config = maybe_load_config(str(checkpoint_dir))
    if config is None:
        raise WeightedMergeError(
            f"Missing distributed checkpoint metadata.json in {checkpoint_dir}."
        )
    return config.sharded_backend


def _directory_size(path: Union[str, Path]) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _directory_size_for_accounting(path: Union[str, Path], byte_accounting: str) -> int:
    if byte_accounting == "none":
        return 0
    if byte_accounting == "rank0" and not is_rank_0():
        return 0
    return _directory_size(path)


def _host_peak_memory_bytes() -> int:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _host_current_memory_bytes() -> int:
    status_path = Path("/proc/self/status")
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    return _host_peak_memory_bytes()


def _gpu_peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def _distributed_memory_peaks(
    host_peak_bytes: int, gpu_peak_bytes: int
) -> tuple[int, int, int, int, int, int]:
    """Return local rank/world size plus max host and GPU peaks across ranks."""

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local_record = {
        "rank": rank,
        "host_peak_bytes": int(host_peak_bytes),
        "gpu_peak_bytes": int(gpu_peak_bytes),
    }
    if world_size == 1:
        return rank, world_size, rank, int(host_peak_bytes), rank, int(gpu_peak_bytes)

    records: list[Optional[dict[str, int]]] = [None for _ in range(world_size)]
    dist.all_gather_object(records, local_record)
    valid_records = [record for record in records if record is not None]
    host_record = max(
        valid_records,
        key=lambda record: (int(record["host_peak_bytes"]), -int(record["rank"])),
    )
    gpu_record = max(
        valid_records,
        key=lambda record: (int(record["gpu_peak_bytes"]), -int(record["rank"])),
    )
    return (
        rank,
        world_size,
        int(host_record["rank"]),
        int(host_record["host_peak_bytes"]),
        int(gpu_record["rank"]),
        int(gpu_record["gpu_peak_bytes"]),
    )


def _should_log_resources(resource_log: str) -> bool:
    if resource_log == "none":
        return False
    if resource_log == "rank0":
        return is_rank_0()
    return True


def _log_resource_checkpoint(
    label: str, *, start_time: Optional[float] = None, resource_log: str = "none"
) -> None:
    if not _should_log_resources(resource_log):
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    elapsed = time.perf_counter() - start_time if start_time is not None else None
    elapsed_text = f", elapsed={elapsed:.2f}s" if elapsed is not None else ""
    print(
        f"Resource checkpoint rank={rank}: {label}{elapsed_text}, "
        f"host_current={_format_bytes(_host_current_memory_bytes())}, "
        f"host_peak={_format_bytes(_host_peak_memory_bytes())}, "
        f"gpu_peak={_format_bytes(_gpu_peak_memory_bytes())}",
        flush=True,
    )


def _flatten_items(value: Any, prefix: tuple[Union[str, int], ...] = ()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_items(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _flatten_items(item, prefix + (index,))
    else:
        yield prefix, value


def _get_path(root: Any, path: tuple[Union[str, int], ...]) -> Any:
    value = root
    for key in path:
        value = value[key]
    return value


def _single_path_state_dict(path: tuple[Union[str, int], ...], leaf: Any) -> ShardedStateDict:
    if not path:
        raise WeightedMergeError("Cannot build a single-path state dict for an empty path.")
    value: Any = leaf
    for key in reversed(path):
        if isinstance(key, int):
            items: list[Any] = [None for _ in range(key + 1)]
            items[key] = value
            value = items
        else:
            value = {key: value}
    return value


def _merge_state_dict_containers(target: Any, source: Any) -> Any:
    if target is None:
        return source
    if isinstance(target, dict) and isinstance(source, dict):
        for key, value in source.items():
            target[key] = _merge_state_dict_containers(target.get(key), value)
        return target
    if isinstance(target, list) and isinstance(source, list):
        if len(target) < len(source):
            target.extend([None] * (len(source) - len(target)))
        for index, value in enumerate(source):
            if value is not None:
                target[index] = _merge_state_dict_containers(target[index], value)
        return target
    raise WeightedMergeError("Conflicting container structure while building partial state dict.")


def _multi_path_state_dict(
    path_leaves: Iterable[tuple[tuple[Union[str, int], ...], Any]],
) -> ShardedStateDict:
    state_dict: Any = None
    for path, leaf in path_leaves:
        state_dict = _merge_state_dict_containers(state_dict, _single_path_state_dict(path, leaf))
    if state_dict is None:
        raise WeightedMergeError("Cannot build a partial state dict with no paths.")
    return state_dict


def _path_label(path: tuple[Union[str, int], ...], leaf: Any = None) -> str:
    key = getattr(leaf, "key", None)
    if key:
        return str(key)
    return ".".join(str(part) for part in path)


def _is_sharded_leaf(value: Any) -> bool:
    return hasattr(value, "data") and hasattr(value, "replica_id")


def _is_extra_state(path: tuple[Union[str, int], ...], leaf: Any) -> bool:
    label = _path_label(path, leaf)
    return (
        label == "_extra_state"
        or label.endswith("._extra_state")
        or any(str(part) == "_extra_state" or str(part).endswith("._extra_state") for part in path)
    )


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    data = getattr(value, "data", None)
    if torch.is_tensor(data):
        return data
    return None


def _copy_loaded_value(value: Any) -> Any:
    tensor = _as_tensor(value)
    if tensor is not None:
        return tensor.detach().cpu().clone()
    return copy.deepcopy(value)


def _assign_leaf_data(leaf: Any, value: Any) -> None:
    if hasattr(leaf, "data"):
        leaf.data = value
    else:
        raise WeightedMergeError(f"Cannot assign merged value to non-sharded leaf {leaf!r}.")


def _clone_sharded_template_without_data(value: Any) -> Any:
    """Clone state-dict containers and sharded metadata without tensor buffers."""

    if isinstance(value, dict):
        return {key: _clone_sharded_template_without_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_sharded_template_without_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_sharded_template_without_data(item) for item in value)
    if isinstance(value, ShardedBase):
        return value.without_data()
    return copy.deepcopy(value)


def _move_sharded_leaf_tensors_to_cpu(sharded_state_dict: ShardedStateDict) -> ShardedStateDict:
    """Replace sharded tensor leaf buffers with CPU buffers of the same shape and dtype."""

    for _, leaf in _flatten_items(sharded_state_dict):
        tensor = _as_tensor(leaf)
        if tensor is None:
            continue
        if tensor.device.type == "cpu":
            continue
        _assign_leaf_data(leaf, torch.empty_like(tensor, device="cpu"))
    return sharded_state_dict


def _state_dict_for_execution_mode(
    sharded_state_dict_factory: Callable[[], ShardedStateDict], execution_mode: str
) -> ShardedStateDict:
    sharded_state_dict = sharded_state_dict_factory()
    if execution_mode in ("cpu-resident", "file-backed-streaming", "direct-dcp-streaming"):
        return _move_sharded_leaf_tensors_to_cpu(sharded_state_dict)
    return sharded_state_dict


def _broadcast_rank0_value(value: Any) -> Any:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    objects = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    return objects[0]


def _run_rank0_filesystem_op(description: str, operation: Callable[[], None]) -> None:
    """Run a filesystem publication step on rank 0 and raise consistently on all ranks."""

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        operation()
        return

    error_messages: list[Optional[str]] = [None]
    if dist.get_rank() == 0:
        try:
            operation()
        except Exception as exc:
            error_messages[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error_messages, src=0)
    if error_messages[0] is not None:
        raise WeightedMergeError(f"{description} failed on rank 0: {error_messages[0]}")


def _direct_dcp_save_uses_no_dist() -> bool:
    """Return whether direct DCP output should bypass distributed save orchestration."""

    return not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    revision = result.stdout.strip()
    return revision or None


def _checkpoint_iteration_from_dir(checkpoint_dir: Path) -> Optional[int]:
    match = ITERATION_RE.match(checkpoint_dir.name)
    if match:
        return int(match.group(1))
    return None


def _add_merge_provenance(
    common_state: dict[str, Any],
    *,
    input_dirs: list[Path],
    weights: list[float],
    normalize: bool,
    save_dtype: str,
    output_iteration: Optional[int],
    extra_state_source_index: int,
    strict: StrictHandling,
    execution_mode: str,
    merge_style: Optional[str],
    file_backed_staging_layout: str,
) -> None:
    common_state["weighted_merge_provenance"] = {
        "format_version": 1,
        "input_paths": [str(path) for path in input_dirs],
        "source_iterations": [_checkpoint_iteration_from_dir(path) for path in input_dirs],
        "weights": [float(weight) for weight in weights],
        "normalize": bool(normalize),
        "merge_style": merge_style if merge_style is not None else "manual",
        "output_iteration": output_iteration,
        "output_dtype": save_dtype,
        "extra_state_source_index": extra_state_source_index,
        "extra_state_source_path": str(input_dirs[extra_state_source_index]),
        "implementation_mode": execution_mode,
        "file_backed_staging_layout": file_backed_staging_layout,
        "strict": strict.value,
        "code_revision": _git_revision(),
        "optimizer_merged": False,
        "rng_merged": False,
    }


def _prepare_temporary_output_dir(output_dir: Path, overwrite_output: bool) -> Path:
    if output_dir.exists() and not overwrite_output:
        raise WeightedMergeError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite-merge-output to replace it."
        )
    temporary_name = _broadcast_rank0_value(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary_dir = output_dir.parent / temporary_name
    if temporary_dir.exists():
        raise WeightedMergeError(f"Temporary output directory already exists: {temporary_dir}.")
    temporary_dir.parent.mkdir(parents=True, exist_ok=True)
    return temporary_dir


def _require_publishable_checkpoint_dir(checkpoint_dir: Path) -> None:
    if not (checkpoint_dir / "metadata.json").exists():
        raise WeightedMergeError(
            f"Refusing to publish {checkpoint_dir} because distributed checkpoint metadata is missing."
        )
    try:
        FileSystemReader(checkpoint_dir).read_metadata()
    except Exception as exc:
        raise WeightedMergeError(
            f"Refusing to publish {checkpoint_dir} because DCP metadata is missing or unreadable."
        ) from exc


def _publish_temporary_output_dir(
    temporary_dir: Path, output_dir: Path, *, overwrite_output: bool
) -> None:
    def publish() -> None:
        _require_publishable_checkpoint_dir(temporary_dir)
        if output_dir.exists():
            if not overwrite_output:
                raise WeightedMergeError(
                    f"Output directory already exists: {output_dir}. "
                    "Use --overwrite-merge-output to replace it."
                )
            backup_dir = output_dir.with_name(f".{output_dir.name}.old-{uuid.uuid4().hex}")
            output_dir.rename(backup_dir)
            try:
                temporary_dir.rename(output_dir)
            except Exception:
                if not output_dir.exists() and backup_dir.exists():
                    backup_dir.rename(output_dir)
                raise
            shutil.rmtree(backup_dir)
            return
        temporary_dir.rename(output_dir)

    _run_rank0_filesystem_op(f"Publishing merged checkpoint to {output_dir}", publish)


def _classify_template(
    sharded_state_dict: ShardedStateDict,
) -> tuple[list[tuple[Union[str, int], ...]], list[tuple[Union[str, int], ...]]]:
    merge_paths: list[tuple[Union[str, int], ...]] = []
    extra_paths: list[tuple[Union[str, int], ...]] = []

    for path, leaf in _flatten_items(sharded_state_dict):
        if not _is_sharded_leaf(leaf):
            continue
        if _is_extra_state(path, leaf):
            extra_paths.append(path)
            continue
        tensor = _as_tensor(leaf)
        if tensor is None:
            raise WeightedMergeError(
                f"Template entry '{_path_label(path, leaf)}' has no tensor data to merge."
            )
        merge_paths.append(path)

    if not merge_paths:
        raise WeightedMergeError("No mergeable tensor entries found in the sharded state dict.")
    return merge_paths, extra_paths


def _sharded_keys_by_path(
    sharded_state_dict: ShardedStateDict,
) -> dict[tuple[Union[str, int], ...], str]:
    keys: dict[tuple[Union[str, int], ...], str] = {}
    for path, leaf in _flatten_items(sharded_state_dict):
        key = getattr(leaf, "key", None)
        if key is not None:
            keys[path] = key
    return keys


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _tensor_numel(shape: Iterable[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return int(numel)


def _chunk_ranges_for_local_shape(
    local_shape: Iterable[int], chunk_bytes: int
) -> list[tuple[int, int, int]]:
    shape = tuple(int(dim) for dim in local_shape)
    if chunk_bytes < 1:
        raise WeightedMergeError(f"streaming_chunk_bytes must be positive, got {chunk_bytes}.")
    if not shape:
        return [(-1, 0, 1)]
    if _tensor_numel(shape) == 0:
        return [(0, 0, 0)]

    chunk_dim = max(range(len(shape)), key=lambda index: shape[index])
    chunk_axis = shape[chunk_dim]
    elements_per_step = max(_tensor_numel(shape) // max(chunk_axis, 1), 1)
    max_chunk_len = max(chunk_bytes // (elements_per_step * _dtype_size(torch.float32)), 1)
    return [
        (chunk_dim, start, min(max_chunk_len, chunk_axis - start))
        for start in range(0, chunk_axis, max_chunk_len)
    ]


def _file_backed_tensor(path: Path, shape: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
    shape = tuple(int(dim) for dim in shape)
    numel = _tensor_numel(shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb"):
        pass
    return torch.from_file(str(path), shared=True, size=numel, dtype=dtype).view(shape)


def _dtype_staging_suffix(dtype: torch.dtype) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dtype).replace("torch.", ""))


class _FileBackedTensorStore:
    """Rank-local mapped staging files for merged output tensors."""

    def __init__(
        self,
        staging_dir: Path,
        rank: int,
        dtype_numels: dict[torch.dtype, int],
        *,
        layout: str = "shared-dtype",
    ) -> None:
        if layout not in FILE_BACKED_STAGING_LAYOUTS:
            raise WeightedMergeError(
                f"Unsupported file-backed staging layout '{layout}'. "
                f"Use one of: {', '.join(FILE_BACKED_STAGING_LAYOUTS)}."
            )
        self._staging_dir = staging_dir
        self._rank = rank
        self._layout = layout
        self._flat_tensors: dict[torch.dtype, torch.Tensor] = {}
        self._offsets: dict[torch.dtype, int] = {}
        self._allocation_index = 0
        if layout == "per-tensor":
            return
        for dtype, numel in sorted(dtype_numels.items(), key=lambda item: str(item[0])):
            if numel <= 0:
                continue
            path = staging_dir / f"rank{rank:05d}-{_dtype_staging_suffix(dtype)}.bin"
            self._flat_tensors[dtype] = _file_backed_tensor(path, (numel,), dtype)
            self._offsets[dtype] = 0

    @property
    def file_count(self) -> int:
        if self._layout == "shared-dtype":
            return len(self._flat_tensors)
        return self._allocation_index

    @property
    def layout(self) -> str:
        return self._layout

    def allocate(
        self, shape: Iterable[int], dtype: torch.dtype, label: Optional[str] = None
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        numel = _tensor_numel(shape)
        if numel == 0:
            return torch.empty(shape, dtype=dtype, device="cpu")
        if self._layout == "per-tensor":
            self._allocation_index += 1
            suffix = _dtype_staging_suffix(dtype)
            safe_label = (
                re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
                if label is not None
                else ""
            )
            label_part = f"-{safe_label[:96]}" if safe_label else ""
            path = (
                self._staging_dir
                / f"rank{self._rank:05d}-{self._allocation_index:06d}{label_part}-{suffix}.bin"
            )
            return _file_backed_tensor(path, shape, dtype)
        if dtype not in self._flat_tensors:
            raise WeightedMergeError(f"No file-backed staging tensor was prepared for {dtype}.")
        flat_tensor = self._flat_tensors[dtype]
        offset = self._offsets[dtype]
        if offset + numel > flat_tensor.numel():
            raise WeightedMergeError(
                f"File-backed staging allocation for {dtype} exceeded prepared capacity: "
                f"requested {offset + numel} elements, capacity is {flat_tensor.numel()}."
            )
        self._offsets[dtype] = offset + numel
        return flat_tensor.narrow(0, offset, numel).view(shape)


def _streaming_chunk_leaf(
    leaf: Any,
    *,
    chunk_dim: int,
    chunk_start: int,
    chunk_length: int,
    dtype: torch.dtype,
    global_shape: Iterable[int],
) -> Any:
    tensor = _as_tensor(leaf)
    if tensor is None:
        raise WeightedMergeError(f"Cannot create a tensor chunk for non-tensor leaf {leaf!r}.")

    prepended_axis_num = int(getattr(leaf, "prepend_axis_num", 0))
    local_shape = list(tensor.shape)
    if chunk_dim >= 0:
        local_shape[chunk_dim] = chunk_length
    data_shape = (1,) * prepended_axis_num + tuple(local_shape)
    data = torch.empty(data_shape, dtype=dtype, device="cpu")
    global_offset = list(getattr(leaf, "global_offset"))
    if chunk_dim >= 0:
        global_offset[prepended_axis_num + chunk_dim] += chunk_start
    return replace(
        leaf,
        data=data,
        dtype=dtype,
        local_shape=data_shape,
        global_shape=tuple(int(dim) for dim in global_shape),
        global_offset=tuple(global_offset),
        prepend_axis_num=0,
        axis_fragmentations=None,
    )


def _estimate_whole_shard_memory(
    sharded_state_dict: ShardedStateDict,
    merge_paths: list[tuple[Union[str, int], ...]],
    extra_paths: list[tuple[Union[str, int], ...]],
    *,
    save_dtype: str,
    execution_mode: str,
    streaming_chunk_bytes: int = 128 * 1024 * 1024,
    file_backed_staging_layout: str = "shared-dtype",
) -> MergeMemoryEstimate:
    """Estimate tensor residency for this rank's whole-shard merge work."""

    loaded_checkpoint_bytes = 0
    accumulator_bytes = 0
    output_tensor_bytes = 0
    extra_state_tensor_bytes = 0
    max_streaming_loaded_chunk_bytes = 0
    max_streaming_accumulator_bytes = 0
    file_backed_staging_dtypes: set[torch.dtype] = set()
    file_backed_staging_tensor_count = 0
    template_devices: set[str] = set()

    target_dtype = SAVE_DTYPE_MAP.get(save_dtype)
    for path in merge_paths:
        tensor = _as_tensor(_get_path(sharded_state_dict, path))
        if tensor is None:
            continue
        template_devices.add(tensor.device.type)
        local_numel = tensor.numel()
        loaded_checkpoint_bytes += local_numel * tensor.element_size()
        accumulator_bytes += local_numel * _dtype_size(torch.float32)
        output_dtype = target_dtype if target_dtype is not None else tensor.dtype
        output_tensor_bytes += local_numel * _dtype_size(output_dtype)
        if execution_mode == "file-backed-streaming" and local_numel > 0:
            file_backed_staging_dtypes.add(output_dtype)
            file_backed_staging_tensor_count += 1
        if execution_mode in ("file-backed-streaming", "direct-dcp-streaming"):
            for chunk_dim, _, chunk_length in _chunk_ranges_for_local_shape(
                tensor.shape, streaming_chunk_bytes
            ):
                chunk_shape = list(tensor.shape)
                if chunk_dim >= 0:
                    chunk_shape[chunk_dim] = chunk_length
                chunk_numel = _tensor_numel(chunk_shape)
                max_streaming_loaded_chunk_bytes = max(
                    max_streaming_loaded_chunk_bytes,
                    chunk_numel * tensor.element_size(),
                )
                max_streaming_accumulator_bytes = max(
                    max_streaming_accumulator_bytes,
                    chunk_numel * _dtype_size(torch.float32),
                )

    for path in extra_paths:
        tensor = _as_tensor(_get_path(sharded_state_dict, path))
        if tensor is None:
            continue
        template_devices.add(tensor.device.type)
        extra_state_tensor_bytes += tensor.numel() * tensor.element_size()

    file_backed_staging_bytes = 0
    file_backed_staging_files = 0
    if execution_mode in ("file-backed-streaming", "direct-dcp-streaming"):
        load_bytes_on_cpu = min(
            loaded_checkpoint_bytes,
            max(streaming_chunk_bytes, max_streaming_loaded_chunk_bytes),
        )
        group_accumulator_bytes = min(
            accumulator_bytes,
            max(streaming_chunk_bytes, max_streaming_accumulator_bytes),
        )
        load_bytes_on_gpu = 0
        output_bytes_on_gpu = 0
        save_output_chunk_bytes = min(output_tensor_bytes, max(streaming_chunk_bytes, 1))
        projected_cpu_peak_bytes = (
            group_accumulator_bytes
            + load_bytes_on_cpu
            + save_output_chunk_bytes
            + extra_state_tensor_bytes
        )
        if execution_mode == "file-backed-streaming":
            file_backed_staging_bytes = output_tensor_bytes
            file_backed_staging_files = (
                file_backed_staging_tensor_count
                if file_backed_staging_layout == "per-tensor"
                else len(file_backed_staging_dtypes)
            )
    elif execution_mode == "cpu-resident":
        load_bytes_on_cpu = loaded_checkpoint_bytes
        output_bytes_on_cpu = output_tensor_bytes
        load_bytes_on_gpu = 0
        output_bytes_on_gpu = 0
        projected_cpu_peak_bytes = (
            accumulator_bytes
            + max(load_bytes_on_cpu, output_bytes_on_cpu)
            + extra_state_tensor_bytes
        )
    else:
        template_uses_cuda = "cuda" in template_devices
        load_bytes_on_cpu = 0 if template_uses_cuda else loaded_checkpoint_bytes
        output_bytes_on_cpu = 0 if template_uses_cuda else output_tensor_bytes
        load_bytes_on_gpu = loaded_checkpoint_bytes if template_uses_cuda else 0
        output_bytes_on_gpu = output_tensor_bytes if template_uses_cuda else 0
        projected_cpu_peak_bytes = (
            accumulator_bytes
            + max(load_bytes_on_cpu, output_bytes_on_cpu)
            + extra_state_tensor_bytes
        )

    return MergeMemoryEstimate(
        mergeable_tensors=len(merge_paths),
        extra_state_entries=len(extra_paths),
        loaded_checkpoint_bytes=loaded_checkpoint_bytes,
        accumulator_bytes=accumulator_bytes,
        output_tensor_bytes=output_tensor_bytes,
        extra_state_tensor_bytes=extra_state_tensor_bytes,
        projected_cpu_peak_bytes=projected_cpu_peak_bytes,
        projected_gpu_peak_bytes=max(load_bytes_on_gpu, output_bytes_on_gpu),
        file_backed_staging_bytes=file_backed_staging_bytes,
        file_backed_staging_files=file_backed_staging_files,
        template_devices=tuple(sorted(template_devices)),
    )


def _log_memory_estimate(memory_estimate: MergeMemoryEstimate) -> None:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(
        f"Preflight memory estimate rank={rank}: "
        f"mergeable_tensors={memory_estimate.mergeable_tensors}, "
        f"extra_state_entries={memory_estimate.extra_state_entries}, "
        f"loaded_shard={_format_bytes(memory_estimate.loaded_checkpoint_bytes)}, "
        f"fp32_accumulator={_format_bytes(memory_estimate.accumulator_bytes)}, "
        f"output_shard={_format_bytes(memory_estimate.output_tensor_bytes)}, "
        f"extra_state_tensors={_format_bytes(memory_estimate.extra_state_tensor_bytes)}, "
        f"projected_cpu_peak={_format_bytes(memory_estimate.projected_cpu_peak_bytes)}, "
        f"projected_gpu_peak={_format_bytes(memory_estimate.projected_gpu_peak_bytes)}, "
        f"file_backed_staging={_format_bytes(memory_estimate.file_backed_staging_bytes)}, "
        f"file_backed_staging_files={memory_estimate.file_backed_staging_files}, "
        f"template_devices={','.join(memory_estimate.template_devices) or 'none'}; "
        "projected_cpu_peak excludes model construction, DCP planner overhead, "
        "and file-backed staging storage",
        flush=True,
    )


def _enforce_projected_cpu_memory_guard(
    memory_estimate: MergeMemoryEstimate, max_projected_cpu_bytes: Optional[int]
) -> None:
    if max_projected_cpu_bytes is None:
        return
    if memory_estimate.projected_cpu_peak_bytes > max_projected_cpu_bytes:
        raise WeightedMergeError(
            "Projected CPU peak memory "
            f"{_format_bytes(memory_estimate.projected_cpu_peak_bytes)} exceeds "
            f"--merge-max-projected-cpu-bytes={_format_bytes(max_projected_cpu_bytes)}. "
            "Use a larger merge-time model-parallel size, a bounded streaming mode, "
            "or raise the guard only after confirming node memory capacity."
        )


def _enforce_file_backed_staging_guard(
    memory_estimate: MergeMemoryEstimate, max_file_backed_staging_bytes: Optional[int]
) -> None:
    if max_file_backed_staging_bytes is None:
        return
    if memory_estimate.file_backed_staging_bytes > max_file_backed_staging_bytes:
        raise WeightedMergeError(
            "Projected file-backed staging storage "
            f"{_format_bytes(memory_estimate.file_backed_staging_bytes)} exceeds "
            "--merge-max-file-backed-staging-bytes="
            f"{_format_bytes(max_file_backed_staging_bytes)}. "
            "Use a larger merge-time model-parallel size, fewer rank-local output "
            "tensors, or raise the guard only after confirming filesystem and "
            "page-cache capacity."
        )


def _enforce_file_backed_staging_file_guard(
    memory_estimate: MergeMemoryEstimate, max_file_backed_staging_files: Optional[int]
) -> None:
    if max_file_backed_staging_files is None:
        return
    if memory_estimate.file_backed_staging_files > max_file_backed_staging_files:
        raise WeightedMergeError(
            "Projected file-backed staging file count "
            f"{memory_estimate.file_backed_staging_files} exceeds "
            f"--merge-max-file-backed-staging-files={max_file_backed_staging_files}. "
            "Use shared-dtype staging, a larger merge-time model-parallel size, or raise "
            "the guard only after confirming filesystem metadata capacity."
        )


def _prepare_common_state(
    common_state: dict[str, Any], output_iteration: Optional[int]
) -> dict[str, Any]:
    common_state = copy.deepcopy(common_state)
    if output_iteration is None:
        return common_state

    common_state["iteration"] = output_iteration
    checkpoint_args = common_state.get("args")
    if checkpoint_args is not None and hasattr(checkpoint_args, "iteration"):
        checkpoint_args.iteration = output_iteration
    return common_state


def _load_single_path(
    checkpoint_dir: Path,
    path: tuple[Union[str, int], ...],
    leaf: Any,
    *,
    strict: StrictHandling,
) -> Any:
    loaded = dist_checkpointing.load(
        _single_path_state_dict(path, leaf),
        str(checkpoint_dir),
        validate_access_integrity=False,
        strict=strict,
    )
    return _get_path(loaded, path)


def _load_path_group(
    checkpoint_dir: Path,
    path_leaves: list[tuple[tuple[Union[str, int], ...], Any]],
    *,
    strict: StrictHandling,
) -> dict[tuple[Union[str, int], ...], Any]:
    loaded = dist_checkpointing.load(
        _multi_path_state_dict(path_leaves),
        str(checkpoint_dir),
        validate_access_integrity=False,
        strict=strict,
    )
    return {path: _get_path(loaded, path) for path, _ in path_leaves}


def _load_tensor_path_group_fast(
    checkpoint_dir: Path,
    path_leaves: list[tuple[tuple[Union[str, int], ...], Any]],
    sharded_strategy: TorchDistLoadShardedStrategy,
) -> dict[tuple[Union[str, int], ...], Any]:
    """Load sharded tensor chunks without reloading common checkpoint state."""

    partial_state_dict = _multi_path_state_dict(path_leaves)
    force_all_tensors_to_non_fp8(partial_state_dict)
    loaded = sharded_strategy.load(partial_state_dict, checkpoint_dir, async_strategy="mcore")
    return {path: _get_path(loaded, path) for path, _ in path_leaves}


def _streaming_validation_state_dict(initial_template: ShardedStateDict) -> ShardedStateDict:
    validation_state_dict, _, _ = load_preprocess(initial_template)
    force_all_tensors_to_non_fp8(validation_state_dict)
    sharded_state_dict, _ = extract_sharded_base(validation_state_dict)
    return sharded_state_dict


def _validate_streaming_load_contract(
    *,
    initial_template: ShardedStateDict,
    resolved_input_dirs: list[Path],
    load_strategies: dict[Path, TorchDistLoadShardedStrategy],
    strict: StrictHandling,
    validate_access_integrity: bool,
) -> None:
    """Run DCP strict/access checks once before streaming bypasses load()."""

    needs_checkpoint_mismatch_check = StrictHandling.requires_explicit_ckpt_mismatch_check(strict)
    needs_global_metadata = validate_access_integrity or StrictHandling.requires_global_app_metadata(
        strict
    )
    if not needs_checkpoint_mismatch_check and not needs_global_metadata:
        return

    validation_state_dict = _streaming_validation_state_dict(initial_template)
    local_metadata = global_metadata = None
    if needs_global_metadata:
        local_metadata, global_metadata = determine_global_metadata(validation_state_dict)

    if validate_access_integrity:
        validate_integrity_and_strict_load(
            validation_state_dict,
            StrictHandling.ASSUME_OK_UNEXPECTED,
            True,
            local_metadata,
            global_metadata,
            None,
        )

    if not needs_checkpoint_mismatch_check:
        return

    for checkpoint_dir in resolved_input_dirs:
        checkpoint_metadata = load_sharded_metadata(
            str(checkpoint_dir), load_strategies[checkpoint_dir]
        )
        try:
            validate_integrity_and_strict_load(
                validation_state_dict,
                strict,
                False,
                local_metadata,
                global_metadata,
                checkpoint_metadata,
            )
        except Exception as exc:
            raise WeightedMergeError(
                f"Checkpoint {checkpoint_dir} failed strict validation for "
                f"file-backed streaming: {exc}"
            ) from exc


def _save_strategy_from_source_metadata(
    source_metadata: Optional[Any], *, requested: bool
) -> Optional[TorchDistSaveShardedStrategy]:
    if not requested:
        return None
    if source_metadata is None:
        print_rank_0(
            "DCP save metadata cache was requested, but no source metadata cache is available; "
            "falling back to normal save planning.",
            flush=True,
        )
        return None
    if getattr(source_metadata, "all_local_plans", None) is None:
        print_rank_0(
            "DCP save metadata cache was requested, but source metadata does not include "
            "all_local_plans; falling back to normal save planning.",
            flush=True,
        )
        return None
    if not torch.cuda.is_available():
        print_rank_0(
            "DCP save metadata cache was requested, but Megatron's metadata-reuse verifier "
            "requires CUDA; falling back to normal save planning.",
            flush=True,
        )
        return None

    save_strategy = TorchDistSaveShardedStrategy(cached_metadata=True)
    save_strategy.cached_global_metadata = source_metadata
    return save_strategy


def _report_save_metadata_cache_reuse(
    save_strategy: Optional[TorchDistSaveShardedStrategy], *, requested: bool
) -> bool:
    """Report whether DCP accepted seeded source metadata for save planning."""

    if not requested:
        return False

    save_metadata_cache_reused = bool(
        save_strategy is not None
        and getattr(save_strategy, "validated_loaded_metadata_reuse", False)
    )
    if save_metadata_cache_reused:
        print_rank_0("DCP save metadata cache reuse: reused source metadata", flush=True)
    elif save_strategy is None:
        print_rank_0("DCP save metadata cache reuse: not reused", flush=True)
    else:
        print_rank_0(
            "DCP save metadata cache reuse: not reused; source metadata was seeded, "
            "but DCP did not validate loaded metadata reuse, so the save fell back to "
            "metadata generated during save planning.",
            flush=True,
        )
    return save_metadata_cache_reused


def _direct_dcp_global_offsets(
    leaf: Any, *, chunk_dim: int, chunk_start: int
) -> torch.Size:
    global_offset = list(getattr(leaf, "global_offset", ()))
    if not global_offset:
        tensor = _as_tensor(leaf)
        if tensor is None:
            raise WeightedMergeError(f"Cannot infer DCP offsets for non-tensor leaf {leaf!r}.")
        global_offset = [0 for _ in tensor.shape]
    if chunk_dim >= 0:
        global_offset[chunk_dim] += chunk_start
    return torch.Size(global_offset)


def _direct_dcp_chunk_sizes(chunk_shape: Iterable[int]) -> torch.Size:
    return torch.Size(tuple(int(dim) for dim in chunk_shape))


def _validate_direct_dcp_plan(plan: _StreamingTensorPlan) -> None:
    if plan.factory_components is not None:
        raise WeightedMergeError(
            "direct-dcp-streaming currently supports ordinary ShardedTensor leaves only; "
            f"'{_path_label(plan.path, plan.template_leaf)}' is a ShardedTensorFactory. "
            "Use --merge-execution-mode=file-backed-streaming for this checkpoint."
        )
    leaf = plan.template_leaf
    if int(getattr(leaf, "prepend_axis_num", 0) or 0) != 0:
        raise WeightedMergeError(
            "direct-dcp-streaming does not yet support prepended-axis tensors; "
            f"'{_path_label(plan.path, leaf)}' has prepend_axis_num="
            f"{getattr(leaf, 'prepend_axis_num')}. Use file-backed-streaming."
        )
    if getattr(leaf, "flattened_range", None) is not None:
        raise WeightedMergeError(
            "direct-dcp-streaming does not yet support flattened-range tensors; "
            f"'{_path_label(plan.path, leaf)}' uses flattened_range. Use file-backed-streaming."
        )
    sharded_key = getattr(leaf, "key", None)
    if not sharded_key:
        raise WeightedMergeError(
            f"direct-dcp-streaming requires a sharded key for '{_path_label(plan.path, leaf)}'."
        )


def _build_direct_dcp_write_specs(
    *,
    tensor_plans: list[_StreamingTensorPlan],
    extra_paths: list[tuple[Union[str, int], ...]],
    initial_template: ShardedStateDict,
    tensor_metadata_by_checkpoint: list[dict[str, Any]],
    extra_state_source_index: int,
    streaming_chunk_bytes: int,
) -> tuple[list[_DirectDcpWriteSpec], int]:
    write_specs: list[_DirectDcpWriteSpec] = []
    merge_chunk_count = 0

    for plan in tensor_plans:
        _validate_direct_dcp_plan(plan)
        if plan.source_dtype is None or plan.source_shape is None:
            raise WeightedMergeError(
                f"Direct DCP tensor plan for '{_path_label(plan.path, plan.template_leaf)}' "
                "is missing source metadata."
            )
        sharded_key = str(getattr(plan.template_leaf, "key"))
        for chunk_dim, chunk_start, chunk_length in _chunk_ranges_for_local_shape(
            plan.template_shape, streaming_chunk_bytes
        ):
            if chunk_length == 0:
                continue
            chunk_shape = list(plan.template_shape)
            if chunk_dim >= 0:
                chunk_shape[chunk_dim] = chunk_length
            chunk_shape_tuple = tuple(int(dim) for dim in chunk_shape)
            write_specs.append(
                _DirectDcpWriteSpec(
                    path=plan.path,
                    template_leaf=plan.template_leaf,
                    sharded_key=sharded_key,
                    global_shape=tuple(int(dim) for dim in plan.source_shape),
                    global_offsets=tuple(
                        int(offset)
                        for offset in _direct_dcp_global_offsets(
                            plan.template_leaf,
                            chunk_dim=chunk_dim,
                            chunk_start=chunk_start,
                        )
                    ),
                    chunk_shape=tuple(
                        int(size) for size in _direct_dcp_chunk_sizes(chunk_shape_tuple)
                    ),
                    target_dtype=plan.target_dtype,
                    chunk_dim=chunk_dim,
                    chunk_start=chunk_start,
                    chunk_length=chunk_length,
                    source_dtype=plan.source_dtype,
                    source_shape=plan.source_shape,
                )
            )
            merge_chunk_count += 1

    extra_metadata = tensor_metadata_by_checkpoint[extra_state_source_index]
    for path in extra_paths:
        template_leaf = _get_path(initial_template, path)
        sharded_key = getattr(template_leaf, "key", None)
        if not sharded_key:
            raise WeightedMergeError(
                f"direct-dcp-streaming requires a sharded key for _extra_state "
                f"'{_path_label(path, template_leaf)}'."
            )
        template_tensor = _as_tensor(template_leaf)
        if template_tensor is None:
            raise WeightedMergeError(
                "direct-dcp-streaming currently supports tensor _extra_state leaves only; "
                f"'{_path_label(path, template_leaf)}' is {type(template_leaf).__name__}."
            )
        if sharded_key not in extra_metadata:
            raise WeightedMergeError(
                f"Checkpoint {extra_state_source_index} is missing tensor metadata for "
                f"_extra_state '{_path_label(path, template_leaf)}'."
            )
        metadata_entry = extra_metadata[sharded_key]
        metadata_shape = tuple(int(dim) for dim in metadata_entry.global_shape)
        expected_shape = _global_shape_tuple(template_leaf)
        if expected_shape is not None and metadata_shape != expected_shape:
            raise WeightedMergeError(
                f"Shape mismatch for _extra_state '{_path_label(path, template_leaf)}': "
                f"template expects global shape {expected_shape}, but checkpoint metadata "
                f"has {metadata_shape}."
            )
        chunk_shape = tuple(int(dim) for dim in template_tensor.shape)
        write_specs.append(
            _DirectDcpWriteSpec(
                path=path,
                template_leaf=template_leaf,
                sharded_key=str(sharded_key),
                global_shape=metadata_shape,
                global_offsets=tuple(
                    int(offset)
                    for offset in _direct_dcp_global_offsets(
                        template_leaf,
                        chunk_dim=-1,
                        chunk_start=0,
                    )
                ),
                chunk_shape=chunk_shape,
                target_dtype=metadata_entry.dtype,
                chunk_dim=-1,
                chunk_start=0,
                chunk_length=1,
                source_dtype=metadata_entry.dtype,
                source_shape=metadata_shape,
                is_extra_state=True,
            )
        )

    return write_specs, merge_chunk_count


def _merge_direct_dcp_streaming(
    *,
    resolved_input_dirs: list[Path],
    weights: list[float],
    output_dir: Path,
    initial_template: ShardedStateDict,
    merge_paths: list[tuple[Union[str, int], ...]],
    extra_paths: list[tuple[Union[str, int], ...]],
    common_state: dict[str, Any],
    save_dtype: str,
    extra_state_source_index: int,
    strict: StrictHandling,
    validate_access_integrity: bool,
    byte_accounting: str,
    streaming_chunk_bytes: int,
) -> tuple[float, float, float, int]:
    """Stream merged chunks directly into PyTorch DCP storage and metadata."""

    from megatron.core.dist_checkpointing.core import CheckpointingConfig, save_config
    from megatron.core.dist_checkpointing.strategies.common import save_common

    output_dir.mkdir(parents=True, exist_ok=True)
    streaming_load_strategies = {
        checkpoint_dir: TorchDistLoadShardedStrategy(cache_metadata=True)
        for checkpoint_dir in resolved_input_dirs
    }
    _validate_streaming_load_contract(
        initial_template=initial_template,
        resolved_input_dirs=resolved_input_dirs,
        load_strategies=streaming_load_strategies,
        strict=strict,
        validate_access_integrity=validate_access_integrity,
    )

    tensor_metadata_by_checkpoint = []
    bytes_read = 0
    for checkpoint_dir in resolved_input_dirs:
        tensor_metadata_by_checkpoint.append(
            dist_checkpointing.load_tensors_metadata(
                str(checkpoint_dir), streaming_load_strategies[checkpoint_dir]
            )
        )
        bytes_read += _directory_size_for_accounting(checkpoint_dir, byte_accounting)

    target_dtype_override = SAVE_DTYPE_MAP.get(save_dtype)
    tensor_plans = [
        _build_streaming_tensor_plan(
            initial_template=initial_template,
            path=path,
            tensor_index=tensor_index,
            tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
            resolved_input_dirs=resolved_input_dirs,
            save_dtype=save_dtype,
            target_dtype_override=target_dtype_override,
        )
        for tensor_index, path in enumerate(merge_paths)
    ]
    write_specs, tensor_chunk_count = _build_direct_dcp_write_specs(
        tensor_plans=tensor_plans,
        extra_paths=extra_paths,
        initial_template=initial_template,
        tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
        extra_state_source_index=extra_state_source_index,
        streaming_chunk_bytes=streaming_chunk_bytes,
    )

    planner = _WeightedMergeDirectOutputSavePlanner(
        write_specs=write_specs,
        resolved_input_dirs=resolved_input_dirs,
        weights=weights,
        load_strategies=streaming_load_strategies,
        extra_state_source_index=extra_state_source_index,
    )
    writer = FileSystemWriter(
        output_dir,
        single_file_per_rank=True,
        sync_files=True,
        thread_count=1,
        per_thread_copy_ahead=0,
    )
    no_dist = _direct_dcp_save_uses_no_dist()
    dcp_save_start = time.perf_counter()
    try:
        torch_dcp.save({}, storage_writer=writer, planner=planner, no_dist=no_dist)
    except Exception as exc:
        rank_suffix = (
            f" on rank {dist.get_rank()}"
            if dist.is_available() and dist.is_initialized()
            else ""
        )
        raise WeightedMergeError(
            f"Direct DCP streaming save failed{rank_suffix}: {type(exc).__name__}: {exc}"
        ) from exc
    dcp_save_wall_time = time.perf_counter() - dcp_save_start
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    sidecar_start = time.perf_counter()

    def write_sidecars() -> None:
        save_common(common_state, str(output_dir))
        save_config(CheckpointingConfig("torch_dist", 1), str(output_dir))

    _run_rank0_filesystem_op("Writing Megatron checkpoint sidecars", write_sidecars)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    sidecar_time = time.perf_counter() - sidecar_start

    load_time = planner.load_time
    accumulation_time = planner.accumulation_time
    save_time = max(0.0, dcp_save_wall_time - load_time - accumulation_time) + sidecar_time
    print_rank_0(
        "Direct DCP streaming wrote "
        f"{len(tensor_plans)} tensors as {tensor_chunk_count} chunks with "
        f"chunk_budget={_format_bytes(streaming_chunk_bytes)}, "
        f"max_resolved_output={_format_bytes(planner.max_resolved_output_tensor_bytes)}. "
        "Output writing uses public DCP FileSystemWriter; source read volume follows "
        "source checkpoint storage-record size, not the requested chunk size.",
        flush=True,
    )
    return load_time, accumulation_time, save_time, bytes_read


def _streaming_work_groups(
    work_items: list[_StreamingWorkItem], streaming_chunk_bytes: int
) -> list[list[_StreamingWorkItem]]:
    groups: list[list[_StreamingWorkItem]] = []
    current_group: list[_StreamingWorkItem] = []
    current_bytes = 0
    current_paths: set[tuple[Union[str, int], ...]] = set()

    for item in work_items:
        would_exceed_budget = (
            current_group and current_bytes + item.accumulator_bytes > streaming_chunk_bytes
        )
        would_duplicate_path = item.load_path in current_paths
        if would_exceed_budget or would_duplicate_path:
            groups.append(current_group)
            current_group = []
            current_bytes = 0
            current_paths = set()
        current_group.append(item)
        current_bytes += item.accumulator_bytes
        current_paths.add(item.load_path)

    if current_group:
        groups.append(current_group)
    return groups


def _streaming_source_metadata(
    *,
    leaf: Any,
    label: str,
    tensor_metadata_by_checkpoint: list[dict[str, Any]],
    resolved_input_dirs: list[Path],
    save_dtype: str,
) -> tuple[torch.dtype, tuple[int, ...]]:
    sharded_key = getattr(leaf, "key", None)
    if sharded_key is None:
        raise WeightedMergeError(f"Template key '{label}' does not expose a sharded key.")

    source_dtype: Optional[torch.dtype] = None
    source_shape: Optional[tuple[int, ...]] = None
    for checkpoint_dir, tensor_metadata in zip(resolved_input_dirs, tensor_metadata_by_checkpoint):
        if sharded_key not in tensor_metadata:
            raise WeightedMergeError(
                f"Checkpoint {checkpoint_dir} is missing tensor metadata for '{label}' "
                f"(sharded key '{sharded_key}')."
            )
        metadata_entry = tensor_metadata[sharded_key]
        metadata_shape = tuple(metadata_entry.global_shape)
        if source_shape is None:
            source_shape = metadata_shape
            source_dtype = metadata_entry.dtype
            continue
        if metadata_shape != source_shape:
            raise WeightedMergeError(
                f"Shape mismatch for '{label}': expected {source_shape}, "
                f"got {metadata_shape} in {checkpoint_dir}."
            )
        if save_dtype == "same" and metadata_entry.dtype != source_dtype:
            raise WeightedMergeError(
                f"Dtype mismatch for '{label}' with --merge-save-dtype=same: "
                f"expected {source_dtype}, got {metadata_entry.dtype} in {checkpoint_dir}."
            )

    assert source_dtype is not None
    assert source_shape is not None
    expected_shape = _global_shape_tuple(leaf)
    if expected_shape is not None and source_shape != expected_shape:
        raise WeightedMergeError(
            f"Shape mismatch for '{label}': template expects global shape "
            f"{expected_shape}, but checkpoint metadata has {source_shape}."
        )
    return source_dtype, source_shape


def _global_shape_tuple(leaf: Any) -> Optional[tuple[int, ...]]:
    global_shape = getattr(leaf, "global_shape", None)
    if global_shape is None:
        return None
    return tuple(int(dim) for dim in global_shape)


def _streaming_factory_component_load_path(
    factory_path: tuple[Union[str, int], ...],
    component_index: int,
) -> tuple[Union[str, int], ...]:
    return (
        factory_path
        + ("__streaming_factory_components__", f"component_{component_index:06d}")
    )


def _build_streaming_tensor_plan(
    *,
    initial_template: ShardedStateDict,
    path: tuple[Union[str, int], ...],
    tensor_index: int,
    tensor_metadata_by_checkpoint: list[dict[str, Any]],
    resolved_input_dirs: list[Path],
    save_dtype: str,
    target_dtype_override: Optional[torch.dtype],
) -> _StreamingTensorPlan:
    template_leaf = _get_path(initial_template, path)
    template_tensor = _as_tensor(template_leaf)
    if template_tensor is None:
        raise WeightedMergeError(f"Template key '{_path_label(path)}' is not a tensor.")
    if not template_tensor.is_floating_point():
        raise WeightedMergeError(
            f"Template key '{_path_label(path)}' has non-floating dtype "
            f"{template_tensor.dtype}; weighted averaging is only supported for floating tensors."
        )

    template_shape = tuple(int(dim) for dim in template_tensor.shape)
    template_dtype = template_tensor.dtype
    if isinstance(template_leaf, ShardedTensorFactory):
        template_components = list(_flatten_items(template_leaf.build()))
        component_plans: list[_StreamingFactoryComponentPlan] = []
        target_dtype = target_dtype_override if target_dtype_override is not None else template_dtype
        for component_path, component_leaf in template_components:
            source_dtype, source_shape = _streaming_source_metadata(
                leaf=component_leaf,
                label=_path_label(path + component_path, component_leaf),
                tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
                resolved_input_dirs=resolved_input_dirs,
                save_dtype=save_dtype,
            )
            component_plans.append(
                _StreamingFactoryComponentPlan(
                    component_path=component_path,
                    component_leaf=component_leaf,
                    source_dtype=source_dtype,
                    source_shape=source_shape,
                )
            )
        if target_dtype is None:
            raise WeightedMergeError(
                f"Factory '{_path_label(path, template_leaf)}' produced no tensor components."
            )
        return _StreamingTensorPlan(
            path=path,
            tensor_index=tensor_index,
            template_leaf=template_leaf,
            template_shape=template_shape,
            target_dtype=target_dtype,
            factory_components=component_plans,
        )

    source_dtype, source_shape = _streaming_source_metadata(
        leaf=template_leaf,
        label=_path_label(path, template_leaf),
        tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
        resolved_input_dirs=resolved_input_dirs,
        save_dtype=save_dtype,
    )
    target_dtype = target_dtype_override if target_dtype_override is not None else template_dtype
    return _StreamingTensorPlan(
        path=path,
        tensor_index=tensor_index,
        template_leaf=template_leaf,
        template_shape=template_shape,
        target_dtype=target_dtype,
        source_dtype=source_dtype,
        source_shape=source_shape,
    )


def _merge_file_backed_streaming(
    *,
    resolved_input_dirs: list[Path],
    weights: list[float],
    initial_template: ShardedStateDict,
    merge_paths: list[tuple[Union[str, int], ...]],
    extra_paths: list[tuple[Union[str, int], ...]],
    sharded_keys: dict[tuple[Union[str, int], ...], str],
    common_state: dict[str, Any],
    save_dtype: str,
    extra_state_source_index: int,
    strict: StrictHandling,
    validate_access_integrity: bool,
    byte_accounting: str,
    staging_dir: Path,
    streaming_chunk_bytes: int,
    file_backed_staging_layout: str,
) -> tuple[ShardedStateDict, float, float, int, Optional[Any]]:
    streaming_prepare_start = time.perf_counter()
    target_dtype_override = SAVE_DTYPE_MAP.get(save_dtype)
    merged_state_dict = _clone_sharded_template_without_data(initial_template)
    streaming_load_strategies = {
        checkpoint_dir: TorchDistLoadShardedStrategy(cache_metadata=True)
        for checkpoint_dir in resolved_input_dirs
    }
    _validate_streaming_load_contract(
        initial_template=initial_template,
        resolved_input_dirs=resolved_input_dirs,
        load_strategies=streaming_load_strategies,
        strict=strict,
        validate_access_integrity=validate_access_integrity,
    )

    tensor_metadata_by_checkpoint: list[dict[str, Any]] = []
    bytes_read = 0
    load_time = 0.0
    accumulation_time = 0.0
    extra_state_values: dict[tuple[Union[str, int], ...], Any] = {}

    for checkpoint_dir in resolved_input_dirs:
        tensor_metadata_by_checkpoint.append(
            dist_checkpointing.load_tensors_metadata(
                str(checkpoint_dir), streaming_load_strategies[checkpoint_dir]
            )
        )
        bytes_read += _directory_size_for_accounting(checkpoint_dir, byte_accounting)

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    dtype_numels: dict[torch.dtype, int] = {}

    for tensor_index, path in enumerate(merge_paths):
        plan = _build_streaming_tensor_plan(
            initial_template=initial_template,
            path=path,
            tensor_index=tensor_index,
            tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
            resolved_input_dirs=resolved_input_dirs,
            save_dtype=save_dtype,
            target_dtype_override=target_dtype_override,
        )
        dtype_numels[plan.target_dtype] = dtype_numels.get(
            plan.target_dtype, 0
        ) + _tensor_numel(plan.template_shape)

    staging_bytes = sum(numel * _dtype_size(dtype) for dtype, numel in dtype_numels.items())
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_store = _FileBackedTensorStore(
        staging_dir, rank, dtype_numels, layout=file_backed_staging_layout
    )
    streaming_setup_time = time.perf_counter() - streaming_prepare_start
    if staging_store.layout == "per-tensor":
        print_rank_0(
            "Prepared per-tensor file-backed staging store for "
            f"{len(merge_paths)} streaming tensors on each rank in {streaming_setup_time:.2f}s "
            f"(files created lazily, local_staging={_format_bytes(staging_bytes)})",
            flush=True,
        )
    else:
        print_rank_0(
            f"Prepared {staging_store.file_count} file-backed staging files for "
            f"{len(merge_paths)} streaming tensors on each rank in {streaming_setup_time:.2f}s "
            f"(layout={staging_store.layout}, local_staging={_format_bytes(staging_bytes)})",
            flush=True,
        )

    current_group: list[_StreamingWorkItem] = []
    current_group_bytes = 0
    current_group_paths: set[tuple[Union[str, int], ...]] = set()
    work_item_count = 0
    work_group_count = 0
    streaming_stream_start = time.perf_counter()

    def process_work_group(work_group: list[_StreamingWorkItem]) -> None:
        nonlocal load_time, accumulation_time, work_group_count
        if not work_group:
            return
        work_group_count += 1
        if work_group_count == 1 or work_group_count <= 10 or work_group_count % 25 == 0:
            staging_file_text = (
                f", staging_files={staging_store.file_count}"
                if staging_store.layout == "per-tensor"
                else ""
            )
            print_rank_0(
                f"Streaming DCP load group {work_group_count} items={len(work_group)}"
                f"{staging_file_text}",
                flush=True,
            )
        accumulators = [
            torch.zeros(item.chunk_shape, dtype=torch.float32, device="cpu")
            for item in work_group
        ]
        path_leaves = [(item.load_path, item.load_leaf) for item in work_group]
        for checkpoint_dir, weight in zip(resolved_input_dirs, weights):
            load_start = time.perf_counter()
            loaded_by_path = _load_tensor_path_group_fast(
                checkpoint_dir,
                path_leaves,
                streaming_load_strategies[checkpoint_dir],
            )
            load_time += time.perf_counter() - load_start
            for item, accumulator in zip(work_group, accumulators):
                tensor = _as_tensor(loaded_by_path[item.load_path])
                if tensor is None:
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(item.merge_path)}' "
                        "is not a tensor."
                    )
                if not tensor.is_floating_point():
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(item.merge_path)}' "
                        f"has non-floating dtype {tensor.dtype}; weighted averaging is only "
                        "supported for floating tensors."
                    )
                if tuple(tensor.shape) != item.chunk_shape:
                    if tensor.numel() != accumulator.numel():
                        raise WeightedMergeError(
                            f"Checkpoint {checkpoint_dir} key '{_path_label(item.merge_path)}' "
                            f"loaded chunk shape {tuple(tensor.shape)} cannot be reshaped to "
                            f"{item.chunk_shape}."
                        )
                    tensor = tensor.reshape(item.chunk_shape)
                accumulation_start = time.perf_counter()
                accumulator.add_(tensor.detach().to(dtype=torch.float32, device="cpu"), alpha=weight)
                accumulation_time += time.perf_counter() - accumulation_start

        for item, accumulator in zip(work_group, accumulators):
            output_chunk = accumulator.to(dtype=item.target_tensor.dtype)
            if item.chunk_dim < 0:
                target_chunk = item.target_tensor
            else:
                target_chunk = item.target_tensor.narrow(
                    item.chunk_dim, item.chunk_start, item.chunk_length
                )
            target_chunk.copy_(output_chunk)

    def flush_work_group() -> None:
        nonlocal current_group, current_group_bytes, current_group_paths
        if not current_group:
            return
        work_group = current_group
        current_group = []
        current_group_bytes = 0
        current_group_paths = set()
        process_work_group(work_group)

    def queue_work_item(item: _StreamingWorkItem) -> None:
        nonlocal current_group_bytes, work_item_count
        would_exceed_budget = (
            current_group and current_group_bytes + item.accumulator_bytes > streaming_chunk_bytes
        )
        would_duplicate_path = item.load_path in current_group_paths
        if would_exceed_budget or would_duplicate_path:
            flush_work_group()
        current_group.append(item)
        current_group_bytes += item.accumulator_bytes
        current_group_paths.add(item.load_path)
        work_item_count += 1

    def append_leaf_work_items(
        *,
        merge_path: tuple[Union[str, int], ...],
        load_path: tuple[Union[str, int], ...],
        tensor_index: int,
        template_leaf: Any,
        target_tensor: torch.Tensor,
        source_dtype: torch.dtype,
        source_shape: tuple[int, ...],
    ) -> int:
        chunk_ranges = _chunk_ranges_for_local_shape(tuple(target_tensor.shape), streaming_chunk_bytes)
        appended_chunks = 0
        for chunk_dim, chunk_start, chunk_length in chunk_ranges:
            if chunk_length == 0:
                continue
            chunk_shape = list(target_tensor.shape)
            if chunk_dim >= 0:
                chunk_shape[chunk_dim] = chunk_length
            chunk_shape = tuple(int(dim) for dim in chunk_shape)
            queue_work_item(
                _StreamingWorkItem(
                    merge_path=merge_path,
                    load_path=load_path,
                    tensor_index=tensor_index,
                    load_leaf=_streaming_chunk_leaf(
                        template_leaf,
                        chunk_dim=chunk_dim,
                        chunk_start=chunk_start,
                        chunk_length=chunk_length,
                        dtype=source_dtype,
                        global_shape=source_shape,
                    ),
                    target_tensor=target_tensor,
                    chunk_dim=chunk_dim,
                    chunk_start=chunk_start,
                    chunk_length=chunk_length,
                    chunk_shape=chunk_shape,
                    accumulator_bytes=_tensor_numel(chunk_shape) * _dtype_size(torch.float32),
                )
            )
            appended_chunks += 1
        return appended_chunks

    for tensor_index, path in enumerate(merge_paths):
        plan = _build_streaming_tensor_plan(
            initial_template=initial_template,
            path=path,
            tensor_index=tensor_index,
            tensor_metadata_by_checkpoint=tensor_metadata_by_checkpoint,
            resolved_input_dirs=resolved_input_dirs,
            save_dtype=save_dtype,
            target_dtype_override=target_dtype_override,
        )
        total_chunks_for_tensor = 0
        if plan.factory_components is not None:
            staged_tensor = staging_store.allocate(
                plan.template_shape, plan.target_dtype, _path_label(plan.path, plan.template_leaf)
            )
            output_leaf = _get_path(merged_state_dict, plan.path)
            _assign_leaf_data(output_leaf, staged_tensor)
            staged_components = list(_flatten_items(output_leaf.build()))
            if len(staged_components) != len(plan.factory_components):
                raise WeightedMergeError(
                    f"Factory '{_path_label(plan.path, plan.template_leaf)}' produced "
                    f"{len(plan.factory_components)} template components but "
                    f"{len(staged_components)} staged components."
                )
            for component_index, (component_plan, (_, staged_component)) in enumerate(
                zip(plan.factory_components, staged_components)
            ):
                target_component_tensor = _as_tensor(staged_component)
                if target_component_tensor is None:
                    raise WeightedMergeError(
                        f"Factory component "
                        f"'{_path_label(plan.path + component_plan.component_path, component_plan.component_leaf)}' "
                        "does not expose tensor data."
                    )
                total_chunks_for_tensor += append_leaf_work_items(
                    merge_path=plan.path,
                    load_path=_streaming_factory_component_load_path(plan.path, component_index),
                    tensor_index=plan.tensor_index,
                    template_leaf=component_plan.component_leaf,
                    target_tensor=target_component_tensor,
                    source_dtype=component_plan.source_dtype,
                    source_shape=component_plan.source_shape,
                )
        else:
            if plan.source_dtype is None or plan.source_shape is None:
                raise WeightedMergeError(
                    f"Streaming tensor plan for '{_path_label(plan.path, plan.template_leaf)}' "
                    "is missing source metadata."
                )
            staged_tensor = staging_store.allocate(
                plan.template_shape, plan.target_dtype, _path_label(plan.path, plan.template_leaf)
            )
            output_leaf = _get_path(merged_state_dict, plan.path)
            _assign_leaf_data(output_leaf, staged_tensor)
            if hasattr(output_leaf, "dtype"):
                output_leaf.dtype = plan.target_dtype
            total_chunks_for_tensor = append_leaf_work_items(
                merge_path=plan.path,
                load_path=plan.path,
                tensor_index=plan.tensor_index,
                template_leaf=plan.template_leaf,
                target_tensor=staged_tensor,
                source_dtype=plan.source_dtype,
                source_shape=plan.source_shape,
            )

        if (
            total_chunks_for_tensor > 1
            or plan.tensor_index < 20
            or (plan.tensor_index + 1) % 1000 == 0
        ):
            print_rank_0(
                f"Queued streaming tensor {plan.tensor_index + 1}/{len(merge_paths)}: "
                f"{_path_label(plan.path, plan.template_leaf)} chunks={total_chunks_for_tensor}",
                flush=True,
            )

    flush_work_group()
    streaming_stream_time = time.perf_counter() - streaming_stream_start
    print_rank_0(
        f"Streamed file-backed work plan in {streaming_stream_time:.2f}s: "
        f"{len(merge_paths)} tensors as {work_item_count} chunks in "
        f"{work_group_count} DCP load groups with "
        f"chunk_budget={_format_bytes(streaming_chunk_bytes)}, "
        f"layout={staging_store.layout}, staging_files={staging_store.file_count}, "
        f"local_staging={_format_bytes(staging_bytes)}",
        flush=True,
    )

    extra_path_leaves: list[tuple[tuple[Union[str, int], ...], Any]] = []
    for path in extra_paths:
        template_leaf = _get_path(initial_template, path)
        load_leaf = copy.deepcopy(template_leaf)
        tensor = _as_tensor(load_leaf)
        if tensor is not None:
            _assign_leaf_data(load_leaf, torch.empty_like(tensor, device="cpu"))
        extra_path_leaves.append((path, load_leaf))

    if extra_path_leaves:
        # The full template strict/access check above already validated mismatch
        # behavior. This extra-state copy intentionally loads only a partial tree.
        loaded_extras = _load_path_group(
            resolved_input_dirs[extra_state_source_index],
            extra_path_leaves,
            strict=StrictHandling.ASSUME_OK_UNEXPECTED,
        )
        for path in extra_paths:
            extra_state_values[path] = _copy_loaded_value(loaded_extras[path])

    if len(extra_state_values) != len(extra_paths):
        missing = len(extra_paths) - len(extra_state_values)
        raise WeightedMergeError(
            f"Failed to copy {missing} _extra_state entries from source checkpoint."
        )

    for path, value in extra_state_values.items():
        leaf = _get_path(merged_state_dict, path)
        _assign_leaf_data(leaf, value)

    for key, value in common_state.items():
        merged_state_dict[key] = value

    source_save_metadata = streaming_load_strategies[
        resolved_input_dirs[0]
    ].cached_global_metadata
    return merged_state_dict, load_time, accumulation_time, bytes_read, source_save_metadata


def merge_sharded_checkpoints(
    input_paths: list[Union[str, Path]],
    weights: list[float],
    output_root: Union[str, Path],
    sharded_state_dict_factory: Callable[[], ShardedStateDict],
    *,
    normalize: bool = False,
    save_dtype: str = "same",
    output_iteration: Optional[int] = None,
    write_latest: bool = True,
    extra_state_source_index: int = 0,
    strict: Union[str, StrictHandling] = StrictHandling.RAISE_UNEXPECTED,
    validate_access_integrity: bool = True,
    model_init_time: float = 0.0,
    verify_load: bool = False,
    execution_mode: str = "baseline",
    byte_accounting: str = "rank0",
    overwrite_output: bool = False,
    atomic_output: bool = True,
    streaming_chunk_bytes: int = 128 * 1024 * 1024,
    staging_dir: Optional[Union[str, Path]] = None,
    merge_style: Optional[str] = None,
    resource_log: str = "none",
    max_projected_cpu_bytes: Optional[int] = None,
    max_file_backed_staging_bytes: Optional[int] = None,
    max_file_backed_staging_files: Optional[int] = None,
    reuse_source_metadata_for_save: bool = False,
    file_backed_staging_layout: str = "shared-dtype",
    preflight_only: bool = False,
) -> MergeResult:
    """Merge distributed checkpoints using a caller-provided sharded template."""

    total_start = time.perf_counter()
    discovery_start = time.perf_counter()
    ensure_process_group()

    if len(input_paths) != len(weights):
        raise WeightedMergeError(f"Got {len(input_paths)} input paths but {len(weights)} weights.")
    if not input_paths:
        raise WeightedMergeError("At least one input checkpoint is required.")
    if save_dtype != "same" and save_dtype not in SAVE_DTYPE_MAP:
        raise WeightedMergeError(
            f"Unsupported save dtype '{save_dtype}'. Use same, float32, float16, or bfloat16."
        )
    if execution_mode not in MERGE_EXECUTION_MODES:
        raise WeightedMergeError(
            f"Unsupported execution mode '{execution_mode}'. "
            f"Use one of: {', '.join(MERGE_EXECUTION_MODES)}."
        )
    if byte_accounting not in BYTE_ACCOUNTING_MODES:
        raise WeightedMergeError(
            f"Unsupported byte accounting mode '{byte_accounting}'. "
            f"Use one of: {', '.join(BYTE_ACCOUNTING_MODES)}."
        )
    if resource_log not in RESOURCE_LOG_MODES:
        raise WeightedMergeError(
            f"Unsupported resource log mode '{resource_log}'. "
            f"Use one of: {', '.join(RESOURCE_LOG_MODES)}."
        )
    if file_backed_staging_layout not in FILE_BACKED_STAGING_LAYOUTS:
        raise WeightedMergeError(
            f"Unsupported file-backed staging layout '{file_backed_staging_layout}'. "
            f"Use one of: {', '.join(FILE_BACKED_STAGING_LAYOUTS)}."
        )
    if max_projected_cpu_bytes is not None and max_projected_cpu_bytes < 1:
        raise WeightedMergeError(
            f"max_projected_cpu_bytes must be positive, got {max_projected_cpu_bytes}."
        )
    if max_file_backed_staging_bytes is not None and max_file_backed_staging_bytes < 1:
        raise WeightedMergeError(
            "max_file_backed_staging_bytes must be positive, "
            f"got {max_file_backed_staging_bytes}."
        )
    if max_file_backed_staging_files is not None and max_file_backed_staging_files < 1:
        raise WeightedMergeError(
            "max_file_backed_staging_files must be positive, "
            f"got {max_file_backed_staging_files}."
        )
    if preflight_only and verify_load:
        raise WeightedMergeError("preflight_only cannot be combined with verify_load.")
    if reuse_source_metadata_for_save and execution_mode != "file-backed-streaming":
        raise WeightedMergeError(
            "--merge-reuse-source-metadata-for-save is currently supported only with "
            "--merge-execution-mode=file-backed-streaming."
        )
    if reuse_source_metadata_for_save and save_dtype != "same":
        raise WeightedMergeError(
            "--merge-reuse-source-metadata-for-save requires --merge-save-dtype=same so "
            "source and output DCP tensor metadata can be reused safely."
        )
    if extra_state_source_index < 0 or extra_state_source_index >= len(input_paths):
        raise WeightedMergeError(
            f"extra_state_source_index {extra_state_source_index} is out of range."
        )
    strict = parse_strict_flag(strict)
    if StrictHandling.requires_returning_mismatch_keys(strict):
        raise WeightedMergeError(
            f"strict={strict.value} is not supported by weighted merge because it changes "
            "dist_checkpointing.load() return type."
        )

    resolved_input_dirs = [resolve_checkpoint_dir(path) for path in input_paths]
    input_formats = [_checkpoint_format(path) for path in resolved_input_dirs]
    first_format = input_formats[0]
    for checkpoint_dir, checkpoint_format in zip(resolved_input_dirs[1:], input_formats[1:]):
        if checkpoint_format != first_format:
            raise WeightedMergeError(
                f"Checkpoint format mismatch: expected {first_format}, "
                f"got {checkpoint_format} in {checkpoint_dir}."
            )
    if first_format not in SUPPORTED_INPUT_BACKENDS:
        raise WeightedMergeError(
            f"Unsupported checkpoint format '{first_format}'. Weighted merge currently supports "
            f"{', '.join(SUPPORTED_INPUT_BACKENDS)}; run fsdp_dtensor as an explicit compatibility "
            "experiment before claiming support."
        )
    _log_resource_checkpoint(
        f"before initial template extraction execution_mode={execution_mode}",
        start_time=total_start,
        resource_log=resource_log,
    )
    template_start = time.perf_counter()
    initial_template = _state_dict_for_execution_mode(sharded_state_dict_factory, execution_mode)
    _log_resource_checkpoint(
        "after initial template extraction",
        start_time=template_start,
        resource_log=resource_log,
    )
    classify_start = time.perf_counter()
    merge_paths, extra_paths = _classify_template(initial_template)
    _log_resource_checkpoint(
        f"after template classification mergeable={len(merge_paths)} extra_state={len(extra_paths)}",
        start_time=classify_start,
        resource_log=resource_log,
    )
    key_index_start = time.perf_counter()
    sharded_keys = _sharded_keys_by_path(initial_template)
    _log_resource_checkpoint(
        f"after sharded-key indexing keys={len(sharded_keys)}",
        start_time=key_index_start,
        resource_log=resource_log,
    )
    estimate_start = time.perf_counter()
    memory_estimate = _estimate_whole_shard_memory(
        initial_template,
        merge_paths,
        extra_paths,
        save_dtype=save_dtype,
        execution_mode=execution_mode,
        streaming_chunk_bytes=streaming_chunk_bytes,
        file_backed_staging_layout=file_backed_staging_layout,
    )
    _log_resource_checkpoint(
        "after memory estimate",
        start_time=estimate_start,
        resource_log=resource_log,
    )
    weights = normalize_weights(weights) if normalize else validate_weights(weights)
    output_dir = output_checkpoint_dir(output_root, output_iteration)
    if not atomic_output and output_dir.exists() and not overwrite_output:
        raise WeightedMergeError(
            f"Output directory already exists: {output_dir}. "
            "Use --overwrite-merge-output to replace it."
        )
    discovery_time = time.perf_counter() - discovery_start

    action = "Preflighting merge of" if preflight_only else "Merging"
    print_rank_0(
        f"{action} {len(resolved_input_dirs)} checkpoints into {output_dir} "
        f"with weights {weights} using execution_mode={execution_mode}",
        flush=True,
    )
    _log_memory_estimate(memory_estimate)
    _enforce_projected_cpu_memory_guard(memory_estimate, max_projected_cpu_bytes)
    _enforce_file_backed_staging_guard(memory_estimate, max_file_backed_staging_bytes)
    _enforce_file_backed_staging_file_guard(
        memory_estimate, max_file_backed_staging_files
    )
    if preflight_only:
        host_peak_bytes = _host_peak_memory_bytes()
        gpu_peak_bytes = _gpu_peak_memory_bytes()
        (
            rank,
            world_size,
            max_host_peak_rank,
            max_host_peak_bytes,
            max_gpu_peak_rank,
            max_gpu_peak_bytes,
        ) = _distributed_memory_peaks(host_peak_bytes, gpu_peak_bytes)
        timings = MergeTimings(
            discovery=discovery_time,
            model_init=model_init_time,
            total=time.perf_counter() - total_start,
        )
        return MergeResult(
            output_dir=output_dir,
            input_dirs=tuple(resolved_input_dirs),
            weights=tuple(weights),
            timings=timings,
            averaged_tensors=len(merge_paths),
            copied_extra_states=len(extra_paths),
            backend=first_format,
            host_peak_bytes=host_peak_bytes,
            gpu_peak_bytes=gpu_peak_bytes,
            max_host_peak_bytes=max_host_peak_bytes,
            max_host_peak_rank=max_host_peak_rank,
            max_gpu_peak_bytes=max_gpu_peak_bytes,
            max_gpu_peak_rank=max_gpu_peak_rank,
            world_size=world_size,
            rank=rank,
            implementation_mode=execution_mode,
            memory_estimate=memory_estimate,
            save_metadata_cache_requested=reuse_source_metadata_for_save,
            file_backed_staging_layout=file_backed_staging_layout,
            preflight_only=True,
        )

    temporary_output_dir = (
        _prepare_temporary_output_dir(output_dir, overwrite_output)
        if atomic_output
        else output_dir
    )
    streaming_staging_dir = (
        Path(staging_dir)
        if staging_dir is not None
        else temporary_output_dir.with_name(f".{temporary_output_dir.name}.staging")
    )

    base_common_state = dist_checkpointing.load_common_state_dict(str(resolved_input_dirs[0]))
    common_state = _prepare_common_state(base_common_state, output_iteration)
    _add_merge_provenance(
        common_state,
        input_dirs=resolved_input_dirs,
        weights=weights,
        normalize=normalize,
        save_dtype=save_dtype,
        output_iteration=output_iteration,
        extra_state_source_index=extra_state_source_index,
        strict=strict,
        execution_mode=execution_mode,
        merge_style=merge_style,
        file_backed_staging_layout=file_backed_staging_layout,
    )

    direct_dcp_save_completed = False
    direct_dcp_save_time = 0.0
    if execution_mode == "direct-dcp-streaming":
        if overwrite_output and not atomic_output and output_dir.exists():
            _run_rank0_filesystem_op(
                f"Removing existing output directory {output_dir}",
                lambda: shutil.rmtree(output_dir),
            )
            if dist.is_initialized():
                dist.barrier()
        (
            load_time,
            accumulation_time,
            direct_dcp_save_time,
            bytes_read,
        ) = _merge_direct_dcp_streaming(
            resolved_input_dirs=resolved_input_dirs,
            weights=weights,
            output_dir=temporary_output_dir,
            initial_template=initial_template,
            merge_paths=merge_paths,
            extra_paths=extra_paths,
            common_state=common_state,
            save_dtype=save_dtype,
            extra_state_source_index=extra_state_source_index,
            strict=strict,
            validate_access_integrity=validate_access_integrity,
            byte_accounting=byte_accounting,
            streaming_chunk_bytes=streaming_chunk_bytes,
        )
        source_save_metadata = None
        direct_dcp_save_completed = True
    elif execution_mode == "file-backed-streaming":
        (
            merged_state_dict,
            load_time,
            accumulation_time,
            bytes_read,
            source_save_metadata,
        ) = _merge_file_backed_streaming(
            resolved_input_dirs=resolved_input_dirs,
            weights=weights,
            initial_template=initial_template,
            merge_paths=merge_paths,
            extra_paths=extra_paths,
            sharded_keys=sharded_keys,
            common_state=common_state,
            save_dtype=save_dtype,
            extra_state_source_index=extra_state_source_index,
            strict=strict,
            validate_access_integrity=validate_access_integrity,
            byte_accounting=byte_accounting,
            staging_dir=streaming_staging_dir,
            streaming_chunk_bytes=streaming_chunk_bytes,
            file_backed_staging_layout=file_backed_staging_layout,
        )
    else:
        source_save_metadata = None
        accumulators: dict[tuple[Union[str, int], ...], torch.Tensor] = {}
        source_dtypes: dict[tuple[Union[str, int], ...], torch.dtype] = {}
        source_shapes: dict[tuple[Union[str, int], ...], torch.Size] = {}
        source_metadata_dtypes: dict[tuple[Union[str, int], ...], torch.dtype] = {}
        source_metadata_shapes: dict[tuple[Union[str, int], ...], tuple[int, ...]] = {}
        extra_state_values: dict[tuple[Union[str, int], ...], Any] = {}
        bytes_read = 0
        load_time = 0.0
        accumulation_time = 0.0

        for checkpoint_index, (checkpoint_dir, weight) in enumerate(zip(resolved_input_dirs, weights)):
            print_rank_0(
                f"Loading checkpoint {checkpoint_index + 1}/{len(resolved_input_dirs)}: {checkpoint_dir}",
                flush=True,
            )
            tensor_metadata = dist_checkpointing.load_tensors_metadata(str(checkpoint_dir))
            for path in merge_paths:
                sharded_key = sharded_keys.get(path)
                if sharded_key is None or sharded_key not in tensor_metadata:
                    continue
                metadata_entry = tensor_metadata[sharded_key]
                metadata_shape = tuple(metadata_entry.global_shape)
                if path not in source_metadata_shapes:
                    source_metadata_shapes[path] = metadata_shape
                    source_metadata_dtypes[path] = metadata_entry.dtype
                else:
                    if metadata_shape != source_metadata_shapes[path]:
                        raise WeightedMergeError(
                            f"Shape mismatch for '{_path_label(path)}': expected "
                            f"{source_metadata_shapes[path]}, got {metadata_shape} in {checkpoint_dir}."
                        )
                    if save_dtype == "same" and metadata_entry.dtype != source_metadata_dtypes[path]:
                        raise WeightedMergeError(
                            f"Dtype mismatch for '{_path_label(path)}' with --merge-save-dtype=same: "
                            f"expected {source_metadata_dtypes[path]}, got {metadata_entry.dtype} "
                            f"in {checkpoint_dir}."
                        )

            load_template = _state_dict_for_execution_mode(sharded_state_dict_factory, execution_mode)
            load_start = time.perf_counter()
            loaded_state = dist_checkpointing.load(
                load_template,
                str(checkpoint_dir),
                validate_access_integrity=validate_access_integrity,
                strict=strict,
            )
            load_time += time.perf_counter() - load_start
            bytes_read += _directory_size_for_accounting(checkpoint_dir, byte_accounting)

            loaded_by_path = dict(_flatten_items(loaded_state))
            accumulation_start = time.perf_counter()
            for path in merge_paths:
                if path not in loaded_by_path:
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} is missing key '{_path_label(path)}'."
                    )
                tensor = _as_tensor(loaded_by_path[path])
                if tensor is None:
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(path)}' is not a tensor."
                    )
                if not tensor.is_floating_point():
                    raise WeightedMergeError(
                        f"Checkpoint {checkpoint_dir} key '{_path_label(path)}' has non-floating "
                        f"dtype {tensor.dtype}; weighted averaging is only supported for floating tensors."
                    )

                if path not in accumulators:
                    accumulators[path] = torch.zeros(tensor.shape, dtype=torch.float32, device="cpu")
                    source_dtypes[path] = tensor.dtype
                    source_shapes[path] = tensor.shape
                else:
                    if tensor.shape != source_shapes[path]:
                        raise WeightedMergeError(
                            f"Shape mismatch for '{_path_label(path)}': expected "
                            f"{tuple(source_shapes[path])}, got {tuple(tensor.shape)} in {checkpoint_dir}."
                        )
                    if save_dtype == "same" and tensor.dtype != source_dtypes[path]:
                        raise WeightedMergeError(
                            f"Dtype mismatch for '{_path_label(path)}' with --merge-save-dtype=same: "
                            f"expected {source_dtypes[path]}, got {tensor.dtype} in {checkpoint_dir}."
                        )

                accumulators[path].add_(
                    tensor.detach().to(dtype=torch.float32, device="cpu"), alpha=weight
                )

            if checkpoint_index == extra_state_source_index:
                for path in extra_paths:
                    if path not in loaded_by_path:
                        raise WeightedMergeError(
                            f"Checkpoint {checkpoint_dir} is missing _extra_state key '{_path_label(path)}'."
                        )
                    extra_state_values[path] = _copy_loaded_value(loaded_by_path[path])

            accumulation_time += time.perf_counter() - accumulation_start
            del loaded_state
            if execution_mode == "baseline" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(extra_state_values) != len(extra_paths):
            missing = len(extra_paths) - len(extra_state_values)
            raise WeightedMergeError(
                f"Failed to copy {missing} _extra_state entries from source checkpoint."
            )

        merged_state_dict = _state_dict_for_execution_mode(sharded_state_dict_factory, execution_mode)
        target_dtype = SAVE_DTYPE_MAP.get(save_dtype)
        for path, accumulator in accumulators.items():
            leaf = _get_path(merged_state_dict, path)
            template_tensor = _as_tensor(leaf)
            device = (
                torch.device("cpu")
                if execution_mode == "cpu-resident"
                else template_tensor.device if template_tensor is not None else torch.device("cpu")
            )
            dtype = target_dtype if target_dtype is not None else source_dtypes[path]
            _assign_leaf_data(leaf, accumulator.to(device=device, dtype=dtype))

        for path, value in extra_state_values.items():
            leaf = _get_path(merged_state_dict, path)
            template_tensor = _as_tensor(leaf)
            if torch.is_tensor(value) and template_tensor is not None and execution_mode != "cpu-resident":
                value = value.to(device=template_tensor.device)
            _assign_leaf_data(leaf, value)

        for key, value in common_state.items():
            merged_state_dict[key] = value

    save_start = time.perf_counter()
    save_metadata_cache_reused = False
    if not direct_dcp_save_completed:
        save_strategy = _save_strategy_from_source_metadata(
            source_save_metadata, requested=reuse_source_metadata_for_save
        )
        if overwrite_output and not atomic_output and output_dir.exists():
            _run_rank0_filesystem_op(
                f"Removing existing output directory {output_dir}",
                lambda: shutil.rmtree(output_dir),
            )
            if dist.is_initialized():
                dist.barrier()
        temporary_output_dir.mkdir(parents=True, exist_ok=True)
        dist_checkpointing.save(
            merged_state_dict,
            str(temporary_output_dir),
            sharded_strategy=save_strategy,
            validate_access_integrity=validate_access_integrity,
        )
        save_metadata_cache_reused = _report_save_metadata_cache_reuse(
            save_strategy, requested=reuse_source_metadata_for_save
        )
    if dist.is_initialized():
        dist.barrier()
    if atomic_output:
        _publish_temporary_output_dir(
            temporary_output_dir, output_dir, overwrite_output=overwrite_output
        )
        if dist.is_initialized():
            dist.barrier()
    if output_iteration is not None and write_latest:
        _run_rank0_filesystem_op(
            f"Writing {LATEST_CHECKPOINTED_ITERATION}",
            lambda: write_latest_checkpointed_iteration(output_dir, output_iteration),
        )
    if dist.is_initialized():
        dist.barrier()
    save_time = time.perf_counter() - save_start
    if direct_dcp_save_completed:
        save_time += direct_dcp_save_time
    bytes_written = _directory_size_for_accounting(output_dir, byte_accounting)

    verification_time = 0.0
    if verify_load:
        verify_start = time.perf_counter()
        verification_state_dict = _state_dict_for_execution_mode(
            sharded_state_dict_factory, execution_mode
        )
        dist_checkpointing.load(
            verification_state_dict,
            str(output_dir),
            validate_access_integrity=validate_access_integrity,
            strict=strict,
        )
        if dist.is_initialized():
            dist.barrier()
        verification_time = time.perf_counter() - verify_start

    if execution_mode == "file-backed-streaming" and staging_dir is None:
        _run_rank0_filesystem_op(
            f"Removing merge staging directory {streaming_staging_dir}",
            lambda: shutil.rmtree(streaming_staging_dir, ignore_errors=True),
        )
        if dist.is_initialized():
            dist.barrier()

    host_peak_bytes = _host_peak_memory_bytes()
    gpu_peak_bytes = _gpu_peak_memory_bytes()
    (
        rank,
        world_size,
        max_host_peak_rank,
        max_host_peak_bytes,
        max_gpu_peak_rank,
        max_gpu_peak_bytes,
    ) = _distributed_memory_peaks(host_peak_bytes, gpu_peak_bytes)

    timings = MergeTimings(
        discovery=discovery_time,
        model_init=model_init_time,
        load=load_time,
        accumulation=accumulation_time,
        save=save_time,
        verification=verification_time,
        total=time.perf_counter() - total_start,
    )
    return MergeResult(
        output_dir=output_dir,
        input_dirs=tuple(resolved_input_dirs),
        weights=tuple(weights),
        timings=timings,
        averaged_tensors=len(merge_paths),
        copied_extra_states=len(extra_paths),
        bytes_read=bytes_read,
        bytes_written=bytes_written,
        backend=first_format,
        verified_load=verify_load,
        host_peak_bytes=host_peak_bytes,
        gpu_peak_bytes=gpu_peak_bytes,
        max_host_peak_bytes=max_host_peak_bytes,
        max_host_peak_rank=max_host_peak_rank,
        max_gpu_peak_bytes=max_gpu_peak_bytes,
        max_gpu_peak_rank=max_gpu_peak_rank,
        world_size=world_size,
        rank=rank,
        implementation_mode=execution_mode,
        memory_estimate=memory_estimate,
        save_metadata_cache_requested=reuse_source_metadata_for_save,
        save_metadata_cache_reused=save_metadata_cache_reused,
        file_backed_staging_layout=file_backed_staging_layout,
        preflight_only=False,
    )


def _determine_checkpoint_for_args(
    merge_inputs: list[str], start_checkpoint: Optional[int], end_checkpoint: Optional[int]
) -> tuple[Path, Optional[int]]:
    if not merge_inputs:
        raise WeightedMergeError("--merge-inputs is required.")

    first_input = merge_inputs[0]
    if ":" in first_input:
        path = resolve_checkpoint_dir(first_input.rsplit(":", 1)[0])
        match = ITERATION_RE.match(path.name)
        if match:
            return path.parent, int(match.group(1))
        if path.name == "release":
            return path.parent, None
        return path, None

    checkpoint_root = Path(first_input)
    if (
        end_checkpoint is not None
        and (checkpoint_root / iteration_dir_name(end_checkpoint)).is_dir()
    ):
        return checkpoint_root, end_checkpoint
    if (
        start_checkpoint is not None
        and (checkpoint_root / iteration_dir_name(start_checkpoint)).is_dir()
    ):
        return checkpoint_root, start_checkpoint
    if (checkpoint_root / LATEST_CHECKPOINTED_ITERATION).exists():
        latest = _read_latest_checkpointed_iteration(checkpoint_root)
        return checkpoint_root, None if latest == "release" else int(latest)

    iterations = discover_checkpoint_iterations(checkpoint_root)
    if not iterations:
        raise WeightedMergeError(f"No iter_* checkpoint directories found under {checkpoint_root}.")
    return checkpoint_root, iterations[0]


def add_merge_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="weighted checkpoint merge")
    group.add_argument(
        "--merge-inputs",
        nargs="+",
        required=True,
        help=(
            "Manual mode: PATH:WEIGHT entries. Window/range mode: one checkpoint root "
            "containing iter_* directories."
        ),
    )
    group.add_argument("--merge-output", required=True, help="Output checkpoint root.")
    group.add_argument("--normalize", action="store_true", help="Normalize manual weights.")
    group.add_argument("--start-checkpoint", type=int, help="Inclusive start iteration.")
    group.add_argument("--end-checkpoint", type=int, help="Inclusive target/end iteration.")
    group.add_argument(
        "--merge-window-btoks",
        type=int,
        help=(
            "Window size in billions of tokens. Requires --end-checkpoint and uses "
            "checkpoint seq_length/global_batch_size to derive the start iteration."
        ),
    )
    group.add_argument(
        "--merge-style",
        choices=get_valid_styles(),
        default="linear",
        help="Coefficient schedule for range/window mode.",
    )
    group.add_argument(
        "--coefficient-seed",
        type=int,
        default=0,
        help="Deterministic seed for '__scramble' coefficient styles.",
    )
    group.add_argument(
        "--min-iteration-interval",
        type=int,
        default=None,
        help="Only keep selected checkpoints separated by at least this many iterations.",
    )
    group.add_argument(
        "--min-checkpoints",
        type=int,
        default=None,
        help="Fail if fewer than this many input checkpoints are selected.",
    )
    group.add_argument(
        "--merge-save-dtype",
        choices=("same", "float32", "float16", "bfloat16"),
        default="same",
        help=(
            "Saved dtype for averaged tensors. 'same' preserves the loaded/template dtype "
            "and requires compatible input dtypes."
        ),
    )
    group.add_argument(
        "--output-iteration",
        type=int,
        default=None,
        help=(
            "If set, write --merge-output/iter_XXXXXXX and update "
            "latest_checkpointed_iteration.txt. Defaults to --end-checkpoint in range/window mode."
        ),
    )
    group.add_argument(
        "--merge-execution-mode",
        choices=MERGE_EXECUTION_MODES,
        default="baseline",
        help=(
            "Execution placement for DCP load/save tensors. 'cpu-resident' keeps loaded "
            "checkpoint tensors, fp32 accumulators, and merged output tensors on CPU. "
            "'file-backed-streaming' loads one tensor chunk at a time and stages merged "
            "local shards through file-backed CPU tensors before final DCP save. "
            "'direct-dcp-streaming' writes output chunks directly with public DCP "
            "FileSystemWriter, using distributed DCP save when the process group has "
            "more than one rank, without file-backed full-output staging; source reads "
            "still follow the source checkpoint storage-record layout."
        ),
    )
    group.add_argument(
        "--merge-streaming-chunk-bytes",
        type=int,
        default=128 * 1024 * 1024,
        help=(
            "Approximate fp32 accumulator budget per tensor chunk for streaming modes."
        ),
    )
    group.add_argument(
        "--merge-staging-dir",
        type=Path,
        default=None,
        help=(
            "Directory for file-backed streaming shard staging. Defaults to a temporary "
            "sibling of the atomic output directory and is removed after a successful merge."
        ),
    )
    group.add_argument(
        "--merge-file-backed-staging-layout",
        choices=FILE_BACKED_STAGING_LAYOUTS,
        default="shared-dtype",
        help=(
            "File-backed streaming staging layout. 'shared-dtype' minimizes staging file "
            "count by using one rank-local mmap per dtype. 'per-tensor' uses one exact-size "
            "mmap per output tensor, which avoids DCP's CPU view clone path at the cost of "
            "more staging files."
        ),
    )
    group.add_argument(
        "--merge-byte-accounting",
        choices=BYTE_ACCOUNTING_MODES,
        default="rank0",
        help=(
            "How to measure recursive checkpoint byte totals. 'rank0' avoids multiplying "
            "filesystem metadata traffic by distributed rank."
        ),
    )
    group.add_argument(
        "--merge-resource-log",
        choices=RESOURCE_LOG_MODES,
        default="none",
        help=(
            "Emit resource checkpoints around expensive setup phases. 'rank0' keeps logs "
            "compact; 'all' reports every distributed rank for memory diagnostics."
        ),
    )
    group.add_argument(
        "--merge-max-projected-cpu-bytes",
        type=int,
        default=None,
        help=(
            "Optional preflight guard. If set, fail before loading checkpoint tensors "
            "when the projected per-rank CPU peak exceeds this byte limit."
        ),
    )
    group.add_argument(
        "--merge-max-file-backed-staging-bytes",
        type=int,
        default=None,
        help=(
            "Optional preflight guard for --merge-execution-mode=file-backed-streaming. "
            "If set, fail before creating mmap staging files when projected per-rank "
            "file-backed output staging exceeds this byte limit."
        ),
    )
    group.add_argument(
        "--merge-max-file-backed-staging-files",
        type=int,
        default=None,
        help=(
            "Optional metadata preflight guard for --merge-execution-mode=file-backed-streaming. "
            "If set, fail before creating mmap staging files when the projected per-rank "
            "staging file count exceeds this limit."
        ),
    )
    group.add_argument(
        "--merge-preflight-only",
        action="store_true",
        help=(
            "Build the merge template, print memory/staging estimates, enforce preflight "
            "guards, and exit without loading, staging, saving, verifying, or writing output."
        ),
    )
    group.add_argument(
        "--merge-reuse-source-metadata-for-save",
        action="store_true",
        help=(
            "Experimental torch_dist optimization for file-backed streaming with "
            "--merge-save-dtype=same: seed DCP save planning with cached metadata from "
            "the first input checkpoint and reuse it only when Megatron validates the "
            "output layout is identical."
        ),
    )
    group.add_argument(
        "--merge-template-init-mode",
        choices=MERGE_TEMPLATE_INIT_MODES,
        default="default",
        help=(
            "How to instantiate the Megatron model used only for sharded-state template "
            "metadata. 'default' preserves normal Megatron construction, 'cpu' keeps the "
            "template model on CPU, and 'meta' builds meta tensors for metadata-only "
            "template extraction."
        ),
    )
    group.add_argument(
        "--overwrite-merge-output",
        action="store_true",
        help="Allow replacing an existing merged output checkpoint directory.",
    )
    group.add_argument(
        "--no-atomic-merge-output",
        action="store_true",
        help="Write directly to --merge-output instead of publishing from a temporary directory.",
    )
    group.add_argument(
        "--extra-state-source-index",
        type=int,
        default=0,
        help="Input checkpoint index whose Transformer Engine _extra_state values are copied.",
    )
    group.add_argument(
        "--model-builder",
        choices=("gpt", "hybrid", "mamba"),
        default="gpt",
        help="Model builder used to instantiate the sharded-state template.",
    )
    group.add_argument(
        "--verify-load",
        action="store_true",
        help="Reload the merged checkpoint with the same sharded-state template after save.",
    )
    group.add_argument(
        "--strict",
        choices=tuple(
            flag.value
            for flag in StrictHandling
            if not StrictHandling.requires_returning_mismatch_keys(flag)
        ),
        default=StrictHandling.RAISE_UNEXPECTED.value,
        help=(
            "Distributed-checkpoint strictness. The default requires requested model keys "
            "to exist while tolerating extra checkpoint shards such as optimizer state."
        ),
    )
    group.add_argument(
        "--allow-data-parallel-merge",
        action="store_true",
        help="Permit data-parallel merge-time world sizes above one. This is experimental.",
    )
    return parser


def _build_model_state_dict_factory(
    model_builder_type: str,
    template_init_mode: str = "default",
    resource_log: str = "none",
) -> Callable[[], ShardedStateDict]:
    from gpt_builders import gpt_builder
    from hybrid_builders import hybrid_builder
    from megatron.training import get_args, get_model
    from model_provider import model_provider

    args = get_args()
    apply_hybrid_layer_pattern_compat(args, model_builder_type)
    load_context = nullcontext()
    if getattr(args, "fp8", None):
        from transformer_engine.pytorch.fp8 import fp8_model_init

        load_context = fp8_model_init()

    builder = hybrid_builder if model_builder_type in ("hybrid", "mamba") else gpt_builder
    original_template_flags = {
        "init_model_with_meta_device": getattr(args, "init_model_with_meta_device", False),
        "use_cpu_initialization": getattr(args, "use_cpu_initialization", False),
        "use_torch_fsdp2": getattr(args, "use_torch_fsdp2", False),
    }
    if template_init_mode == "cpu":
        args.init_model_with_meta_device = False
        args.use_cpu_initialization = True
        args.use_torch_fsdp2 = True
    elif template_init_mode == "meta":
        args.init_model_with_meta_device = True
        args.use_cpu_initialization = False
        args.use_torch_fsdp2 = True

    model_start = time.perf_counter()
    _log_resource_checkpoint(
        f"before get_model template_init_mode={template_init_mode}",
        start_time=model_start,
        resource_log=resource_log,
    )
    try:
        with load_context:
            models = get_model(partial(model_provider, builder), wrap_with_ddp=False)
    finally:
        for key, value in original_template_flags.items():
            setattr(args, key, value)
    _log_resource_checkpoint(
        f"after get_model template_init_mode={template_init_mode}",
        start_time=model_start,
        resource_log=resource_log,
    )

    for model in models:
        model.eval()

    factory_call_count = 0

    def state_dict_factory() -> ShardedStateDict:
        nonlocal factory_call_count
        factory_call_count += 1
        factory_start = time.perf_counter()
        _log_resource_checkpoint(
            f"before sharded_state_dict factory_call={factory_call_count}",
            start_time=factory_start,
            resource_log=resource_log,
        )
        if len(models) == 1:
            state_dict = {"model": models[0].sharded_state_dict(prefix="")}
        else:
            state_dict = {
                f"model{index}": model.sharded_state_dict(prefix="")
                for index, model in enumerate(models)
            }
        _log_resource_checkpoint(
            f"after sharded_state_dict factory_call={factory_call_count}",
            start_time=factory_start,
            resource_log=resource_log,
        )
        return state_dict

    return state_dict_factory


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _format_bandwidth(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{_format_bytes(int(num_bytes / seconds))}/s"


def apply_hybrid_layer_pattern_compat(args: argparse.Namespace, model_builder_type: str) -> None:
    """Translate legacy hybrid checkpoint args before building the model template."""

    if (
        model_builder_type in ("hybrid", "mamba")
        and getattr(args, "hybrid_layer_pattern", None) is None
        and getattr(args, "hybrid_override_pattern", None) is not None
    ):
        args.hybrid_layer_pattern = args.hybrid_override_pattern


def parse_and_validate_merge_args(args_defaults: dict[str, Any]) -> argparse.Namespace:
    """Parse Megatron args for merge without constructing a tokenizer."""

    from megatron.training.arguments import parse_args, validate_args
    from megatron.training.global_vars import set_global_variables

    args = parse_args(extra_args_provider=add_merge_args)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        from megatron.training.checkpointing import load_args_from_checkpoint

        assert args.load is not None or args.pretrained_checkpoint is not None, (
            "--use-checkpoint-args requires --load or --pretrained-checkpoint argument"
        )
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args, load_arg="pretrained_checkpoint")
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        from megatron.training.yaml_arguments import validate_yaml

        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    set_global_variables(args, build_tokenizer=False)
    return args


def _data_parallel_size_from_args(args: argparse.Namespace) -> int:
    data_parallel_size = getattr(args, "data_parallel_size", None)
    if data_parallel_size is not None:
        return int(data_parallel_size)
    tensor_model_parallel_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
    pipeline_model_parallel_size = int(getattr(args, "pipeline_model_parallel_size", 1) or 1)
    context_parallel_size = int(getattr(args, "context_parallel_size", 1) or 1)
    expert_model_parallel_size = int(getattr(args, "expert_model_parallel_size", 1) or 1)
    model_parallel_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
        * expert_model_parallel_size
    )
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    return max(world_size // max(model_parallel_size, 1), 1)


def main() -> None:
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--merge-inputs", nargs="+")
    pre_parser.add_argument("--start-checkpoint", type=int)
    pre_parser.add_argument("--end-checkpoint", type=int)
    pre_args, _ = pre_parser.parse_known_args()

    help_requested = any(arg in ("-h", "--help") for arg in sys.argv[1:])
    if pre_args.merge_inputs and not help_requested:
        checkpoint_root, checkpoint_iteration = _determine_checkpoint_for_args(
            pre_args.merge_inputs, pre_args.start_checkpoint, pre_args.end_checkpoint
        )
        if "--load" not in sys.argv:
            sys.argv.extend(["--load", str(checkpoint_root)])
        if checkpoint_iteration is not None and "--ckpt-step" not in sys.argv:
            sys.argv.extend(["--ckpt-step", str(checkpoint_iteration)])
        if "--use-checkpoint-args" not in sys.argv:
            sys.argv.append("--use-checkpoint-args")

    from megatron.training import get_args
    from megatron.training.initialize import initialize_megatron

    parse_and_validate_merge_args(
        args_defaults={
            "exit_on_missing_checkpoint": False,
            "no_load_optim": True,
            "no_load_rng": True,
        },
    )
    initialize_megatron()
    args = get_args()
    data_parallel_size = _data_parallel_size_from_args(args)
    if data_parallel_size != 1 and not args.allow_data_parallel_merge:
        raise WeightedMergeError(
            "Weighted merge currently requires data-parallel size 1 by default. "
            f"Got data_parallel_size={data_parallel_size}; pass --allow-data-parallel-merge "
            "only after validating replica read/write behavior for this run."
        )
    print_rank_0(f"Merge data_parallel_size={data_parallel_size}", flush=True)

    model_init_start = time.perf_counter()
    state_dict_factory = _build_model_state_dict_factory(
        args.model_builder, args.merge_template_init_mode, args.merge_resource_log
    )
    model_init_time = time.perf_counter() - model_init_start
    use_selection_mode = args.start_checkpoint is not None or args.merge_window_btoks is not None
    output_iteration = args.output_iteration

    if use_selection_mode:
        if len(args.merge_inputs) != 1:
            raise WeightedMergeError("Range/window mode expects exactly one --merge-inputs root.")
        if args.end_checkpoint is None:
            raise WeightedMergeError("--end-checkpoint is required for range/window mode.")

        checkpoint_root = Path(args.merge_inputs[0])
        selected_iterations = select_checkpoints_in_window(
            checkpoint_root,
            start_iteration=args.start_checkpoint,
            end_iteration=args.end_checkpoint,
            token_window_btok=args.merge_window_btoks,
            seq_length=getattr(args, "seq_length", None),
            global_batch_size=getattr(args, "global_batch_size", None),
            min_iteration_interval=args.min_iteration_interval,
        )
        coefficient_map = checkpoint_coefficients(
            selected_iterations, args.merge_style, seed=args.coefficient_seed
        )
        input_paths = checkpoint_paths_for_iterations(checkpoint_root, selected_iterations)
        weights = [coefficient_map[iteration] for iteration in selected_iterations]
        if output_iteration is None:
            output_iteration = args.end_checkpoint
        merge_style = args.merge_style
    else:
        input_paths, weights = parse_weighted_inputs(args.merge_inputs)
        merge_style = None
        warn_manual_weight_policy(weights, normalize=args.normalize)

    validate_min_checkpoints(len(input_paths), args.min_checkpoints)
    result = merge_sharded_checkpoints(
        input_paths,
        weights,
        args.merge_output,
        state_dict_factory,
        normalize=args.normalize,
        save_dtype=args.merge_save_dtype,
        output_iteration=output_iteration,
        model_init_time=model_init_time,
        verify_load=args.verify_load,
        strict=args.strict,
        extra_state_source_index=args.extra_state_source_index,
        execution_mode=args.merge_execution_mode,
        byte_accounting=args.merge_byte_accounting,
        overwrite_output=args.overwrite_merge_output,
        atomic_output=not args.no_atomic_merge_output,
        streaming_chunk_bytes=args.merge_streaming_chunk_bytes,
        staging_dir=args.merge_staging_dir,
        merge_style=merge_style,
        resource_log=args.merge_resource_log,
        max_projected_cpu_bytes=args.merge_max_projected_cpu_bytes,
        max_file_backed_staging_bytes=args.merge_max_file_backed_staging_bytes,
        max_file_backed_staging_files=args.merge_max_file_backed_staging_files,
        reuse_source_metadata_for_save=args.merge_reuse_source_metadata_for_save,
        file_backed_staging_layout=args.merge_file_backed_staging_layout,
        preflight_only=args.merge_preflight_only,
    )
    if result.preflight_only:
        print_rank_0(
            "Merge preflight complete: "
            f"averaged={result.averaged_tensors}, "
            f"copied_extra_state={result.copied_extra_states}, "
            f"backend={result.backend}, implementation_mode={result.implementation_mode}, "
            f"file_backed_staging_layout={result.file_backed_staging_layout}, "
            f"output_not_written={result.output_dir}",
            flush=True,
        )
    else:
        print_rank_0(
            "Merge complete: "
            f"averaged={result.averaged_tensors}, copied_extra_state={result.copied_extra_states}, "
            f"backend={result.backend}, implementation_mode={result.implementation_mode}, "
            f"verify_load={result.verified_load}, "
            f"save_metadata_cache_reused={result.save_metadata_cache_reused}, "
            f"file_backed_staging_layout={result.file_backed_staging_layout}, "
            f"output={result.output_dir}",
            flush=True,
        )
    print_rank_0(
        "Timing: "
        f"discovery={result.timings.discovery:.2f}s, "
        f"model_init={result.timings.model_init:.2f}s, "
        f"load={result.timings.load:.2f}s, "
        f"load_per_checkpoint={result.timings.load / len(result.input_dirs):.2f}s, "
        f"accumulation={result.timings.accumulation:.2f}s, "
        f"save={result.timings.save:.2f}s, "
        f"verification={result.timings.verification:.2f}s, "
        f"total={result.timings.total:.2f}s",
        flush=True,
    )
    print_rank_0(
        "I/O: "
        f"read={_format_bytes(result.bytes_read)} "
        f"({_format_bandwidth(result.bytes_read, result.timings.load)}), "
        f"wrote={_format_bytes(result.bytes_written)} "
        f"({_format_bandwidth(result.bytes_written, result.timings.save)})",
        flush=True,
    )
    print(
        f"Memory rank={result.rank}: host_peak={_format_bytes(result.host_peak_bytes)}, "
        f"gpu_peak={_format_bytes(result.gpu_peak_bytes)}",
        flush=True,
    )
    print_rank_0(
        "Memory distributed max: "
        f"host_peak={_format_bytes(result.max_host_peak_bytes)} "
        f"(rank {result.max_host_peak_rank}/{result.world_size}), "
        f"gpu_peak={_format_bytes(result.max_gpu_peak_bytes)} "
        f"(rank {result.max_gpu_peak_rank}/{result.world_size})",
        flush=True,
    )


if __name__ == "__main__":
    main()
