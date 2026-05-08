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
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.core import maybe_load_config
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.validation import StrictHandling, parse_strict_flag

ITERATION_RE = re.compile(r"^iter_(\d+)$")
LATEST_CHECKPOINTED_ITERATION = "latest_checkpointed_iteration.txt"
SAVE_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
VALID_MODIFIERS = ("reverse", "scramble")
SUPPORTED_INPUT_BACKENDS = ("torch_dist",)


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
    parent = checkpoint_dir.parent if ITERATION_RE.match(checkpoint_dir.name) else checkpoint_dir
    tracker = parent / LATEST_CHECKPOINTED_ITERATION
    tracker.write_text(f"{iteration}\n", encoding="utf-8")


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


def _host_peak_memory_bytes() -> int:
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _gpu_peak_memory_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


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
    initial_template = sharded_state_dict_factory()
    merge_paths, extra_paths = _classify_template(initial_template)
    sharded_keys = _sharded_keys_by_path(initial_template)
    weights = normalize_weights(weights) if normalize else validate_weights(weights)
    output_dir = output_checkpoint_dir(output_root, output_iteration)
    discovery_time = time.perf_counter() - discovery_start

    print_rank_0(
        f"Merging {len(resolved_input_dirs)} checkpoints into {output_dir} "
        f"with weights {weights}",
        flush=True,
    )

    base_common_state = dist_checkpointing.load_common_state_dict(str(resolved_input_dirs[0]))
    common_state = _prepare_common_state(base_common_state, output_iteration)

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

        load_template = sharded_state_dict_factory()
        load_start = time.perf_counter()
        loaded_state = dist_checkpointing.load(
            load_template,
            str(checkpoint_dir),
            validate_access_integrity=validate_access_integrity,
            strict=strict,
        )
        load_time += time.perf_counter() - load_start
        bytes_read += _directory_size(checkpoint_dir)

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(extra_state_values) != len(extra_paths):
        missing = len(extra_paths) - len(extra_state_values)
        raise WeightedMergeError(
            f"Failed to copy {missing} _extra_state entries from source checkpoint."
        )

    merged_state_dict = sharded_state_dict_factory()
    target_dtype = SAVE_DTYPE_MAP.get(save_dtype)
    for path, accumulator in accumulators.items():
        leaf = _get_path(merged_state_dict, path)
        template_tensor = _as_tensor(leaf)
        device = template_tensor.device if template_tensor is not None else torch.device("cpu")
        dtype = target_dtype if target_dtype is not None else source_dtypes[path]
        _assign_leaf_data(leaf, accumulator.to(device=device, dtype=dtype))

    for path, value in extra_state_values.items():
        leaf = _get_path(merged_state_dict, path)
        template_tensor = _as_tensor(leaf)
        if torch.is_tensor(value) and template_tensor is not None:
            value = value.to(device=template_tensor.device)
        _assign_leaf_data(leaf, value)

    for key, value in common_state.items():
        merged_state_dict[key] = value

    save_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(
        merged_state_dict, str(output_dir), validate_access_integrity=validate_access_integrity
    )
    if output_iteration is not None and write_latest and is_rank_0():
        write_latest_checkpointed_iteration(output_dir, output_iteration)
    if dist.is_initialized():
        dist.barrier()
    save_time = time.perf_counter() - save_start
    bytes_written = _directory_size(output_dir)

    verification_time = 0.0
    if verify_load:
        verify_start = time.perf_counter()
        verification_state_dict = sharded_state_dict_factory()
        dist_checkpointing.load(
            verification_state_dict,
            str(output_dir),
            validate_access_integrity=validate_access_integrity,
            strict=strict,
        )
        if dist.is_initialized():
            dist.barrier()
        verification_time = time.perf_counter() - verify_start

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
        host_peak_bytes=_host_peak_memory_bytes(),
        gpu_peak_bytes=_gpu_peak_memory_bytes(),
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
        help="Saved dtype for averaged tensors. 'same' requires matching source dtypes.",
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
    return parser


def _build_model_state_dict_factory(model_builder_type: str) -> Callable[[], ShardedStateDict]:
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
    with load_context:
        models = get_model(partial(model_provider, builder), wrap_with_ddp=False)

    for model in models:
        model.eval()

    if len(models) == 1:
        return lambda: {"model": models[0].sharded_state_dict(prefix="")}
    return lambda: {
        f"model{index}": model.sharded_state_dict(prefix="") for index, model in enumerate(models)
    }


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

    model_init_start = time.perf_counter()
    state_dict_factory = _build_model_state_dict_factory(args.model_builder)
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
    else:
        input_paths, weights = parse_weighted_inputs(args.merge_inputs)

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
    )
    print_rank_0(
        "Merge complete: "
        f"averaged={result.averaged_tensors}, copied_extra_state={result.copied_extra_states}, "
        f"backend={result.backend}, verify_load={result.verified_load}, output={result.output_dir}",
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
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    print(
        f"Memory rank={rank}: host_peak={_format_bytes(result.host_peak_bytes)}, "
        f"gpu_peak={_format_bytes(result.gpu_peak_bytes)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
