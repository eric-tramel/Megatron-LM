# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.strategies import filesystem_async
from tools.checkpoint import weighted_merge as wm


def _path_from_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value)


def _patch_cpu_only_dcp_save() -> None:
    if torch.cuda.is_available():
        return
    torch.cuda.synchronize = lambda *args, **kwargs: None
    torch.cuda.current_device = lambda: torch.device("cpu")
    if not filesystem_async.HAVE_PSUTIL:
        filesystem_async._process_memory = lambda: 0


def _write_fsynced(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    output_root = _path_from_env("WM_DIRECT_RANK_LOSS_OUTPUT_ROOT")
    sentinel_dir = _path_from_env("WM_DIRECT_RANK_LOSS_SENTINEL_DIR")
    _patch_cpu_only_dcp_save()

    wm.ensure_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _write_fsynced(sentinel_dir / f"rank{rank}.pid", str(os.getpid()))

    def rank_offsets():
        return ((0, rank, world_size),)

    def template(value=0.0, *, extra_value=0.0):
        offsets = rank_offsets()
        return {
            "model": {
                "weight": ShardedTensor.from_rank_offsets(
                    "model.weight",
                    torch.full((2, 2), value, dtype=torch.float32),
                    *offsets,
                    replica_id=0,
                ),
                "bias": ShardedTensor.from_rank_offsets(
                    "model.bias",
                    torch.full((2,), value + 1, dtype=torch.float32),
                    *offsets,
                    replica_id=0,
                ),
                "decoder.layers.0._extra_state": ShardedTensor.from_rank_offsets(
                    "model.decoder.layers.0._extra_state",
                    torch.tensor([extra_value], dtype=torch.float32),
                    *offsets,
                    replica_id=0,
                ),
            }
        }

    def write_checkpoint(path, value, *, extra_value=0.0, iteration=0):
        path.mkdir(parents=True, exist_ok=True)
        state_dict = template(value, extra_value=extra_value)
        state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = iteration
        dist_checkpointing.save(state_dict, str(path))

    def block_before_publish(temporary_dir, output_dir, *, overwrite_output):
        wm._require_publishable_checkpoint_dir(temporary_dir)
        _write_fsynced(sentinel_dir / f"rank{rank}.before_publish", str(temporary_dir))
        dist.barrier()
        while True:
            time.sleep(1)

    wm._publish_temporary_output_dir = block_before_publish
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_a = output_root / "input_a"
    ckpt_b = output_root / "input_b"
    write_checkpoint(ckpt_a, 1.0 + rank, extra_value=111.0, iteration=1)
    write_checkpoint(ckpt_b, 5.0 + rank, extra_value=999.0, iteration=2)
    dist.barrier()

    wm.merge_sharded_checkpoints(
        [ckpt_a, ckpt_b],
        [0.25, 0.75],
        output_root,
        template,
        output_iteration=30,
        execution_mode="direct-dcp-streaming",
        streaming_chunk_bytes=16,
    )


if __name__ == "__main__":
    main()
