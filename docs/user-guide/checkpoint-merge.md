<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Weighted Checkpoint Merge

`tools/checkpoint/weighted_merge.py` merges Megatron distributed checkpoints by
weighted averaging model tensors. It is intended for checkpoint utilities such
as warmup-stable-merge experiments, where several compatible model checkpoints
are merged into a new checkpoint that can be loaded for evaluation or resumed
with `--no-load-optim --no-load-rng`.

The utility uses Megatron's distributed-checkpointing API with the current
model's sharded state dict. It does not gather a full production model on rank
0. Existing full-tensor checkpoint helpers under `tools/checkpoint` are useful
for small conversion tests and debugging, but should not be used as the primary
path for large model merging.

The script still runs Megatron initialization to build the sharded state dict
template, so launch it in a normal Megatron runtime with the required distributed
and CUDA dependencies. `cpu-resident`, `file-backed-streaming`, and
`direct-dcp-streaming` control merge tensor placement after template
construction; they are not CPU-only execution guarantees.

## Manual Weighted Merge

Manual mode takes explicit `PATH:WEIGHT` inputs:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs \
    /checkpoints/run_a/iter_0001000:0.25 \
    /checkpoints/run_a/iter_0002000:0.75 \
  --merge-output /checkpoints/merged/manual \
  --output-iteration 2000 \
  --ckpt-format torch_dist \
  --model-builder gpt
```

Use `--normalize` when the manual weights should be normalized before merging.
Without `--normalize`, the weights are used exactly as provided.

## Range And Window Merge

Range mode selects `iter_*` directories from a checkpoint root and computes
weights from a schedule:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --start-checkpoint 1000 \
  --end-checkpoint 5000 \
  --merge-style minus-sqrt \
  --min-iteration-interval 1000 \
  --merge-output /checkpoints/merged/minus_sqrt \
  --ckpt-format torch_dist \
  --model-builder gpt
```

Token-window mode derives the start iteration from checkpoint args:

```bash
python tools/checkpoint/weighted_merge.py \
  --merge-inputs /checkpoints/run_a \
  --end-checkpoint 5000 \
  --merge-window-btoks 125 \
  --merge-style linear \
  --merge-output /checkpoints/merged/linear_125btok \
  --ckpt-format torch_dist \
  --model-builder gpt
```

`--merge-window-btoks` uses `ceil(window_tokens / (seq_length *
global_batch_size))` and always requires the target `--end-checkpoint` to exist.
Minimum-interval filtering walks backward from the target checkpoint, preserving
the target iteration. Add `--min-checkpoints` to fail when filtering selects too
few checkpoints for a meaningful merge.

## Coefficients

Supported base schedules are:

- `linear`: uniform average over the selected checkpoint set.
- `minus-sqrt`: discrete difference of `1 - sqrt(x)` over the selected
  checkpoint positions.

The supported modifiers are `__reverse` and `__scramble`, for example
`minus-sqrt__reverse`. Scramble uses `--coefficient-seed` and is deterministic
by default.

## Dtype And Metadata Rules

Floating model tensors are accumulated in fp32 on CPU. `--merge-save-dtype`
controls the saved dtype:

- `same`: preserve the dtype observed through the requested model state dict
  template. In normal `--use-checkpoint-args` usage this should match the source
  checkpoint dtype. All input dtypes for a tensor must match.
- `float32`, `float16`, `bfloat16`: cast averaged tensors to the requested dtype
  before save.

Transformer Engine `_extra_state` entries are copied from the first input
checkpoint and are not averaged. Common checkpoint metadata is copied from the
first input checkpoint. When `--output-iteration` is set, the output checkpoint
is written under `--merge-output/iter_XXXXXXX`, `iteration` metadata is updated,
and `latest_checkpointed_iteration.txt` is written in the output root.

Optimizer state and RNG state are not averaged. Load the merged checkpoint for
evaluation or resume with:

```bash
--no-load-optim --no-load-rng
```

## Compatibility

All input checkpoints must use the same distributed checkpoint format and must
be compatible with the model built by `--model-builder`. The merge fails early
when requested model keys are absent, shapes differ, checkpoint formats differ,
or non-floating model tensors cannot be merged. By default `--strict` is
`raise_unexpected`, which tolerates extra sharded checkpoint entries such as
optimizer state while still requiring every requested model tensor to exist. Use
`--strict raise_all` when debugging an exact model-only checkpoint.

The first implementation supports `torch_dist` model checkpoints. `fsdp_dtensor`
should be run as an explicit compatibility experiment and documented as limited
unless support is added in a later change.

The script prints wall-clock timing for checkpoint discovery, model
initialization, checkpoint load, accumulation, output save, and optional
post-save verification. Add `--verify-load` to reload the merged checkpoint with
the same sharded-state template after saving. It also prints input bytes read,
output bytes written, effective read/write bandwidth, and per-rank peak host/GPU
memory observed by the merge process. Cluster runtime metrics such as
`/usr/bin/time -v`, `sacct`, or site telemetry should also be captured when
building a performance table for a PR.

## Experimental Direct DCP Streaming

`--merge-execution-mode=direct-dcp-streaming` is an experimental, guarded
single-rank `torch_dist` prototype. It streams fp32-accumulated output chunks
through a tool-local public PyTorch DCP `SavePlanner` and public
`FileSystemWriter`, avoiding the file-backed full-output staging tensor used by
`file-backed-streaming` in local tests.

Use this mode only as an opt-in experiment. It no longer constructs private DCP
output storage records or manually writes DCP metadata/payloads, but it still
rejects merge-time world sizes greater than one. It supports ordinary
`ShardedTensor` leaves in the tested local fixtures, copies tensor
`_extra_state` entries from the selected source checkpoint, and can be validated
with `--verify-load`.

This mode is output-bounded only for ordinary one-payload source checkpoints.
Local source-read instrumentation showed that a 1 MiB logical request against a
normal source tensor caused PyTorch DCP to deserialize the full 16 MiB stored
payload before narrowing; a chunked-source layout loaded the 1 MiB source chunk.
End-to-end bounded RSS therefore requires chunked source checkpoint storage or a
lower-level reader that avoids full-payload deserialization. Multi-rank behavior,
`ShardedTensorFactory` support, prepended-axis or flattened-range tensors, object
`_extra_state` support, generated model-family coverage, and Super-scale RSS are
also not validated. Use `file-backed-streaming` for those layouts until the
direct writer is broadened and validated.
