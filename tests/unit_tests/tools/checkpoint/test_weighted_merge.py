# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
import io
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as torch_dcp

from megatron.core import dist_checkpointing
from megatron.core import parallel_state as ps
from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.mapping import ShardedTensorFactory
from megatron.core.dist_checkpointing.strategies import filesystem_async
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint import weighted_merge as weighted_merge_module
from tools.checkpoint.weighted_merge import (
    WeightedMergeError,
    _clone_sharded_template_without_data,
    apply_hybrid_layer_pattern_compat,
    checkpoint_coefficients,
    derive_start_iteration_from_token_window,
    ensure_process_group,
    filter_checkpoints_by_interval,
    get_valid_styles,
    iteration_dir_name,
    merge_same_layout_dcp_metadata_checkpoints,
    merge_sharded_checkpoints,
    normalize_weights,
    output_checkpoint_dir,
    parse_and_validate_merge_args,
    parse_weighted_inputs,
    resolve_checkpoint_dir,
    select_checkpoints_in_window,
    validate_min_checkpoints,
    validate_weights,
    write_latest_checkpointed_iteration,
)


@pytest.fixture
def process_group():
    already_initialized = dist.is_available() and dist.is_initialized()
    ensure_process_group()
    yield
    if not already_initialized and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def cpu_only_dcp_save(monkeypatch):
    if torch.cuda.is_available():
        return
    # DCP's mcore async writer synchronizes CUDA even for these CPU-only fixtures.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    if not filesystem_async.HAVE_PSUTIL:
        monkeypatch.setattr(filesystem_async, "_process_memory", lambda: 0)


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _unused_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return str(server.getsockname()[1])


def _rank_offsets():
    world_size = _world_size()
    return ((0, _rank(), world_size),) if world_size > 1 else ()


def _template(
    value=0.0,
    *,
    dtype=torch.float32,
    extra_value=0.0,
    include_bias=True,
    shape=(2, 2),
    device=None,
):
    rank_offsets = _rank_offsets()
    model_state_dict = {
        "weight": ShardedTensor.from_rank_offsets(
            "model.weight",
            torch.full(shape, value, dtype=dtype, device=device),
            *rank_offsets,
            replica_id=0,
        ),
        "decoder.layers.0._extra_state": ShardedTensor.from_rank_offsets(
            "model.decoder.layers.0._extra_state",
            torch.tensor([extra_value], dtype=torch.float32, device=device),
            *rank_offsets,
            replica_id=0,
        ),
    }
    if include_bias:
        model_state_dict["bias"] = ShardedTensor.from_rank_offsets(
            "model.bias",
            torch.full((2,), value + 1, dtype=dtype, device=device),
            *rank_offsets,
            replica_id=0,
        )
    return {"model": model_state_dict}


def _write_checkpoint(
    path, value, *, dtype=torch.float32, extra_value=0.0, iteration=0, shape=(2, 2)
):
    state_dict = _template(value, dtype=dtype, extra_value=extra_value, shape=shape)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _generated_gpt_model_state(value):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        add_bias_linear=True,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=16,
        max_sequence_length=8,
        pre_process=True,
        post_process=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return {
        key: sharded
        for key, sharded in model.sharded_state_dict(prefix="model.").items()
        if not key.endswith("._extra_state")
    }


def _write_generated_gpt_checkpoint(path, value):
    dist_checkpointing.save(_generated_gpt_model_state(value), str(path))


def _generated_moe_gpt_model_state(value):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        add_bias_linear=True,
        num_moe_experts=2,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(
            num_experts=2,
            moe_grouped_gemm=False,
        ),
        vocab_size=16,
        max_sequence_length=8,
        pre_process=True,
        post_process=True,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    return model.sharded_state_dict(prefix="model.")


def _write_generated_moe_gpt_checkpoint(path, value):
    dist_checkpointing.save(_generated_moe_gpt_model_state(value), str(path))


UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY = "decoder.layers.0.mlp.linear_fc1._extra_state"
UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY = "mtp.layers.0.eh_proj._extra_state"


def _unprefixed_gpt_like_model_state(value, *, rank_sharded=False):
    rank_offsets = _rank_offsets() if rank_sharded else ()
    return {
        "decoder.final_layernorm.weight": ShardedTensor.from_rank_offsets(
            "decoder.final_layernorm.weight",
            torch.full((3,), value, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "decoder.layers.0.mlp.linear_fc1.weight": ShardedTensor.from_rank_offsets(
            "decoder.layers.0.mlp.linear_fc1.weight",
            torch.full((2, 3), value + 1, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "embedding.word_embeddings.weight": ShardedTensor.from_rank_offsets(
            "embedding.word_embeddings.weight",
            torch.full((4, 3), value + 2, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
        "output_layer.weight": ShardedTensor.from_rank_offsets(
            "output_layer.weight",
            torch.full((4, 3), value + 3, dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
    }


def _write_unprefixed_gpt_like_checkpoint(path, value):
    dist_checkpointing.save(_unprefixed_gpt_like_model_state(value), str(path))


def _unprefixed_gpt_like_byte_extra_template(extra_value=0):
    return {
        UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY: ShardedObject(
            UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY,
            _bytesio_state(extra_value),
            (_world_size(),),
            (_rank(),),
            replica_id=0,
        )
    }


def _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
    path, value, extra_value, *, rank_sharded=False
):
    state = _unprefixed_gpt_like_model_state(value, rank_sharded=rank_sharded)
    state[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY] = ShardedObject(
        UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY,
        _bytesio_state(extra_value),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_mtp_state(path, value, extra_value):
    state = _unprefixed_gpt_like_model_state(value)
    state["mtp.layers.0.eh_proj.weight"] = ShardedTensor.from_rank_offsets(
        "mtp.layers.0.eh_proj.weight",
        torch.full((2, 2), value + 4, dtype=torch.float32),
        replica_id=0,
    )
    state[UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY] = ShardedObject(
        UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY,
        _bytesio_state(extra_value),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(path, value):
    state = _unprefixed_gpt_like_model_state(value)
    state["optimizer._extra_state"] = ShardedObject(
        "optimizer._extra_state",
        _bytesio_state(123),
        (_world_size(),),
        (_rank(),),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _write_unprefixed_gpt_like_checkpoint_with_optimizer_tensor(path, value):
    state = _unprefixed_gpt_like_model_state(value)
    state["optimizer.param_groups.0.lr"] = ShardedTensor.from_rank_offsets(
        "optimizer.param_groups.0.lr",
        torch.tensor([value], dtype=torch.float32),
        replica_id=0,
    )
    dist_checkpointing.save(state, str(path))


def _split_weight_factory(sharded_tensor):
    sharded_tensor_without_data = sharded_tensor.without_data()
    split_point = sharded_tensor.data.shape[0] // 2
    split_sections = (split_point, sharded_tensor.data.shape[0] - split_point)

    def build(key, tensor, replica_id, flattened_range):
        base = replace(
            sharded_tensor_without_data,
            key=key,
            data=tensor,
            dtype=tensor.dtype,
            replica_id=replica_id,
            flattened_range=flattened_range,
        )
        chunks = []
        start = 0
        for name, length in zip(("left", "right"), split_sections):
            chunk = base.narrow(0, start, length)[0]
            chunk.key = f"{chunk.key}.{name}"
            chunks.append(chunk)
            start += length
        return chunks

    return ShardedTensorFactory(
        sharded_tensor.key,
        sharded_tensor.data,
        build,
        lambda chunks: torch.cat(chunks, dim=0),
        sharded_tensor.replica_id,
    )


def _factory_template(value=0.0, *, dtype=torch.float32, shape=(4, 2)):
    rank_offsets = _rank_offsets()
    weight = ShardedTensor.from_rank_offsets(
        "model.factory_weight",
        torch.full(shape, value, dtype=dtype),
        *rank_offsets,
        replica_id=0,
    )
    return {"model": {"weight": _split_weight_factory(weight)}}


def _write_factory_checkpoint(path, value, *, iteration=0, shape=(4, 2)):
    state_dict = _factory_template(value, shape=shape)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _prepended_axis_template(value=0.0, *, dtype=torch.float32, shape=(4, 2)):
    weight = ShardedTensor.from_rank_offsets(
        "model.prepended_weight",
        torch.full(shape, value, dtype=dtype),
        replica_id=0,
        prepend_axis_num=1,
    )
    return {"model": {"weight": weight}}


def _write_prepended_axis_checkpoint(path, value, *, iteration=0, shape=(4, 2)):
    state_dict = _prepended_axis_template(value, shape=shape)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _load_checkpoint(path, *, dtype=torch.float32, shape=(2, 2)):
    return dist_checkpointing.load(_template(dtype=dtype, shape=shape), str(path))


def _full_public_dcp_state(path, *, dtype=torch.float32, shape=(2, 2)):
    world_size = _world_size()
    state_dict = {
        "model.weight": torch.empty((shape[0] * world_size, shape[1]), dtype=dtype),
        "model.bias": torch.empty((2 * world_size,), dtype=dtype),
        "model.decoder.layers.0._extra_state": torch.empty((world_size,), dtype=torch.float32),
    }
    torch_dcp.load(state_dict, checkpoint_id=str(path), no_dist=True)
    return state_dict


def _dcp_metadata_summary(path):
    metadata = torch_dcp.FileSystemReader(path).read_metadata()
    chunk_records = []
    for fqn, tensor_metadata in metadata.state_dict_metadata.items():
        for chunk in getattr(tensor_metadata, "chunks", []) or []:
            chunk_records.append(
                (
                    fqn,
                    tuple(int(offset) for offset in chunk.offsets),
                    tuple(int(size) for size in chunk.sizes),
                )
            )

    storage_data = getattr(metadata, "storage_data", {}) or {}
    storage_records = [
        (
            getattr(record, "relative_path", None),
            getattr(record, "offset", None),
            getattr(record, "length", None),
        )
        for record in storage_data.values()
    ]
    duplicate_chunk_offsets = {
        key: count
        for key, count in Counter((fqn, offsets) for fqn, offsets, _ in chunk_records).items()
        if count > 1
    }
    duplicate_storage_records = {
        key: count for key, count in Counter(storage_records).items() if count > 1
    }
    return {
        "chunk_records": chunk_records,
        "duplicate_chunk_offsets": duplicate_chunk_offsets,
        "duplicate_storage_records": duplicate_storage_records,
        "storage_file_count": len({str(record[0]) for record in storage_records}),
    }


def _assert_no_direct_atomic_publication(output_root, output_iteration):
    assert not (output_root / iteration_dir_name(output_iteration)).exists()
    assert not (output_root / "latest_checkpointed_iteration.txt").exists()


def _bytesio_state(value):
    data = io.BytesIO()
    torch.save({"value": value}, data)
    data.seek(0)
    return data


def _decode_sharded_object_value(value):
    if value is None:
        return None
    if not isinstance(value, io.BytesIO):
        return value
    value.seek(0)
    payload = torch.load(value, weights_only=False)
    if isinstance(payload, list):
        assert len(payload) == 1
        payload = payload[0]
    if isinstance(payload, io.BytesIO):
        payload.seek(0)
        return torch.load(payload, weights_only=False)
    return payload


def _object_extra_template(extra_value=0):
    rank_offsets = _rank_offsets()
    return {
        "model": {
            "weight": ShardedTensor.from_rank_offsets(
                "model.weight",
                torch.zeros((2, 2), dtype=torch.float32),
                *rank_offsets,
                replica_id=0,
            ),
            "decoder.layers.0._extra_state": ShardedObject(
                "model.decoder.layers.0._extra_state",
                _bytesio_state(extra_value),
                (_world_size(),),
                (_rank(),),
                replica_id=0,
            ),
        }
    }


def _write_object_extra_checkpoint(path, value, extra_value, iteration):
    state_dict = _object_extra_template(extra_value)
    state_dict["model"]["weight"].data.fill_(value)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _rank_local_object_extra_key(rank=None):
    rank = _rank() if rank is None else rank
    return f"decoder.layers.{rank}._extra_state"


def _rank_local_object_extra_template(extra_value=0):
    rank = _rank()
    rank_offsets = _rank_offsets()
    local_extra_key = _rank_local_object_extra_key(rank)
    return {
        "model": {
            "weight": ShardedTensor.from_rank_offsets(
                "model.weight",
                torch.zeros((2, 2), dtype=torch.float32),
                *rank_offsets,
                replica_id=0,
            ),
            local_extra_key: ShardedObject(
                f"model.{local_extra_key}",
                _bytesio_state(extra_value),
                (1,),
                (0,),
                replica_id=0,
            ),
        }
    }


def _write_rank_local_object_extra_checkpoint(path, value, extra_value, iteration):
    state_dict = _rank_local_object_extra_template(extra_value)
    state_dict["model"]["weight"].data.fill_(value)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def _multi_chunk_template(value=0.0, *, dtype=torch.float32):
    rank_offsets = _rank_offsets()
    return {
        "model0": {
            "weight": ShardedTensor.from_rank_offsets(
                "model0.weight", torch.full((2, 2), value, dtype=dtype), *rank_offsets, replica_id=0
            )
        },
        "model1": {
            "weight": ShardedTensor.from_rank_offsets(
                "model1.weight",
                torch.full((2, 2), value + 1, dtype=dtype),
                *rank_offsets,
                replica_id=0,
            )
        },
    }


def _write_multi_chunk_checkpoint(path, value, *, iteration=0):
    state_dict = _multi_chunk_template(value)
    state_dict["args"] = SimpleNamespace(iteration=iteration, hidden_size=2)
    state_dict["checkpoint_version"] = 3.0
    state_dict["iteration"] = iteration
    dist_checkpointing.save(state_dict, str(path))


def test_linear_coefficients_are_uniform():
    coeffs = checkpoint_coefficients([100, 200, 300, 400], "linear")
    assert list(coeffs) == [100, 200, 300, 400]
    assert all(abs(value - 0.25) < 1e-12 for value in coeffs.values())
    assert abs(sum(coeffs.values()) - 1.0) < 1e-12


def test_minus_sqrt_matches_discrete_difference():
    checkpoints = [10, 20, 30, 40]
    coeffs = checkpoint_coefficients(checkpoints, "minus-sqrt")
    decay = [1 - (index / len(checkpoints)) ** 0.5 for index in range(len(checkpoints))]
    expected = [
        1 - ((decay[1] - decay[2]) + (decay[2] - decay[3]) + decay[3]),
        decay[1] - decay[2],
        decay[2] - decay[3],
        decay[3],
    ]

    assert list(coeffs.values()) == pytest.approx(expected)
    assert abs(sum(coeffs.values()) - 1.0) < 1e-12


@pytest.mark.parametrize("style", get_valid_styles())
@pytest.mark.parametrize("n_checkpoints", [1, 2, 5])
def test_supported_coefficients_are_deterministic_and_normalized(style, n_checkpoints):
    checkpoints = list(range(n_checkpoints))
    coeffs_a = checkpoint_coefficients(checkpoints, style, seed=123)
    coeffs_b = checkpoint_coefficients(checkpoints, style, seed=123)

    assert coeffs_a == coeffs_b
    assert list(coeffs_a) == checkpoints
    assert sum(coeffs_a.values()) == pytest.approx(1.0)


def test_modifiers_are_deterministic():
    checkpoints = [1, 2, 3, 4, 5]
    normal = list(checkpoint_coefficients(checkpoints, "minus-sqrt").values())
    reversed_coeffs = list(checkpoint_coefficients(checkpoints, "minus-sqrt__reverse").values())
    scrambled_a = checkpoint_coefficients(checkpoints, "minus-sqrt__scramble", seed=17)
    scrambled_b = checkpoint_coefficients(checkpoints, "minus-sqrt__scramble", seed=17)

    assert reversed_coeffs == list(reversed(normal))
    assert scrambled_a == scrambled_b
    assert sorted(scrambled_a.values()) == sorted(normal)


def test_parse_weighted_inputs_uses_last_colon_for_weight():
    paths, weights = parse_weighted_inputs(["/tmp/with:colon/iter_0000010:0.75"])
    assert str(paths[0]) == "/tmp/with:colon/iter_0000010"
    assert weights == [0.75]


def test_manual_weight_policy_warnings():
    assert weighted_merge_module._manual_weight_warnings([0.25, 0.75], normalize=False) == []
    assert weighted_merge_module._manual_weight_warnings([2.0, -1.0], normalize=False) == [
        "WARNING: manual merge weights include negative values; this is allowed for "
        "subtractive merges but can produce outputs outside the input checkpoint range."
    ]

    unnormalized = weighted_merge_module._manual_weight_warnings([0.25, 0.25], normalize=False)
    assert len(unnormalized) == 1
    assert "sum to 0.5 without --normalize" in unnormalized[0]

    normalized = weighted_merge_module._manual_weight_warnings([0.25, 0.25], normalize=True)
    assert normalized == []


def test_save_strategy_from_source_metadata_uses_cached_plans(monkeypatch):
    class SourceMetadata:
        all_local_plans = [object()]

    class FakeSaveStrategy:
        def __init__(self, *, cached_metadata=False):
            self.cached_metadata = cached_metadata
            self.cached_global_metadata = None

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(weighted_merge_module, "TorchDistSaveShardedStrategy", FakeSaveStrategy)

    source_metadata = SourceMetadata()
    strategy = weighted_merge_module._save_strategy_from_source_metadata(
        source_metadata, requested=True
    )

    assert strategy.cached_metadata is True
    assert strategy.cached_global_metadata is source_metadata


def test_save_strategy_from_source_metadata_falls_back_without_cuda(monkeypatch):
    class SourceMetadata:
        all_local_plans = [object()]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert (
        weighted_merge_module._save_strategy_from_source_metadata(
            SourceMetadata(), requested=True
        )
        is None
    )


def test_save_metadata_cache_reuse_reports_seeded_dcp_rejection(capsys):
    strategy = SimpleNamespace(validated_loaded_metadata_reuse=False)

    assert (
        weighted_merge_module._report_save_metadata_cache_reuse(strategy, requested=True)
        is False
    )

    captured = capsys.readouterr()
    assert "DCP save metadata cache reuse: not reused" in captured.out
    assert "source metadata was seeded" in captured.out
    assert "DCP did not validate loaded metadata reuse" in captured.out
    assert "metadata generated during save planning" in captured.out


def test_weighted_merge_product_code_imports_no_private_dcp_filesystem_symbols():
    source = Path(weighted_merge_module.__file__).read_text()
    banned_tokens = {
        "_StorageInfo",
        "_metadata_fn",
        "CURRENT_DCP_VERSION",
        "DEFAULT_SUFFIX",
        "__0_0",
        "torch.distributed.checkpoint.filesystem",
        "pickle.dump",
    }
    assert not {token for token in banned_tokens if token in source}

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch.distributed.checkpoint.filesystem"


def test_publish_temporary_output_dir_requires_public_dcp_metadata(tmp_path):
    temporary_dir = tmp_path / ".merged.tmp-deadbeef"
    output_dir = tmp_path / "merged"
    temporary_dir.mkdir()
    (temporary_dir / "metadata.json").write_text(
        '{"sharded_backend": "torch_dist", "sharded_backend_version": 1, '
        '"common_backend": "torch", "common_backend_version": 1}',
        encoding="utf-8",
    )

    with pytest.raises(WeightedMergeError, match="DCP metadata"):
        weighted_merge_module._publish_temporary_output_dir(
            temporary_dir, output_dir, overwrite_output=False
        )

    assert temporary_dir.exists()
    assert not output_dir.exists()


def test_direct_dcp_streaming_rejects_existing_output_overwrite_for_crash_safety(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_overwrite_reject_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_overwrite_reject_b") as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_direct_overwrite_reject_out"
        ) as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)
        final_dir = output_root / "iter_0000030"
        final_dir.mkdir()
        _write_checkpoint(final_dir, 9.0, iteration=29)
        write_latest_checkpointed_iteration(final_dir, 29)

        with pytest.raises(WeightedMergeError, match="crash-atomic"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
                overwrite_output=True,
            )

        restored = _load_checkpoint(final_dir)
        torch.testing.assert_close(restored["model"]["weight"], torch.full((2, 2), 9.0))
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "29"
        assert not list(output_root.glob(".iter_0000030.old-*"))
        assert not list(output_root.glob(".iter_0000030.tmp-*"))


def test_publish_temporary_output_dir_overwrite_backup_cleanup_failure_is_nonfatal(
    tmp_path_dist_ckpt, process_group, monkeypatch, capsys
):
    output_dir = tmp_path_dist_ckpt / "weighted_merge_publish_cleanup_fail_out"
    temporary_dir = tmp_path_dist_ckpt / ".weighted_merge_publish_cleanup_fail_out.tmp-deadbeef"
    output_dir.mkdir()
    temporary_dir.mkdir()
    _write_checkpoint(output_dir, 3.0, iteration=3)
    _write_checkpoint(temporary_dir, 7.0, iteration=7)

    real_rmtree = weighted_merge_module.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        path = Path(path)
        if path.parent == tmp_path_dist_ckpt and path.name.startswith(
            ".weighted_merge_publish_cleanup_fail_out.old-"
        ):
            raise OSError("injected backup cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(weighted_merge_module.shutil, "rmtree", fail_backup_cleanup)

    weighted_merge_module._publish_temporary_output_dir(
        temporary_dir, output_dir, overwrite_output=True
    )

    published = _load_checkpoint(output_dir)
    torch.testing.assert_close(published["model"]["weight"], torch.full((2, 2), 7.0))
    assert not temporary_dir.exists()
    backups = list(tmp_path_dist_ckpt.glob(".weighted_merge_publish_cleanup_fail_out.old-*"))
    assert len(backups) == 1
    backup_state = _load_checkpoint(backups[0])
    torch.testing.assert_close(backup_state["model"]["weight"], torch.full((2, 2), 3.0))
    assert "failed to remove overwritten checkpoint backup" in capsys.readouterr().out


def test_file_backed_per_tensor_staging_uses_exact_storage(tmp_path):
    shared_store = weighted_merge_module._FileBackedTensorStore(
        tmp_path / "shared",
        rank=0,
        dtype_numels={torch.float32: 6},
        layout="shared-dtype",
    )
    shared_first = shared_store.allocate((2,), torch.float32, "first")
    shared_store.allocate((4,), torch.float32, "second")

    per_tensor_store = weighted_merge_module._FileBackedTensorStore(
        tmp_path / "per_tensor",
        rank=0,
        dtype_numels={torch.float32: 6},
        layout="per-tensor",
    )
    per_tensor_first = per_tensor_store.allocate((2,), torch.float32, "first")
    per_tensor_store.allocate((4,), torch.float32, "second")

    assert shared_first.untyped_storage().size() != shared_first.numel() * shared_first.itemsize
    assert per_tensor_first.untyped_storage().size() == (
        per_tensor_first.numel() * per_tensor_first.itemsize
    )
    assert shared_store.file_count == 1
    assert per_tensor_store.file_count == 2


def test_legacy_hybrid_override_pattern_is_used_for_hybrid_builder():
    args = SimpleNamespace(hybrid_layer_pattern=None, hybrid_override_pattern="MEME")

    apply_hybrid_layer_pattern_compat(args, "mamba")

    assert args.hybrid_layer_pattern == "MEME"


def test_legacy_hybrid_override_pattern_does_not_override_existing_pattern():
    args = SimpleNamespace(hybrid_layer_pattern="M*M*", hybrid_override_pattern="MEME")

    apply_hybrid_layer_pattern_compat(args, "hybrid")

    assert args.hybrid_layer_pattern == "M*M*"


def test_legacy_hybrid_override_pattern_is_ignored_for_gpt_builder():
    args = SimpleNamespace(hybrid_layer_pattern=None, hybrid_override_pattern="MEME")

    apply_hybrid_layer_pattern_compat(args, "gpt")

    assert args.hybrid_layer_pattern is None


def test_merge_arg_parsing_skips_tokenizer_build(monkeypatch):
    parsed_args = SimpleNamespace(
        use_checkpoint_args=False,
        yaml_cfg=None,
    )
    calls = {}

    def fake_parse_args(extra_args_provider=None):
        calls["extra_args_provider"] = extra_args_provider
        return parsed_args

    def fake_validate_args(args, args_defaults):
        calls["validated"] = (args, args_defaults)

    def fake_set_global_variables(args, build_tokenizer=True):
        calls["set_global_variables"] = (args, build_tokenizer)

    monkeypatch.setattr("megatron.training.arguments.parse_args", fake_parse_args)
    monkeypatch.setattr("megatron.training.arguments.validate_args", fake_validate_args)
    monkeypatch.setattr(
        "megatron.training.global_vars.set_global_variables", fake_set_global_variables
    )

    result = parse_and_validate_merge_args({"no_load_optim": True})

    assert result is parsed_args
    assert calls["extra_args_provider"] is not None
    assert calls["validated"] == (parsed_args, {"no_load_optim": True})
    assert calls["set_global_variables"] == (parsed_args, False)


def test_invalid_pure_inputs_raise(tmp_path):
    with pytest.raises(WeightedMergeError, match="PATH:WEIGHT"):
        parse_weighted_inputs([str(tmp_path / "iter_0000010")])
    with pytest.raises(WeightedMergeError, match="Invalid weight"):
        parse_weighted_inputs([f"{tmp_path / 'iter_0000010'}:not-a-float"])
    with pytest.raises(WeightedMergeError, match="Unknown coefficient schedule"):
        checkpoint_coefficients([1, 2], "unknown")
    with pytest.raises(WeightedMergeError, match="Unknown coefficient modifier"):
        checkpoint_coefficients([1, 2], "linear__unknown")
    with pytest.raises(WeightedMergeError, match="Weight sum"):
        normalize_weights([1.0, -1.0])
    with pytest.raises(WeightedMergeError, match="finite"):
        normalize_weights([1.0, float("nan")])
    with pytest.raises(WeightedMergeError, match="finite"):
        validate_weights([1.0, float("inf")])
    with pytest.raises(WeightedMergeError, match="min_checkpoints"):
        validate_min_checkpoints(2, 0)
    with pytest.raises(WeightedMergeError, match="at least 3"):
        validate_min_checkpoints(2, 3)
    with pytest.raises(WeightedMergeError, match="does not match requested iteration"):
        output_checkpoint_dir(tmp_path / "iter_0000001", 2)
    with pytest.raises(WeightedMergeError, match="not a distributed checkpoint"):
        resolve_checkpoint_dir(tmp_path)


def test_select_checkpoints_preserves_target_and_applies_interval(tmp_path):
    for iteration in [100, 150, 210, 260, 300]:
        (tmp_path / iteration_dir_name(iteration)).mkdir()

    selected = select_checkpoints_in_window(
        tmp_path, start_iteration=100, end_iteration=300, min_iteration_interval=100
    )

    assert selected == [150, 300]


def test_token_window_selection_uses_ceil_and_requires_target(tmp_path):
    for iteration in [0, 10, 20, 30]:
        (tmp_path / iteration_dir_name(iteration)).mkdir()

    start = derive_start_iteration_from_token_window(
        end_iteration=30, token_window_btok=1, seq_length=128, global_batch_size=1_000_000
    )
    selected = select_checkpoints_in_window(
        tmp_path,
        start_iteration=None,
        end_iteration=30,
        token_window_btok=1,
        seq_length=128,
        global_batch_size=1_000_000,
    )

    assert start == 22
    assert selected == [30]
    with pytest.raises(WeightedMergeError, match="Target iteration"):
        select_checkpoints_in_window(tmp_path, start_iteration=0, end_iteration=40)
    with pytest.raises(WeightedMergeError, match="requires seq_length and global_batch_size"):
        select_checkpoints_in_window(
            tmp_path, start_iteration=None, end_iteration=30, token_window_btok=1
        )


def test_filter_checkpoints_by_interval_keeps_last_checkpoint():
    assert filter_checkpoints_by_interval([100, 180, 240, 300], 100) == [180, 300]


def test_merge_sharded_checkpoints_round_trip(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            lambda: _template(),
            output_iteration=30,
            verify_load=True,
        )

        assert result.output_dir == output_root / "iter_0000030"
        assert result.verified_load
        assert result.timings.verification >= 0.0
        assert result.memory_estimate.mergeable_tensors == 2
        assert result.memory_estimate.extra_state_entries == 1
        assert result.memory_estimate.loaded_checkpoint_bytes == 24
        assert result.memory_estimate.accumulator_bytes == 24
        assert result.memory_estimate.output_tensor_bytes == 24
        assert result.memory_estimate.projected_cpu_peak_bytes == 52
        assert result.memory_estimate.projected_gpu_peak_bytes == 0
        assert result.memory_estimate.file_backed_staging_bytes == 0
        assert result.memory_estimate.file_backed_staging_files == 0
        assert result.max_host_peak_bytes >= result.host_peak_bytes
        assert result.max_gpu_peak_bytes >= result.gpu_peak_bytes
        assert result.world_size >= 1
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"

        loaded = _load_checkpoint(result.output_dir)
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0]))
        assert loaded["checkpoint_version"] == 3.0
        assert loaded["iteration"] == 30
        assert loaded["args"].iteration == 30
        assert loaded["args"].hidden_size == 2
        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["weights"] == [0.25, 0.75]
        assert provenance["implementation_mode"] == "baseline"
        assert provenance["extra_state_source_index"] == 0
        assert provenance["optimizer_merged"] is False
        assert provenance["rng_merged"] is False


def test_projected_cpu_memory_guard_fails_before_loading(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_guard_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_guard_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_guard_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        with pytest.raises(WeightedMergeError, match="Projected CPU peak memory"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                max_projected_cpu_bytes=51,
            )

        assert not (output_root / "merged").exists()


def test_file_backed_staging_guard_fails_before_staging(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_guard_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_guard_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_guard_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        with pytest.raises(WeightedMergeError, match="file-backed staging storage"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                execution_mode="file-backed-streaming",
                max_file_backed_staging_bytes=23,
            )

        assert not (output_root / "merged").exists()
        assert not (output_root / ".merged.staging").exists()


def test_file_backed_staging_file_guard_fails_before_staging(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_file_guard_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_file_guard_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_staging_file_guard_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        with pytest.raises(WeightedMergeError, match="staging file count"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                execution_mode="file-backed-streaming",
                file_backed_staging_layout="per-tensor",
                max_file_backed_staging_files=1,
            )

        assert not (output_root / "merged").exists()
        assert not (output_root / ".merged.staging").exists()


def test_file_backed_streaming_preflight_only_writes_nothing(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_preflight_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_preflight_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_preflight_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _template(),
            execution_mode="file-backed-streaming",
            preflight_only=True,
        )

        assert result.preflight_only is True
        assert result.output_dir == output_root / "merged"
        assert result.bytes_read == 0
        assert result.bytes_written == 0
        assert result.timings.load == 0.0
        assert result.timings.save == 0.0
        assert result.memory_estimate.file_backed_staging_bytes == 24
        assert result.memory_estimate.file_backed_staging_files == 1
        assert not (output_root / "merged").exists()
        assert not (output_root / ".merged.staging").exists()
        assert not list(output_root.glob(".merged.tmp-*"))


def test_merge_without_output_iteration_preserves_common_metadata(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_meta_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_meta_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_meta_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=10)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=20)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged", lambda: _template()
        )
        loaded = _load_checkpoint(result.output_dir)

        assert result.output_dir == output_root / "merged"
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()
        assert loaded["iteration"] == 10
        assert loaded["args"].iteration == 10
        assert loaded["args"].hidden_size == 2


def test_cpu_resident_mode_uses_cpu_buffers_for_meta_template(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cpu_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cpu_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cpu_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1)
        _write_checkpoint(ckpt_b, 3.0, extra_value=999.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _template(device="meta"),
            execution_mode="cpu-resident",
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir)
        assert result.implementation_mode == "cpu-resident"
        assert result.verified_load
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 2.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0]))


def test_file_backed_streaming_mode_round_trip(tmp_path_dist_ckpt, process_group):
    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)
        staging_root = output_root / "staging"

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(shape=shape),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            staging_dir=staging_root,
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert result.implementation_mode == "file-backed-streaming"
        assert result.verified_load
        assert result.memory_estimate.projected_gpu_peak_bytes == 0
        assert result.memory_estimate.file_backed_staging_bytes == (
            result.memory_estimate.output_tensor_bytes
        )
        assert result.memory_estimate.file_backed_staging_files == 1
        assert result.memory_estimate.projected_cpu_peak_bytes < (
            result.memory_estimate.accumulator_bytes
            + result.memory_estimate.loaded_checkpoint_bytes
            + result.memory_estimate.extra_state_tensor_bytes
        )
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0]))
        staged_files = list(staging_root.glob("rank*-*.bin"))
        assert len(staged_files) == 1
        assert staged_files[0].name.endswith("float32.bin")
        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        assert common_state["weighted_merge_provenance"]["implementation_mode"] == (
            "file-backed-streaming"
        )


def test_file_backed_streaming_metadata_cache_request_falls_back_when_strategy_unavailable(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_cache_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_cache_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_cache_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        monkeypatch.setattr(
            weighted_merge_module,
            "_save_strategy_from_source_metadata",
            lambda source_metadata, *, requested: None,
        )
        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(shape=shape),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            reuse_source_metadata_for_save=True,
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert result.save_metadata_cache_requested is True
        assert result.save_metadata_cache_reused is False
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))


def test_file_backed_streaming_per_tensor_staging_round_trip(
    tmp_path_dist_ckpt, process_group
):
    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_per_tensor_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_per_tensor_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_per_tensor_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)
        staging_root = output_root / "staging"

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(shape=shape),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            staging_dir=staging_root,
            file_backed_staging_layout="per-tensor",
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        staged_files = list(staging_root.glob("rank*-*.bin"))
        assert result.file_backed_staging_layout == "per-tensor"
        assert result.memory_estimate.file_backed_staging_files == 2
        assert len(staged_files) == 2
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))


def test_file_backed_streaming_metadata_cache_falls_back_without_cached_plans(
    tmp_path_dist_ckpt, process_group
):
    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_reuse_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_reuse_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_reuse_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(shape=shape),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            reuse_source_metadata_for_save=True,
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert result.save_metadata_cache_requested is True
        assert result.save_metadata_cache_reused is False
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))


def test_source_metadata_save_reuse_requires_file_backed_same_dtype(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cache_guard_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_cache_guard_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)

        with pytest.raises(WeightedMergeError, match="file-backed-streaming"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root / "baseline",
                lambda: _template(),
                reuse_source_metadata_for_save=True,
            )

        with pytest.raises(WeightedMergeError, match="merge-save-dtype=same"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root / "streaming",
                lambda: _template(),
                execution_mode="file-backed-streaming",
                save_dtype="float32",
                reuse_source_metadata_for_save=True,
            )


def test_file_backed_streaming_loads_before_all_outputs_are_staged(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    shape = (5, 2)
    allocations = []
    first_load = {}
    real_allocate = weighted_merge_module._FileBackedTensorStore.allocate
    real_load = weighted_merge_module._load_tensor_path_group_fast

    def tracking_allocate(self, shape, dtype, label=None):
        allocations.append(tuple(int(dim) for dim in shape))
        return real_allocate(self, shape, dtype, label)

    def tracking_load(checkpoint_dir, path_leaves, sharded_strategy):
        first_load.setdefault("allocation_count", len(allocations))
        first_load.setdefault("path_count", len(path_leaves))
        return real_load(checkpoint_dir, path_leaves, sharded_strategy)

    monkeypatch.setattr(
        weighted_merge_module._FileBackedTensorStore,
        "allocate",
        tracking_allocate,
    )
    monkeypatch.setattr(
        weighted_merge_module,
        "_load_tensor_path_group_fast",
        tracking_load,
    )

    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_lazy_stream_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_lazy_stream_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_lazy_stream_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(shape=shape),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            staging_dir=output_root / "staging",
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert first_load["allocation_count"] == 1
        assert first_load["path_count"] == 1
        assert len(allocations) == 2
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))


def test_streaming_output_template_clone_drops_tensor_buffers(process_group):
    template = _template(shape=(4, 2))
    factory_template = _factory_template(shape=(4, 2))
    cloned = _clone_sharded_template_without_data(
        {"plain": template["model"]["weight"], "factory": factory_template["model"]["weight"]}
    )

    assert template["model"]["weight"].data is not None
    assert factory_template["model"]["weight"].data is not None
    assert cloned["plain"].data is None
    assert cloned["factory"].data is None
    assert cloned["plain"].key == template["model"]["weight"].key
    assert cloned["factory"].key == factory_template["model"]["weight"].key


def test_file_backed_streaming_supports_sharded_tensor_factory(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_factory_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_factory_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_factory_out") as output_root,
    ):
        _write_factory_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_factory_checkpoint(ckpt_b, 5.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _factory_template(),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )

        loaded = dist_checkpointing.load(_factory_template(), str(result.output_dir))
        assert result.implementation_mode == "file-backed-streaming"
        assert result.verified_load
        assert torch.allclose(loaded["model"]["weight"], torch.full((4, 2), 4.0))


def test_file_backed_streaming_supports_prepended_axis_tensors(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_prepend_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_prepend_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_prepend_out") as output_root,
    ):
        _write_prepended_axis_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_prepended_axis_checkpoint(ckpt_b, 5.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _prepended_axis_template(),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )

        loaded = dist_checkpointing.load(_prepended_axis_template(), str(result.output_dir))
        assert result.implementation_mode == "file-backed-streaming"
        assert result.verified_load
        assert torch.allclose(loaded["model"]["weight"], torch.full((4, 2), 4.0))


def test_file_backed_streaming_same_dtype_matches_baseline_template_dtype(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_same_dtype_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_same_dtype_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_same_dtype_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.bfloat16, iteration=1)
        _write_checkpoint(ckpt_b, 1.0078125, dtype=torch.bfloat16, iteration=2)

        baseline = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "baseline",
            lambda: _template(dtype=torch.float32),
            save_dtype="same",
        )
        streaming = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "streaming",
            lambda: _template(dtype=torch.float32),
            save_dtype="same",
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
        )

        baseline_loaded = _load_checkpoint(baseline.output_dir, dtype=torch.float32)
        streaming_loaded = _load_checkpoint(streaming.output_dir, dtype=torch.float32)
        assert torch.equal(
            baseline_loaded["model"]["weight"], streaming_loaded["model"]["weight"]
        )
        assert torch.equal(baseline_loaded["model"]["bias"], streaming_loaded["model"]["bias"])
        streaming_metadata = weighted_merge_module.load_sharded_metadata(str(streaming.output_dir))
        assert streaming_metadata["model.weight"].dtype == torch.float32
        assert streaming_metadata["model.bias"].dtype == torch.float32


def test_direct_dcp_streaming_mode_round_trip_without_file_backed_staging(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    shape = (5, 2)
    metadata_read_paths = []
    real_reader = weighted_merge_module.FileSystemReader

    class TrackingFileSystemReader:
        def __init__(self, path, *args, **kwargs):
            self.path = Path(path)
            self.reader = real_reader(path, *args, **kwargs)

        def read_metadata(self, *args, **kwargs):
            metadata = self.reader.read_metadata(*args, **kwargs)
            metadata_read_paths.append(self.path)
            return metadata

        def __getattr__(self, name):
            return getattr(self.reader, name)

    monkeypatch.setattr(weighted_merge_module, "FileSystemReader", TrackingFileSystemReader)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)
        staging_root = output_root / "staging"

        def fail_file_backed_tensor(*_args, **_kwargs):
            raise AssertionError("direct DCP streaming must not allocate file-backed staging")

        monkeypatch.setattr(weighted_merge_module, "_file_backed_tensor", fail_file_backed_tensor)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            lambda: _template(shape=shape),
            output_iteration=30,
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
            staging_dir=staging_root,
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == "direct-dcp-streaming"
        assert result.verified_load
        assert result.memory_estimate.file_backed_staging_bytes == 0
        assert result.memory_estimate.file_backed_staging_files == 0
        assert not staging_root.exists()
        assert torch.allclose(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0]))
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"
        prepublish_reads = [
            path
            for path in metadata_read_paths
            if path.parent == output_root and path.name.startswith(".iter_0000030.tmp-")
        ]
        assert len(prepublish_reads) == 1
        assert not prepublish_reads[0].exists()

        postpublish_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        assert "model.weight" in postpublish_metadata.state_dict_metadata
        assert "model.bias" in postpublish_metadata.state_dict_metadata

        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["implementation_mode"] == "direct-dcp-streaming"
        assert provenance["weights"] == [0.25, 0.75]
        assert provenance["output_iteration"] == 30
        assert common_state["iteration"] == 30
        assert common_state["args"].iteration == 30

        dcp_loaded = {
            "model.weight": torch.empty(shape, dtype=torch.float32),
            "model.bias": torch.empty((2,), dtype=torch.float32),
            "model.decoder.layers.0._extra_state": torch.empty((1,), dtype=torch.float32),
        }
        torch_dcp.load(dcp_loaded, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(dcp_loaded["model.weight"], torch.full(shape, 4.0))
        assert torch.equal(dcp_loaded["model.bias"], torch.full((2,), 5.0))
        assert torch.equal(dcp_loaded["model.decoder.layers.0._extra_state"], torch.tensor([111.0]))


def test_direct_dcp_streaming_bfloat16_save_dtype_and_chunk_metadata(
    tmp_path_dist_ckpt, process_group
):
    shape = (5, 2)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_bf16_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_bf16_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_bf16_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.float16, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, dtype=torch.float16, extra_value=999.0, iteration=2, shape=shape)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _template(dtype=torch.float16, shape=shape),
            save_dtype="bfloat16",
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )

        loaded = _load_checkpoint(result.output_dir, dtype=torch.bfloat16, shape=shape)
        assert result.implementation_mode == "direct-dcp-streaming"
        assert loaded["model"]["weight"].dtype == torch.bfloat16
        assert loaded["model"]["bias"].dtype == torch.bfloat16
        assert torch.equal(loaded["model"]["weight"], torch.full(shape, 4.0, dtype=torch.bfloat16))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), 5.0, dtype=torch.bfloat16))

        dcp_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        weight_metadata = dcp_metadata.state_dict_metadata["model.weight"]
        assert weight_metadata.properties.dtype == torch.bfloat16
        assert len(weight_metadata.chunks) > 1
        assert all(chunk.sizes.numel() < torch.Size(shape).numel() for chunk in weight_metadata.chunks)


def test_direct_dcp_streaming_verify_load_checks_hidden_temp_before_publication(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_temp_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_temp_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_temp_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)
        final_dir = output_root / "iter_0000030"
        marker = output_root / "latest_checkpointed_iteration.txt"
        verify_calls = []
        real_load = weighted_merge_module.dist_checkpointing.load

        def tracking_load(state_dict, checkpoint_dir, *args, **kwargs):
            checkpoint_path = Path(checkpoint_dir)
            if checkpoint_path.parent == output_root and checkpoint_path.name.startswith(
                ".iter_0000030.tmp-"
            ):
                verify_calls.append(
                    {
                        "path": checkpoint_path,
                        "final_exists": final_dir.exists(),
                        "marker_exists": marker.exists(),
                    }
                )
            return real_load(state_dict, checkpoint_dir, *args, **kwargs)

        monkeypatch.setattr(weighted_merge_module.dist_checkpointing, "load", tracking_load)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            lambda: _template(),
            output_iteration=30,
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )

        assert result.output_dir == final_dir
        assert result.verified_load
        assert verify_calls
        assert len({call["path"] for call in verify_calls}) == 1
        verify_call = verify_calls[0]
        assert verify_call["path"].parent == output_root
        assert verify_call["path"].name.startswith(".iter_0000030.tmp-")
        assert all(not call["final_exists"] for call in verify_calls)
        assert all(not call["marker_exists"] for call in verify_calls)
        assert not verify_call["path"].exists()
        assert final_dir.exists()
        assert marker.read_text(encoding="utf-8").strip() == "30"


def test_direct_dcp_streaming_verify_load_failure_blocks_publication(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    real_load = weighted_merge_module.dist_checkpointing.load
    verify_paths = []

    def fail_hidden_temp_load(state_dict, checkpoint_dir, *args, **kwargs):
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.name.startswith(".iter_0000030.tmp-"):
            verify_paths.append(checkpoint_path)
            raise RuntimeError("injected hidden temp verify failure")
        return real_load(state_dict, checkpoint_dir, *args, **kwargs)

    monkeypatch.setattr(weighted_merge_module.dist_checkpointing, "load", fail_hidden_temp_load)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_fail_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_fail_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_verify_fail_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(RuntimeError, match="injected hidden temp verify failure"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
                verify_load=True,
            )

        assert len(verify_paths) == 1
        _assert_no_direct_atomic_publication(output_root, 30)
        assert verify_paths[0].exists()
        torch_dcp.FileSystemReader(verify_paths[0]).read_metadata()
        assert (verify_paths[0] / "metadata.json").exists()
        assert (verify_paths[0] / "common.pt").exists()


def test_metadata_same_layout_merge_round_trip_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    shape = (5, 2)

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_model_path)
    monkeypatch.setattr(weighted_merge_module, "_state_dict_for_execution_mode", fail_model_path)
    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1, shape=shape)
        _write_checkpoint(ckpt_b, 5.0, extra_value=999.0, iteration=2, shape=shape)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=30,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert result.averaged_tensors == 2
        assert result.copied_extra_states == 1
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"

        loaded = _load_checkpoint(result.output_dir, shape=shape)
        assert torch.equal(loaded["model"]["weight"], torch.full(shape, 4.0))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))

        dcp_loaded = {
            "model.weight": torch.empty(shape, dtype=torch.float32),
            "model.bias": torch.empty((2,), dtype=torch.float32),
            "model.decoder.layers.0._extra_state": torch.empty((1,), dtype=torch.float32),
        }
        torch_dcp.load(dcp_loaded, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(dcp_loaded["model.weight"], torch.full(shape, 4.0))
        assert torch.equal(dcp_loaded["model.bias"], torch.full((2,), 5.0))
        assert torch.equal(dcp_loaded["model.decoder.layers.0._extra_state"], torch.tensor([999.0]))

        source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
        output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        for fqn in (
            "model.weight",
            "model.bias",
            "model.decoder.layers.0._extra_state",
        ):
            source_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in source_metadata.state_dict_metadata[fqn].chunks
            ]
            output_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in output_metadata.state_dict_metadata[fqn].chunks
            ]
            assert output_chunks == source_chunks

        common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
        provenance = common_state["weighted_merge_provenance"]
        assert provenance["implementation_mode"] == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert provenance["extra_state_source_index"] == 1


def test_metadata_same_layout_generated_gpt_round_trip_cpu_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_model_path)
    monkeypatch.setattr(weighted_merge_module, "_state_dict_for_execution_mode", fail_model_path)
    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1)
    try:
        with (
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_b") as ckpt_b,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_gpt_out") as output_root,
        ):
            _write_generated_gpt_checkpoint(ckpt_a, 1.0)
            _write_generated_gpt_checkpoint(ckpt_b, 5.0)

            result = merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=40,
            )

            assert result.output_dir == output_root / "iter_0000040"
            assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert result.averaged_tensors >= 15
            assert result.copied_extra_states == 0
            assert result.memory_estimate.template_devices == ("cpu",)

            output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
            tensor_state = {
                fqn: torch.empty(
                    tuple(int(dim) for dim in metadata.size),
                    dtype=metadata.properties.dtype,
                )
                for fqn, metadata in output_metadata.state_dict_metadata.items()
            }
            assert any("model.decoder.layers" in fqn for fqn in tensor_state)
            assert "model.embedding.word_embeddings.weight" in tensor_state
            assert "model.output_layer.weight" in tensor_state
            assert not any(fqn.endswith("._extra_state") for fqn in tensor_state)

            torch_dcp.load(tensor_state, checkpoint_id=str(result.output_dir), no_dist=True)
            for fqn, tensor in tensor_state.items():
                assert torch.equal(tensor, torch.full_like(tensor, 4.0)), fqn

            source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
            for fqn, output_tensor_metadata in output_metadata.state_dict_metadata.items():
                source_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in source_metadata.state_dict_metadata[fqn].chunks
                ]
                output_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in output_tensor_metadata.chunks
                ]
                assert output_chunks == source_chunks
    finally:
        ps.destroy_model_parallel()


def test_metadata_same_layout_generated_moe_gpt_round_trip_cpu_without_model_builder_path(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_model_path)
    monkeypatch.setattr(weighted_merge_module, "_state_dict_for_execution_mode", fail_model_path)
    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    ps.destroy_model_parallel()
    ps.initialize_model_parallel(1, 1)
    try:
        with (
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_a") as ckpt_a,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_b") as ckpt_b,
            TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_generated_moe_gpt_out") as output_root,
        ):
            _write_generated_moe_gpt_checkpoint(ckpt_a, 1.0)
            _write_generated_moe_gpt_checkpoint(ckpt_b, 5.0)

            result = merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=41,
                extra_state_source_index=1,
            )

            assert result.output_dir == output_root / "iter_0000041"
            assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert result.averaged_tensors == 19
            assert result.copied_extra_states == 6
            assert result.memory_estimate.template_devices == ("cpu",)

            output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
            object_metadata_keys = [
                fqn
                for fqn, metadata in output_metadata.state_dict_metadata.items()
                if not hasattr(metadata, "size")
            ]
            assert len(object_metadata_keys) == 6
            assert all("_extra_state" in fqn for fqn in object_metadata_keys)
            tensor_state = {
                fqn: torch.empty(
                    tuple(int(dim) for dim in metadata.size),
                    dtype=metadata.properties.dtype,
                )
                for fqn, metadata in output_metadata.state_dict_metadata.items()
                if hasattr(metadata, "size")
            }
            assert any(fqn.endswith(".mlp.router.weight") for fqn in tensor_state)
            assert any(fqn.endswith(".mlp.router.bias") for fqn in tensor_state)
            expert_tensor_keys = [fqn for fqn in tensor_state if ".mlp.experts." in fqn]
            assert len(expert_tensor_keys) == 4
            assert not any(fqn.endswith("._extra_state") for fqn in tensor_state)

            torch_dcp.load(tensor_state, checkpoint_id=str(result.output_dir), no_dist=True)
            for fqn, tensor in tensor_state.items():
                assert torch.equal(tensor, torch.full_like(tensor, 4.0)), fqn

            source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
            for fqn in tensor_state:
                output_tensor_metadata = output_metadata.state_dict_metadata[fqn]
                source_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in source_metadata.state_dict_metadata[fqn].chunks
                ]
                output_chunks = [
                    (tuple(chunk.offsets), tuple(chunk.sizes))
                    for chunk in output_tensor_metadata.chunks
                ]
                assert output_chunks == source_chunks
    finally:
        ps.destroy_model_parallel()


def test_metadata_same_layout_unprefixed_gpt_model_tensor_roots_round_trip(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_gpt_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=50,
        )

        assert result.output_dir == output_root / "iter_0000050"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 0

        public_state = {
            "decoder.final_layernorm.weight": torch.empty((3,), dtype=torch.float32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.empty((2, 3), dtype=torch.float32),
            "embedding.word_embeddings.weight": torch.empty((4, 3), dtype=torch.float32),
            "output_layer.weight": torch.empty((4, 3), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)

        assert torch.equal(public_state["decoder.final_layernorm.weight"], torch.full((3,), 4.0))
        assert torch.equal(
            public_state["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), 5.0),
        )
        assert torch.equal(public_state["embedding.word_embeddings.weight"], torch.full((4, 3), 6.0))
        assert torch.equal(public_state["output_layer.weight"], torch.full((4, 3), 7.0))


def test_metadata_same_layout_copies_unprefixed_byte_extra_state_from_selected_source(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_b, 5.0, 999)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=51,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000051"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 1

        public_state = {
            "decoder.final_layernorm.weight": torch.empty((3,), dtype=torch.float32),
            "decoder.layers.0.mlp.linear_fc1.weight": torch.empty((2, 3), dtype=torch.float32),
            "embedding.word_embeddings.weight": torch.empty((4, 3), dtype=torch.float32),
            "output_layer.weight": torch.empty((4, 3), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(public_state["decoder.final_layernorm.weight"], torch.full((3,), 4.0))
        assert torch.equal(
            public_state["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), 5.0),
        )
        assert torch.equal(public_state["embedding.word_embeddings.weight"], torch.full((4, 3), 6.0))
        assert torch.equal(public_state["output_layer.weight"], torch.full((4, 3), 7.0))

        object_key = f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_0_1"
        loaded = dist_checkpointing.load(
            {object_key: ShardedObject.empty_from_unique_key(object_key)},
            str(result.output_dir),
            validate_access_integrity=False,
        )
        assert _decode_sharded_object_value(loaded[object_key]) == {"value": 999}

        normal_loaded = dist_checkpointing.load(
            _unprefixed_gpt_like_byte_extra_template(),
            str(result.output_dir),
        )
        normal_extra_state = normal_loaded[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY]
        normal_extra_state.seek(0)
        assert torch.load(normal_extra_state, weights_only=False) == {"value": 999}


def test_metadata_same_layout_accepts_unprefixed_mtp_tensor_and_byte_extra_state(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_mtp_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_mtp_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint_with_mtp_state(ckpt_b, 5.0, 999)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=52,
            extra_state_source_index=1,
        )

        assert result.output_dir == output_root / "iter_0000052"
        assert result.averaged_tensors == 5
        assert result.copied_extra_states == 1

        public_state = {
            "mtp.layers.0.eh_proj.weight": torch.empty((2, 2), dtype=torch.float32),
        }
        torch_dcp.load(public_state, checkpoint_id=str(result.output_dir), no_dist=True)
        assert torch.equal(
            public_state["mtp.layers.0.eh_proj.weight"],
            torch.full((2, 2), 8.0),
        )

        object_key = f"{UNPREFIXED_MTP_BYTE_EXTRA_STATE_KEY}/shard_0_1"
        loaded = dist_checkpointing.load(
            {object_key: ShardedObject.empty_from_unique_key(object_key)},
            str(result.output_dir),
            validate_access_integrity=False,
        )
        assert _decode_sharded_object_value(loaded[object_key]) == {"value": 999}


def test_metadata_same_layout_rejects_byte_objects_outside_model_roots(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_outside_bytes_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint_with_outside_byte_extra_state(ckpt_b, 5.0)

        with pytest.raises(WeightedMergeError, match="byte/object DCP entries outside model roots"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=53,
            )


def test_metadata_same_layout_rejects_mismatched_byte_extra_state_keys(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_byte_mismatch_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(ckpt_a, 1.0, 111)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        with pytest.raises(
            WeightedMergeError,
            match="identical byte/object _extra_state key sets",
        ):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=54,
            )


def test_metadata_same_layout_rejects_unprefixed_non_model_tensor_root(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unprefixed_optim_out") as output_root,
    ):
        _write_unprefixed_gpt_like_checkpoint_with_optimizer_tensor(ckpt_a, 1.0)
        _write_unprefixed_gpt_like_checkpoint(ckpt_b, 5.0)

        with pytest.raises(WeightedMergeError, match="non-model DCP tensor keys"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=52,
            )


def test_metadata_same_layout_cli_dispatch_skips_megatron_parser(tmp_path, monkeypatch):
    ckpt_a = tmp_path / "iter_0000001"
    ckpt_b = tmp_path / "iter_0000002"
    output_root = tmp_path / "merged"
    calls = {}

    def fail_megatron_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout CLI must not use Megatron parsing")

    def fake_merge(input_paths, weights, output_root_arg, **kwargs):
        calls["input_paths"] = input_paths
        calls["weights"] = weights
        calls["output_root"] = output_root_arg
        calls["kwargs"] = kwargs
        return weighted_merge_module.MergeResult(
            output_dir=Path(output_root_arg) / "iter_0000030",
            input_dirs=tuple(Path(path) for path in input_paths),
            weights=tuple(weights),
            timings=weighted_merge_module.MergeTimings(),
            averaged_tensors=2,
            copied_extra_states=1,
            backend="torch_dist",
            implementation_mode=weighted_merge_module.METADATA_SAME_LAYOUT_MODE,
        )

    monkeypatch.setattr(weighted_merge_module, "parse_and_validate_merge_args", fail_megatron_path)
    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_megatron_path)
    monkeypatch.setattr(
        weighted_merge_module,
        "merge_same_layout_dcp_metadata_checkpoints",
        fake_merge,
    )

    args = weighted_merge_module._parse_metadata_same_layout_args(
        [
            "--merge-execution-mode",
            weighted_merge_module.METADATA_SAME_LAYOUT_MODE,
            "--merge-inputs",
            f"{ckpt_a}:0.25",
            f"{ckpt_b}:0.75",
            "--merge-output",
            str(output_root),
            "--output-iteration",
            "30",
            "--extra-state-source-index",
            "1",
            "--ckpt-format",
            "torch_dist",
        ]
    )
    result = weighted_merge_module._run_metadata_same_layout_cli(args)

    assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
    assert calls["input_paths"] == [ckpt_a, ckpt_b]
    assert calls["weights"] == [0.25, 0.75]
    assert calls["output_root"] == str(output_root)
    assert calls["kwargs"]["output_iteration"] == 30
    assert calls["kwargs"]["extra_state_source_index"] == 1


def test_metadata_same_layout_cli_rejects_unsupported_template_options(tmp_path):
    base_args = [
        "--merge-execution-mode",
        weighted_merge_module.METADATA_SAME_LAYOUT_MODE,
        "--merge-inputs",
        f"{tmp_path / 'iter_0000001'}:1.0",
        "--merge-output",
        str(tmp_path / "merged"),
    ]
    with pytest.raises(WeightedMergeError, match="--verify-load is not supported"):
        weighted_merge_module._parse_metadata_same_layout_args(base_args + ["--verify-load"])
    with pytest.raises(WeightedMergeError, match="--merge-window-btoks is not supported"):
        weighted_merge_module._parse_metadata_same_layout_args(
            base_args + ["--end-checkpoint", "2", "--merge-window-btoks", "1"]
        )
    with pytest.raises(WeightedMergeError, match="--no-atomic-merge-output"):
        weighted_merge_module._parse_metadata_same_layout_args(
            base_args + ["--no-atomic-merge-output"]
        )
    with pytest.raises(WeightedMergeError, match="template/file-backed options"):
        weighted_merge_module._parse_metadata_same_layout_args(
            base_args + ["--merge-template-init-mode", "meta"]
        )


def test_metadata_same_layout_rejects_no_atomic_output(tmp_path_dist_ckpt, process_group):
    with pytest.raises(WeightedMergeError, match="requires atomic output publication"):
        merge_same_layout_dcp_metadata_checkpoints(
            [tmp_path_dist_ckpt / "missing"],
            [1.0],
            tmp_path_dist_ckpt / "merged",
            atomic_output=False,
        )


def test_metadata_same_layout_rejects_existing_output_overwrite_for_crash_safety(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_b") as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_overwrite_reject_out"
        ) as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)
        final_dir = output_root / "iter_0000030"
        final_dir.mkdir()
        _write_checkpoint(final_dir, 9.0, iteration=29)
        write_latest_checkpointed_iteration(final_dir, 29)

        with pytest.raises(WeightedMergeError, match="crash-atomic"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                output_iteration=30,
                overwrite_output=True,
            )

        restored = _load_checkpoint(final_dir)
        torch.testing.assert_close(restored["model"]["weight"], torch.full((2, 2), 9.0))
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "29"
        assert not list(output_root.glob(".iter_0000030.old-*"))
        assert not list(output_root.glob(".iter_0000030.tmp-*"))


def test_metadata_same_layout_rejects_mismatched_metadata(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_metadata_mismatch_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, shape=(2, 2), iteration=1)
        _write_checkpoint(ckpt_b, 5.0, shape=(3, 2), iteration=2)

        with pytest.raises(WeightedMergeError, match="Shape mismatch.*metadata-same-layout"):
            merge_same_layout_dcp_metadata_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root / "merged",
            )

        assert not (output_root / "merged").exists()


def test_direct_dcp_streaming_rejects_no_atomic_output(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_no_atomic_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_no_atomic_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_no_atomic_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(
            WeightedMergeError,
            match="direct-dcp-streaming requires atomic output publication",
        ):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                atomic_output=False,
                streaming_chunk_bytes=16,
            )

        _assert_no_direct_atomic_publication(output_root, 30)


def test_direct_dcp_streaming_planner_failure_does_not_publish_final_output(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    resolve_calls = []

    def fail_resolve_data(self, write_item):
        resolve_calls.append(write_item)
        raise RuntimeError("injected direct planner failure")

    monkeypatch.setattr(
        weighted_merge_module._WeightedMergeDirectOutputSavePlanner,
        "resolve_data",
        fail_resolve_data,
    )
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_plan_fail_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_plan_fail_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_plan_fail_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(WeightedMergeError, match="Direct DCP streaming save failed"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
            )

        assert resolve_calls
        _assert_no_direct_atomic_publication(output_root, 30)
        temporary_dirs = list(output_root.glob(".iter_0000030.tmp-*"))
        assert len(temporary_dirs) == 1
        assert not (temporary_dirs[0] / "metadata.json").exists()


@pytest.mark.parametrize("failing_sidecar", ("save_common", "save_config"))
def test_direct_dcp_streaming_sidecar_failure_after_dcp_save_does_not_publish_final_output(
    tmp_path_dist_ckpt, process_group, monkeypatch, failing_sidecar
):
    if failing_sidecar == "save_common":
        from megatron.core.dist_checkpointing.strategies import common as sidecar_module
    else:
        from megatron.core.dist_checkpointing import core as sidecar_module

    def fail_sidecar(*_args, **_kwargs):
        raise RuntimeError(f"injected {failing_sidecar} failure")

    monkeypatch.setattr(sidecar_module, failing_sidecar, fail_sidecar)
    with (
        TempNamedDir(tmp_path_dist_ckpt / f"weighted_merge_direct_{failing_sidecar}_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / f"weighted_merge_direct_{failing_sidecar}_b") as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / f"weighted_merge_direct_{failing_sidecar}_out"
        ) as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(
            (RuntimeError, WeightedMergeError),
            match=f"injected {failing_sidecar} failure",
        ):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
            )

        _assert_no_direct_atomic_publication(output_root, 30)
        temporary_dirs = list(output_root.glob(".iter_0000030.tmp-*"))
        assert len(temporary_dirs) == 1
        torch_dcp.FileSystemReader(temporary_dirs[0]).read_metadata()
        assert not (temporary_dirs[0] / "metadata.json").exists()


def test_direct_dcp_streaming_sigkill_before_publish_does_not_expose_public_output(tmp_path):
    if not hasattr(os, "killpg") or not hasattr(signal, "SIGKILL"):
        pytest.skip("pre-publish hard-kill proof requires POSIX process groups")

    output_root = tmp_path / "weighted_merge_direct_sigkill_out"
    sentinel_dir = tmp_path / "sentinels"
    repo_root = Path(__file__).resolve().parents[4]
    child_script_path = (
        repo_root / "tests/unit_tests/tools/checkpoint/direct_output_rank_loss_repro.py"
    )
    stdout_path = tmp_path / "torchrun-one-rank.stdout"
    stderr_path = tmp_path / "torchrun-one-rank.stderr"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["WM_DIRECT_RANK_LOSS_OUTPUT_ROOT"] = str(output_root)
    env["WM_DIRECT_RANK_LOSS_SENTINEL_DIR"] = str(sentinel_dir)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=1",
            "--master-addr=127.0.0.1",
            f"--master-port={_unused_tcp_port()}",
            str(child_script_path),
        ],
        cwd=repo_root,
        env=env,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
        start_new_session=True,
    )

    def child_logs():
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return stdout, stderr

    def kill_child_group():
        for pid_path in sentinel_dir.glob("rank*.pid"):
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + 60
    sentinel = sentinel_dir / "rank0.before_publish"
    while time.monotonic() < deadline and not sentinel.exists():
        if process.poll() is not None:
            stdout, stderr = child_logs()
            pytest.fail(
                "child exited before reaching the pre-publish kill point\n"
                f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.05)

    if not sentinel.exists():
        kill_child_group()
        process.wait(timeout=10)
        stdout, stderr = child_logs()
        pytest.fail(
            "child did not reach the pre-publish kill point\n"
            f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
        )

    kill_child_group()
    process.wait(timeout=10)
    stdout, stderr = child_logs()
    assert process.returncode == -signal.SIGKILL, f"stdout={stdout}\nstderr={stderr}"

    temporary_dir = Path(sentinel.read_text(encoding="utf-8").strip())
    assert temporary_dir.exists()
    torch_dcp.FileSystemReader(temporary_dir).read_metadata()
    assert (temporary_dir / "metadata.json").exists()
    assert (temporary_dir / "common.pt").exists()
    assert not (output_root / "iter_0000030").exists()
    assert not (output_root / "latest_checkpointed_iteration.txt").exists()


def test_direct_dcp_streaming_sigkill_after_publish_before_marker_keeps_checkpoint(
    tmp_path, process_group
):
    if not hasattr(os, "killpg") or not hasattr(signal, "SIGKILL"):
        pytest.skip("after-publish hard-kill proof requires POSIX process groups")

    output_root = tmp_path / "weighted_merge_direct_after_publish_sigkill_out"
    sentinel_dir = tmp_path / "sentinels"
    repo_root = Path(__file__).resolve().parents[4]
    child_script_path = (
        repo_root / "tests/unit_tests/tools/checkpoint/direct_output_rank_loss_repro.py"
    )
    stdout_path = tmp_path / "torchrun-after-publish.stdout"
    stderr_path = tmp_path / "torchrun-after-publish.stderr"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["WM_DIRECT_RANK_LOSS_BLOCK_POINT"] = "after_publish_before_marker"
    env["WM_DIRECT_RANK_LOSS_OUTPUT_ROOT"] = str(output_root)
    env["WM_DIRECT_RANK_LOSS_SENTINEL_DIR"] = str(sentinel_dir)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=1",
            "--master-addr=127.0.0.1",
            f"--master-port={_unused_tcp_port()}",
            str(child_script_path),
        ],
        cwd=repo_root,
        env=env,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
        start_new_session=True,
    )

    def child_logs():
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return stdout, stderr

    def kill_child_group():
        for pid_path in sentinel_dir.glob("rank*.pid"):
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        deadline = time.monotonic() + 60
        sentinel = sentinel_dir / "rank0.after_publish_before_marker"
        while time.monotonic() < deadline and not sentinel.exists():
            if process.poll() is not None:
                stdout, stderr = child_logs()
                pytest.fail(
                    "child exited before reaching the after-publish kill point\n"
                    f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.05)

        if not sentinel.exists():
            kill_child_group()
            process.wait(timeout=10)
            stdout, stderr = child_logs()
            pytest.fail(
                "child did not reach the after-publish kill point\n"
                f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
            )

        final_dir = Path(sentinel.read_text(encoding="utf-8").strip())
        assert final_dir == output_root / "iter_0000030"
        assert final_dir.exists()
        torch_dcp.FileSystemReader(final_dir).read_metadata()
        assert (final_dir / "metadata.json").exists()
        assert (final_dir / "common.pt").exists()
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()
        loaded = _load_checkpoint(final_dir)
        torch.testing.assert_close(loaded["model"]["weight"], torch.full((2, 2), 4.0))

        kill_child_group()
        process.wait(timeout=10)
        stdout, stderr = child_logs()
        assert process.returncode == -signal.SIGKILL, f"stdout={stdout}\nstderr={stderr}"
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()
    finally:
        kill_child_group()


def test_direct_dcp_streaming_two_rank_sigkill_before_publish_is_bounded_and_hidden(tmp_path):
    if not hasattr(os, "killpg") or not hasattr(os, "kill") or not hasattr(signal, "SIGKILL"):
        pytest.skip("distributed hard-kill proof requires POSIX process groups")

    output_root = tmp_path / "weighted_merge_direct_two_rank_sigkill_out"
    sentinel_dir = tmp_path / "sentinels"
    repo_root = Path(__file__).resolve().parents[4]
    child_script_path = (
        repo_root / "tests/unit_tests/tools/checkpoint/direct_output_rank_loss_repro.py"
    )
    stdout_path = tmp_path / "torchrun.stdout"
    stderr_path = tmp_path / "torchrun.stderr"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["WM_DIRECT_RANK_LOSS_OUTPUT_ROOT"] = str(output_root)
    env["WM_DIRECT_RANK_LOSS_SENTINEL_DIR"] = str(sentinel_dir)
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={_unused_tcp_port()}",
            str(child_script_path),
        ],
        cwd=repo_root,
        env=env,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
        start_new_session=True,
    )

    def child_logs():
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        return stdout, stderr

    def kill_child_group():
        for pid_path in sentinel_dir.glob("rank*.pid"):
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                pass
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    try:
        prepublish_markers = [
            sentinel_dir / "rank0.before_publish",
            sentinel_dir / "rank1.before_publish",
        ]
        pid_files = [sentinel_dir / "rank0.pid", sentinel_dir / "rank1.pid"]
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not all(path.exists() for path in prepublish_markers):
            if process.poll() is not None:
                stdout, stderr = child_logs()
                pytest.fail(
                    "two-rank child exited before both ranks reached the pre-publish kill point\n"
                    f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.05)

        if not all(path.exists() for path in prepublish_markers):
            kill_child_group()
            process.wait(timeout=10)
            stdout, stderr = child_logs()
            pytest.fail(
                "two-rank child did not reach the pre-publish kill point before timeout\n"
                f"returncode={process.returncode}\nstdout={stdout}\nstderr={stderr}"
            )
        assert all(path.exists() for path in pid_files)

        rank1_pid = int((sentinel_dir / "rank1.pid").read_text(encoding="utf-8").strip())
        os.kill(rank1_pid, signal.SIGKILL)

        launcher_exited_after_rank_loss = True
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            launcher_exited_after_rank_loss = False
            kill_child_group()
            process.wait(timeout=10)

        stdout, stderr = child_logs()
        assert process.returncode != 0, f"stdout={stdout}\nstderr={stderr}"
        assert process.returncode is not None
        assert not launcher_exited_after_rank_loss or process.returncode != 0

        temporary_dirs = {
            Path(path.read_text(encoding="utf-8").strip()) for path in prepublish_markers
        }
        assert len(temporary_dirs) == 1
        temporary_dir = temporary_dirs.pop()
        assert temporary_dir.exists(), f"stdout={stdout}\nstderr={stderr}"
        torch_dcp.FileSystemReader(temporary_dir).read_metadata()
        assert (temporary_dir / "metadata.json").exists()
        assert (temporary_dir / "common.pt").exists()
        assert not (output_root / "iter_0000030").exists()
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()
    finally:
        kill_child_group()


def test_direct_dcp_streaming_latest_marker_failure_is_after_publication(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    real_replace = weighted_merge_module.os.replace
    real_load = weighted_merge_module.dist_checkpointing.load
    verify_paths = []

    def fail_latest_marker_replace(src, dst):
        if Path(dst).name == "latest_checkpointed_iteration.txt":
            assert verify_paths
            raise OSError("injected latest marker failure")
        real_replace(src, dst)

    def tracking_load(state_dict, checkpoint_dir, *args, **kwargs):
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.parent == output_root and checkpoint_path.name.startswith(
            ".iter_0000030.tmp-"
        ):
            verify_paths.append(checkpoint_path)
        return real_load(state_dict, checkpoint_dir, *args, **kwargs)

    monkeypatch.setattr(weighted_merge_module.os, "replace", fail_latest_marker_replace)
    monkeypatch.setattr(weighted_merge_module.dist_checkpointing, "load", tracking_load)
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_marker_fail_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_marker_fail_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_marker_fail_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(OSError, match="injected latest marker failure"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
                verify_load=True,
            )

        final_dir = output_root / "iter_0000030"
        assert verify_paths
        assert len(set(verify_paths)) == 1
        assert verify_paths[0].parent == output_root
        assert verify_paths[0].name.startswith(".iter_0000030.tmp-")
        assert not verify_paths[0].exists()
        assert final_dir.exists()
        torch_dcp.FileSystemReader(final_dir).read_metadata()
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()


def test_direct_dcp_streaming_two_rank_product_round_trip_public_metadata(
    tmp_path_dist_ckpt, process_group
):
    if _world_size() != 2:
        pytest.skip("two-rank direct-DCP product coverage requires torchrun --nproc_per_node=2")

    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_two_rank_a", sync=True) as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_two_rank_b", sync=True) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_direct_two_rank_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_checkpoint(ckpt_a, 1.0 + rank, extra_value=111.0 + rank, iteration=1)
        _write_checkpoint(ckpt_b, 5.0 + (2 * rank), extra_value=999.0 + rank, iteration=2)

        direct = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            lambda: _template(),
            output_iteration=30,
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )
        file_backed = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "file_backed",
            lambda: _template(),
            execution_mode="file-backed-streaming",
            streaming_chunk_bytes=16,
            verify_load=True,
        )

        direct_loaded = _load_checkpoint(direct.output_dir)
        file_backed_loaded = _load_checkpoint(file_backed.output_dir)
        assert direct.implementation_mode == "direct-dcp-streaming"
        assert direct.world_size == 2
        assert direct.verified_load
        assert torch.equal(
            direct_loaded["model"]["weight"], file_backed_loaded["model"]["weight"]
        )
        assert torch.equal(direct_loaded["model"]["bias"], file_backed_loaded["model"]["bias"])
        assert torch.equal(
            direct_loaded["model"]["decoder.layers.0._extra_state"],
            file_backed_loaded["model"]["decoder.layers.0._extra_state"],
        )
        expected_rank_weight = 4.0 + (1.75 * rank)
        assert torch.equal(
            direct_loaded["model"]["weight"], torch.full((2, 2), expected_rank_weight)
        )
        assert torch.equal(
            direct_loaded["model"]["bias"], torch.full((2,), expected_rank_weight + 1.0)
        )
        assert torch.equal(
            direct_loaded["model"]["decoder.layers.0._extra_state"],
            torch.tensor([111.0 + rank]),
        )

        direct_public = _full_public_dcp_state(direct.output_dir)
        file_backed_public = _full_public_dcp_state(file_backed.output_dir)
        expected_public_weight = torch.cat(
            [torch.full((2, 2), 4.0), torch.full((2, 2), 5.75)], dim=0
        )
        expected_public_bias = torch.tensor([5.0, 5.0, 6.75, 6.75])
        expected_public_extra = torch.tensor([111.0, 112.0])
        assert torch.equal(direct_public["model.weight"], file_backed_public["model.weight"])
        assert torch.equal(direct_public["model.bias"], file_backed_public["model.bias"])
        assert torch.equal(
            direct_public["model.decoder.layers.0._extra_state"],
            file_backed_public["model.decoder.layers.0._extra_state"],
        )
        assert torch.equal(direct_public["model.weight"], expected_public_weight)
        assert torch.equal(direct_public["model.bias"], expected_public_bias)
        assert torch.equal(
            direct_public["model.decoder.layers.0._extra_state"],
            expected_public_extra,
        )

        metadata_summary = _dcp_metadata_summary(direct.output_dir)
        assert not metadata_summary["duplicate_chunk_offsets"]
        assert not metadata_summary["duplicate_storage_records"]
        assert metadata_summary["storage_file_count"] == 2
        assert len(metadata_summary["chunk_records"]) == 6

        if _rank() == 0:
            sidecars = {
                path.name
                for path in direct.output_dir.iterdir()
                if path.name != ".metadata" and not path.name.endswith(".distcp")
            }
            assert sidecars == {"common.pt", "metadata.json"}
            assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"
            assert len(list(direct.output_dir.glob("*.distcp"))) == 2


def test_metadata_same_layout_two_rank_product_round_trip_public_metadata(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    if _world_size() != 2:
        pytest.skip("two-rank metadata same-layout coverage requires torchrun --nproc_per_node=2")

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_model_path)
    monkeypatch.setattr(weighted_merge_module, "_state_dict_for_execution_mode", fail_model_path)
    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_a", sync=True
        ) as ckpt_a,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_b", sync=True
        ) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_checkpoint(ckpt_a, 1.0 + rank, extra_value=111.0 + rank, iteration=1)
        _write_checkpoint(ckpt_b, 5.0 + (2 * rank), extra_value=999.0 + rank, iteration=2)

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=30,
            extra_state_source_index=1,
        )

        loaded = _load_checkpoint(result.output_dir)
        assert result.output_dir == output_root / "iter_0000030"
        assert result.implementation_mode == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
        assert result.world_size == 2
        assert result.averaged_tensors == 2
        assert result.copied_extra_states == 1
        assert result.memory_estimate.template_devices == ("cpu",)
        assert result.file_backed_staging_layout == "none"

        expected_rank_weight = 4.0 + (1.75 * rank)
        assert torch.equal(loaded["model"]["weight"], torch.full((2, 2), expected_rank_weight))
        assert torch.equal(loaded["model"]["bias"], torch.full((2,), expected_rank_weight + 1.0))
        assert torch.equal(
            loaded["model"]["decoder.layers.0._extra_state"],
            torch.tensor([999.0 + rank]),
        )

        public = _full_public_dcp_state(result.output_dir)
        expected_public_weight = torch.cat(
            [torch.full((2, 2), 4.0), torch.full((2, 2), 5.75)], dim=0
        )
        expected_public_bias = torch.tensor([5.0, 5.0, 6.75, 6.75])
        expected_public_extra = torch.tensor([999.0, 1000.0])
        assert torch.equal(public["model.weight"], expected_public_weight)
        assert torch.equal(public["model.bias"], expected_public_bias)
        assert torch.equal(public["model.decoder.layers.0._extra_state"], expected_public_extra)

        source_metadata = torch_dcp.FileSystemReader(ckpt_a).read_metadata()
        output_metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        for fqn in (
            "model.weight",
            "model.bias",
            "model.decoder.layers.0._extra_state",
        ):
            source_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in source_metadata.state_dict_metadata[fqn].chunks
            ]
            output_chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in output_metadata.state_dict_metadata[fqn].chunks
            ]
            assert output_chunks == source_chunks

        metadata_summary = _dcp_metadata_summary(result.output_dir)
        assert not metadata_summary["duplicate_chunk_offsets"]
        assert not metadata_summary["duplicate_storage_records"]
        assert metadata_summary["storage_file_count"] == 2
        assert len(metadata_summary["chunk_records"]) == 6

        if _rank() == 0:
            common_state = dist_checkpointing.load_common_state_dict(str(result.output_dir))
            provenance = common_state["weighted_merge_provenance"]
            assert provenance["implementation_mode"] == weighted_merge_module.METADATA_SAME_LAYOUT_MODE
            assert provenance["extra_state_source_index"] == 1
            assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"
            assert len(list(result.output_dir.glob("*.distcp"))) == 2


def test_metadata_same_layout_two_rank_byte_extra_state_round_trip(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    if _world_size() != 2:
        pytest.skip("two-rank metadata same-layout byte coverage requires torchrun --nproc_per_node=2")

    def fail_model_path(*_args, **_kwargs):
        raise AssertionError("metadata same-layout must not use model/template construction")

    monkeypatch.setattr(weighted_merge_module, "_build_model_state_dict_factory", fail_model_path)
    monkeypatch.setattr(weighted_merge_module, "_state_dict_for_execution_mode", fail_model_path)
    monkeypatch.setattr(
        weighted_merge_module.dist_checkpointing,
        "load_tensors_metadata",
        fail_model_path,
    )

    with (
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_a", sync=True
        ) as ckpt_a,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_b", sync=True
        ) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_metadata_two_rank_bytes_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
            ckpt_a,
            1.0 + rank,
            111 + rank,
            rank_sharded=True,
        )
        _write_unprefixed_gpt_like_checkpoint_with_byte_extra_state(
            ckpt_b,
            5.0 + (2 * rank),
            999 + rank,
            rank_sharded=True,
        )

        result = merge_same_layout_dcp_metadata_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root,
            output_iteration=31,
            extra_state_source_index=1,
        )

        load_template = _unprefixed_gpt_like_model_state(0.0, rank_sharded=True)
        load_template.update(_unprefixed_gpt_like_byte_extra_template())
        loaded = dist_checkpointing.load(load_template, str(result.output_dir))

        assert result.output_dir == output_root / "iter_0000031"
        assert result.averaged_tensors == 4
        assert result.copied_extra_states == 2
        expected_base = 4.0 + (1.75 * rank)
        assert torch.equal(
            loaded["decoder.final_layernorm.weight"],
            torch.full((3,), expected_base),
        )
        assert torch.equal(
            loaded["decoder.layers.0.mlp.linear_fc1.weight"],
            torch.full((2, 3), expected_base + 1.0),
        )
        assert torch.equal(
            loaded["embedding.word_embeddings.weight"],
            torch.full((4, 3), expected_base + 2.0),
        )
        assert torch.equal(
            loaded["output_layer.weight"],
            torch.full((4, 3), expected_base + 3.0),
        )
        loaded_extra_state = loaded[UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY]
        loaded_extra_state.seek(0)
        assert torch.load(loaded_extra_state, weights_only=False) == {"value": 999 + rank}

        metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        byte_keys = sorted(
            str(fqn)
            for fqn, entry in metadata.state_dict_metadata.items()
            if type(entry).__name__ == "BytesStorageMetadata"
        )
        assert byte_keys == [
            f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_0_2",
            f"{UNPREFIXED_GPT_BYTE_EXTRA_STATE_KEY}/shard_1_2",
        ]
        metadata_summary = _dcp_metadata_summary(result.output_dir)
        assert not metadata_summary["duplicate_chunk_offsets"]
        assert not metadata_summary["duplicate_storage_records"]
        assert metadata_summary["storage_file_count"] == 2

        if _rank() == 0:
            assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "31"
            assert len(list(result.output_dir.glob("*.distcp"))) == 2


def test_direct_dcp_streaming_fails_clear_for_unsupported_factory(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_factory_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_factory_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_factory_out") as output_root,
    ):
        _write_factory_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_factory_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(WeightedMergeError, match="ShardedTensorFactory"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root / "merged",
                lambda: _factory_template(),
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
            )


def test_direct_dcp_streaming_no_dist_branch_tracks_distributed_world_size(monkeypatch):
    monkeypatch.setattr(weighted_merge_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(weighted_merge_module.dist, "is_initialized", lambda: False)
    assert weighted_merge_module._direct_dcp_save_uses_no_dist() is True

    monkeypatch.setattr(weighted_merge_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(weighted_merge_module.dist, "get_world_size", lambda: 1)
    assert weighted_merge_module._direct_dcp_save_uses_no_dist() is True

    monkeypatch.setattr(weighted_merge_module.dist, "get_world_size", lambda: 2)
    assert weighted_merge_module._direct_dcp_save_uses_no_dist() is False


def test_direct_dcp_streaming_fails_clear_for_unsupported_prepended_axis_without_publication(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_prepended_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_prepended_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_prepended_out") as output_root,
    ):
        _write_prepended_axis_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_prepended_axis_checkpoint(ckpt_b, 5.0, iteration=2)

        with pytest.raises(WeightedMergeError, match="prepended-axis"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.25, 0.75],
                output_root,
                lambda: _prepended_axis_template(),
                output_iteration=30,
                execution_mode="direct-dcp-streaming",
                streaming_chunk_bytes=16,
            )

        assert not (output_root / "iter_0000030").exists()
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()


def test_direct_dcp_streaming_copies_object_extra_state(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_object_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_object_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_direct_object_out") as output_root,
    ):
        _write_object_extra_checkpoint(ckpt_a, 1.0, extra_value=111, iteration=1)
        _write_object_extra_checkpoint(ckpt_b, 5.0, extra_value=999, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _object_extra_template(),
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
        )
        loaded = dist_checkpointing.load(_object_extra_template(), str(result.output_dir))

        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 4.0))
        loaded["model"]["decoder.layers.0._extra_state"].seek(0)
        assert torch.load(loaded["model"]["decoder.layers.0._extra_state"]) == {"value": 111}

        metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        byte_keys = sorted(
            str(fqn)
            for fqn, entry in metadata.state_dict_metadata.items()
            if type(entry).__name__ == "BytesStorageMetadata"
        )
        assert byte_keys == ["model.decoder.layers.0._extra_state/shard_0_1"]


def test_direct_dcp_streaming_two_rank_copies_rank_local_object_extra_state(
    tmp_path_dist_ckpt, process_group
):
    if _world_size() != 2:
        pytest.skip("rank-local direct-DCP byte/object coverage requires torchrun --nproc_per_node=2")

    with (
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_direct_rank_local_object_a", sync=True
        ) as ckpt_a,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_direct_rank_local_object_b", sync=True
        ) as ckpt_b,
        TempNamedDir(
            tmp_path_dist_ckpt / "weighted_merge_direct_rank_local_object_out", sync=True
        ) as output_root,
    ):
        rank = _rank()
        _write_rank_local_object_extra_checkpoint(
            ckpt_a, 1.0 + rank, extra_value=111 + rank, iteration=1
        )
        _write_rank_local_object_extra_checkpoint(
            ckpt_b, 5.0 + (2 * rank), extra_value=999 + rank, iteration=2
        )

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _rank_local_object_extra_template(),
            execution_mode="direct-dcp-streaming",
            streaming_chunk_bytes=16,
            extra_state_source_index=1,
            verify_load=True,
        )
        loaded = dist_checkpointing.load(
            _rank_local_object_extra_template(), str(result.output_dir)
        )

        expected_rank_weight = 4.0 + (1.75 * rank)
        assert torch.equal(
            loaded["model"]["weight"], torch.full((2, 2), expected_rank_weight)
        )
        local_extra_key = _rank_local_object_extra_key(rank)
        loaded["model"][local_extra_key].seek(0)
        assert torch.load(loaded["model"][local_extra_key], weights_only=False) == {
            "value": 999 + rank
        }

        metadata = torch_dcp.FileSystemReader(result.output_dir).read_metadata()
        byte_keys = sorted(
            str(fqn)
            for fqn, entry in metadata.state_dict_metadata.items()
            if type(entry).__name__ == "BytesStorageMetadata"
        )
        assert byte_keys == [
            "model.decoder.layers.0._extra_state/shard_0_1",
            "model.decoder.layers.1._extra_state/shard_0_1",
        ]
        assert not _dcp_metadata_summary(result.output_dir)["duplicate_storage_records"]


def test_merge_rejects_existing_output_directory_by_default(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_exists_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_exists_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        output_dir = output_root / "merged"
        output_dir.mkdir()

        with pytest.raises(WeightedMergeError, match="Output directory already exists"):
            merge_sharded_checkpoints([ckpt_a], [1.0], output_dir, lambda: _template())


def test_byte_accounting_none_skips_recursive_directory_size(
    tmp_path_dist_ckpt, process_group, monkeypatch
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_no_bytes_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_no_bytes_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)

        def fail_directory_size(_path):
            raise AssertionError("directory size should not be measured")

        monkeypatch.setattr("tools.checkpoint.weighted_merge._directory_size", fail_directory_size)
        result = merge_sharded_checkpoints(
            [ckpt_a],
            [1.0],
            output_root / "merged",
            lambda: _template(),
            byte_accounting="none",
        )

        assert result.bytes_read == 0
        assert result.bytes_written == 0


def test_latest_marker_requires_checkpoint_metadata(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000001"
    checkpoint_dir.mkdir()

    with pytest.raises(WeightedMergeError, match="metadata"):
        write_latest_checkpointed_iteration(checkpoint_dir, 1)


def test_merge_sharded_checkpoints_supports_multiple_model_chunks(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_multi_out") as output_root,
    ):
        _write_multi_chunk_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_multi_chunk_checkpoint(ckpt_b, 5.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.25, 0.75],
            output_root / "merged",
            lambda: _multi_chunk_template(),
        )
        loaded = dist_checkpointing.load(_multi_chunk_template(), str(result.output_dir))

        assert torch.allclose(loaded["model0"]["weight"], torch.full((2, 2), 4.0))
        assert torch.allclose(loaded["model1"]["weight"], torch.full((2, 2), 5.0))


def test_merge_save_dtype_policy(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.float16, iteration=1)
        _write_checkpoint(ckpt_b, 3.0, dtype=torch.float16, iteration=2)

        same = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "same",
            lambda: _template(dtype=torch.float16),
            save_dtype="same",
        )
        fp32 = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "fp32",
            lambda: _template(dtype=torch.float16),
            save_dtype="float32",
        )
        bf16 = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "bf16",
            lambda: _template(dtype=torch.float16),
            save_dtype="bfloat16",
        )
        fp16 = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "fp16",
            lambda: _template(dtype=torch.float16),
            save_dtype="float16",
        )

        assert (
            _load_checkpoint(same.output_dir, dtype=torch.float16)["model"]["weight"].dtype
            == torch.float16
        )
        assert (
            _load_checkpoint(fp32.output_dir, dtype=torch.float32)["model"]["weight"].dtype
            == torch.float32
        )
        assert (
            _load_checkpoint(bf16.output_dir, dtype=torch.bfloat16)["model"]["weight"].dtype
            == torch.bfloat16
        )
        assert (
            _load_checkpoint(fp16.output_dir, dtype=torch.float16)["model"]["weight"].dtype
            == torch.float16
        )


def test_merge_accumulates_fp16_inputs_in_fp32(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_precision_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 4096.0, dtype=torch.float16, iteration=1)
        _write_checkpoint(ckpt_b, 1.0, dtype=torch.float16, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _template(dtype=torch.float16),
            save_dtype="float32",
        )
        loaded = _load_checkpoint(result.output_dir, dtype=torch.float32)

        assert loaded["model"]["weight"].dtype == torch.float32
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 2048.5))


def test_merge_rejects_dtype_mismatch_with_same_policy(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_dtype_mismatch_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, dtype=torch.float32, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, dtype=torch.float16, iteration=2)

        with pytest.raises(WeightedMergeError, match="Dtype mismatch"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(dtype=torch.float32),
                save_dtype="same",
            )


def test_merge_rejects_non_floating_tensors(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_int_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1, dtype=torch.int64, iteration=1)
        _write_checkpoint(ckpt_b, 2, dtype=torch.int64, iteration=2)

        with pytest.raises(WeightedMergeError, match="non-floating"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(dtype=torch.int64),
            )


def test_merge_rejects_shape_mismatch(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_shape_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, shape=(2, 2), iteration=1)
        _write_checkpoint(ckpt_b, 2.0, shape=(3, 2), iteration=2)

        with pytest.raises(Exception, match="Shape mismatch|shape|model.weight"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(shape=(2, 2)),
            )


def test_file_backed_streaming_rejects_template_shape_mismatch(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_shape_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_shape_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_shape_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, shape=(3, 2), iteration=1)
        _write_checkpoint(ckpt_b, 2.0, shape=(3, 2), iteration=2)

        with pytest.raises(
            WeightedMergeError, match="template expects global shape|Shape mismatch"
        ):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(shape=(2, 2)),
                execution_mode="file-backed-streaming",
                streaming_chunk_bytes=16,
            )

        assert not (output_root / "merged").exists()


def test_merge_rejects_incompatible_checkpoint_keys(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_good") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_bad") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_bad_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        state_dict = _template(2.0, include_bias=False)
        state_dict["args"] = SimpleNamespace(iteration=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = 2
        dist_checkpointing.save(state_dict, str(ckpt_b))

        with pytest.raises(Exception, match="model.bias|Missing|missing|Unexpected|unexpected"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged", lambda: _template()
            )


def test_merge_allows_extra_checkpoint_keys_by_default(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        state_dict = _template(2.0)
        state_dict["model"]["extra"] = ShardedTensor.from_rank_offsets(
            "model.extra", torch.ones((2,), dtype=torch.float32), *_rank_offsets(), replica_id=0
        )
        state_dict["args"] = SimpleNamespace(iteration=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = 2
        dist_checkpointing.save(state_dict, str(ckpt_b))

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged", lambda: _template()
        )
        loaded = _load_checkpoint(result.output_dir)

        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 1.5))


def test_merge_rejects_extra_checkpoint_keys_with_raise_all(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_strict_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_strict_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_strict_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        state_dict = _template(2.0)
        state_dict["model"]["extra"] = ShardedTensor.from_rank_offsets(
            "model.extra", torch.ones((2,), dtype=torch.float32), *_rank_offsets(), replica_id=0
        )
        state_dict["args"] = SimpleNamespace(iteration=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = 2
        dist_checkpointing.save(state_dict, str(ckpt_b))

        with pytest.raises(Exception, match="model.extra|Missing|missing|Unexpected|unexpected"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                strict=StrictHandling.RAISE_ALL,
            )


def test_file_backed_streaming_rejects_extra_checkpoint_keys_with_raise_all(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_extra_strict_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_extra_strict_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_extra_strict_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        state_dict = _template(2.0)
        state_dict["model"]["extra"] = ShardedTensor.from_rank_offsets(
            "model.extra", torch.ones((2,), dtype=torch.float32), *_rank_offsets(), replica_id=0
        )
        state_dict["args"] = SimpleNamespace(iteration=2)
        state_dict["checkpoint_version"] = 3.0
        state_dict["iteration"] = 2
        dist_checkpointing.save(state_dict, str(ckpt_b))

        with pytest.raises(Exception, match="model.extra|missing|strict validation"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                execution_mode="file-backed-streaming",
                strict=StrictHandling.RAISE_ALL,
            )

        assert not (output_root / "merged").exists()


def test_file_backed_streaming_raise_all_accepts_matching_checkpoint_keys(
    tmp_path_dist_ckpt, process_group
):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_raise_all_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_raise_all_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_stream_raise_all_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _template(),
            execution_mode="file-backed-streaming",
            strict=StrictHandling.RAISE_ALL,
            verify_load=True,
        )
        loaded = _load_checkpoint(result.output_dir)

        assert result.implementation_mode == "file-backed-streaming"
        assert result.verified_load
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 1.5))


def test_merge_rejects_missing_metadata(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_missing_meta_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)
        (ckpt_b / "metadata.json").unlink()

        with pytest.raises(WeightedMergeError, match="not a distributed checkpoint|metadata"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged", lambda: _template()
            )


def test_merge_rejects_checkpoint_format_mismatch(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_format_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)
        (ckpt_b / "metadata.json").write_text(
            '{"sharded_backend": "different_backend", "sharded_backend_version": 1, '
            '"common_backend": "torch", "common_backend_version": 1}',
            encoding="utf-8",
        )

        with pytest.raises(WeightedMergeError, match="Checkpoint format mismatch"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root / "merged", lambda: _template()
            )


def test_merge_rejects_unsupported_checkpoint_format(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unsupported_format") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_unsupported_format_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        (ckpt_a / "metadata.json").write_text(
            '{"sharded_backend": "fsdp_dtensor", "sharded_backend_version": 1, '
            '"common_backend": "torch", "common_backend_version": 1}',
            encoding="utf-8",
        )

        with pytest.raises(WeightedMergeError, match="Unsupported checkpoint format"):
            merge_sharded_checkpoints(
                [ckpt_a], [1.0], output_root / "merged", lambda: _template()
            )


def test_merge_rejects_argument_validation_errors(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_arg_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_arg_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)

        with pytest.raises(WeightedMergeError, match="At least one input"):
            merge_sharded_checkpoints([], [], output_root, lambda: _template())
        with pytest.raises(WeightedMergeError, match="input paths but"):
            merge_sharded_checkpoints([ckpt_a], [0.5, 0.5], output_root, lambda: _template())
        with pytest.raises(WeightedMergeError, match="Unsupported save dtype"):
            merge_sharded_checkpoints(
                [ckpt_a], [1.0], output_root, lambda: _template(), save_dtype="fp8"
            )
        with pytest.raises(WeightedMergeError, match="Unsupported execution mode"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root,
                lambda: _template(),
                execution_mode=weighted_merge_module.METADATA_SAME_LAYOUT_MODE,
            )
        with pytest.raises(WeightedMergeError, match="Unsupported file-backed staging layout"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root,
                lambda: _template(),
                file_backed_staging_layout="one-giant-view",
            )
        with pytest.raises(WeightedMergeError, match="max_file_backed_staging_bytes"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root,
                lambda: _template(),
                max_file_backed_staging_bytes=0,
            )
        with pytest.raises(WeightedMergeError, match="max_file_backed_staging_files"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root,
                lambda: _template(),
                max_file_backed_staging_files=0,
            )
        with pytest.raises(WeightedMergeError, match="preflight_only"):
            merge_sharded_checkpoints(
                [ckpt_a],
                [1.0],
                output_root,
                lambda: _template(),
                preflight_only=True,
                verify_load=True,
            )


def test_merge_rejects_return_style_strict_modes(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_strict_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_strict_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_strict_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, iteration=1)
        _write_checkpoint(ckpt_b, 2.0, iteration=2)

        with pytest.raises(WeightedMergeError, match="return type"):
            merge_sharded_checkpoints(
                [ckpt_a, ckpt_b],
                [0.5, 0.5],
                output_root / "merged",
                lambda: _template(),
                strict=StrictHandling.RETURN_ALL,
            )


def test_object_extra_state_is_copied(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_obj_out") as output_root,
    ):
        _write_object_extra_checkpoint(ckpt_a, 1.0, extra_value=111, iteration=1)
        _write_object_extra_checkpoint(ckpt_b, 3.0, extra_value=999, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _object_extra_template(),
        )
        loaded = dist_checkpointing.load(_object_extra_template(), str(result.output_dir))

        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 2.0))
        loaded["model"]["decoder.layers.0._extra_state"].seek(0)
        assert torch.load(loaded["model"]["decoder.layers.0._extra_state"]) == {"value": 111}


def test_extra_state_source_index_can_be_selected(tmp_path_dist_ckpt, process_group):
    with (
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_a") as ckpt_a,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_b") as ckpt_b,
        TempNamedDir(tmp_path_dist_ckpt / "weighted_merge_extra_source_out") as output_root,
    ):
        _write_checkpoint(ckpt_a, 1.0, extra_value=111.0, iteration=1)
        _write_checkpoint(ckpt_b, 3.0, extra_value=999.0, iteration=2)

        result = merge_sharded_checkpoints(
            [ckpt_a, ckpt_b],
            [0.5, 0.5],
            output_root / "merged",
            lambda: _template(),
            extra_state_source_index=1,
        )
        loaded = _load_checkpoint(result.output_dir)

        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))
