# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import io
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedObject, ShardedTensor
from megatron.core.dist_checkpointing.validation import StrictHandling
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tools.checkpoint.weighted_merge import (
    WeightedMergeError,
    apply_hybrid_layer_pattern_compat,
    checkpoint_coefficients,
    derive_start_iteration_from_token_window,
    ensure_process_group,
    filter_checkpoints_by_interval,
    get_valid_styles,
    iteration_dir_name,
    merge_sharded_checkpoints,
    normalize_weights,
    output_checkpoint_dir,
    parse_and_validate_merge_args,
    parse_weighted_inputs,
    resolve_checkpoint_dir,
    select_checkpoints_in_window,
    validate_min_checkpoints,
    validate_weights,
)


@pytest.fixture
def process_group():
    already_initialized = dist.is_available() and dist.is_initialized()
    ensure_process_group()
    yield
    if not already_initialized and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank_offsets():
    world_size = _world_size()
    return ((0, _rank(), world_size),) if world_size > 1 else ()


def _template(value=0.0, *, dtype=torch.float32, extra_value=0.0, include_bias=True, shape=(2, 2)):
    rank_offsets = _rank_offsets()
    model_state_dict = {
        "weight": ShardedTensor.from_rank_offsets(
            "model.weight", torch.full(shape, value, dtype=dtype), *rank_offsets, replica_id=0
        ),
        "decoder.layers.0._extra_state": ShardedTensor.from_rank_offsets(
            "model.decoder.layers.0._extra_state",
            torch.tensor([extra_value], dtype=torch.float32),
            *rank_offsets,
            replica_id=0,
        ),
    }
    if include_bias:
        model_state_dict["bias"] = ShardedTensor.from_rank_offsets(
            "model.bias", torch.full((2,), value + 1, dtype=dtype), *rank_offsets, replica_id=0
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


def _load_checkpoint(path, *, dtype=torch.float32):
    return dist_checkpointing.load(_template(dtype=dtype), str(path))


def _bytesio_state(value):
    data = io.BytesIO()
    torch.save({"value": value}, data)
    data.seek(0)
    return data


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
        assert (output_root / "latest_checkpointed_iteration.txt").read_text().strip() == "30"

        loaded = _load_checkpoint(result.output_dir)
        assert torch.allclose(loaded["model"]["weight"], torch.full((2, 2), 4.0))
        assert torch.allclose(loaded["model"]["bias"], torch.full((2,), 5.0))
        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([111.0]))
        assert loaded["checkpoint_version"] == 3.0
        assert loaded["iteration"] == 30
        assert loaded["args"].iteration == 30
        assert loaded["args"].hidden_size == 2


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
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template()
        )
        loaded = _load_checkpoint(result.output_dir)

        assert result.output_dir == output_root
        assert not (output_root / "latest_checkpointed_iteration.txt").exists()
        assert loaded["iteration"] == 10
        assert loaded["args"].iteration == 10
        assert loaded["args"].hidden_size == 2


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
            [ckpt_a, ckpt_b], [0.25, 0.75], output_root, lambda: _multi_chunk_template()
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
            output_root,
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
                output_root,
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
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template(dtype=torch.int64)
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
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template(shape=(2, 2))
            )


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
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template()
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
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template()
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
                output_root,
                lambda: _template(),
                strict=StrictHandling.RAISE_ALL,
            )


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
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template()
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
                [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _template()
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
            merge_sharded_checkpoints([ckpt_a], [1.0], output_root, lambda: _template())


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
                output_root,
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
            [ckpt_a, ckpt_b], [0.5, 0.5], output_root, lambda: _object_extra_template()
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
            output_root,
            lambda: _template(),
            extra_state_source_index=1,
        )
        loaded = _load_checkpoint(result.output_dir)

        assert torch.equal(loaded["model"]["decoder.layers.0._extra_state"], torch.tensor([999.0]))
