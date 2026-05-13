from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch

from azchess.config import Config
from azchess.data_manager import DataManager
from azchess.model import PolicyValueNet
from argparse import Namespace

from azchess.tools.bench_local_loop import (
    _eval_delta,
    _fresh_quality_gate,
    _full_npz_eval_batches,
    _normalize_checkpoint_state,
    _prepare_initial_checkpoint,
    _sample_eval_batch,
    _sample_eval_batches,
    _selection_passes,
    _source_configuration_diagnostics,
    _source_pressure_diagnostics,
    _summarize_metric_records,
    copy_anchor_shards,
    evaluate_checkpoint_batches,
    evaluate_checkpoint,
    limit_fresh_selfplay_shards,
    summarize_npz_shards,
    write_loop_config,
)


def _tiny_model_cfg() -> dict:
    return {
        "planes": 19,
        "channels": 32,
        "blocks": 2,
        "policy_size": 4672,
        "self_supervised": False,
    }


def test_summarize_npz_shards_reports_policy_legal_and_ssl(tmp_path):
    dm = DataManager(base_dir=str(tmp_path))
    states = np.zeros((4, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((4, 4672), dtype=np.float32)
    legal = np.zeros((4, 4672), dtype=np.uint8)
    pi[:, 0] = 0.7
    pi[:, 1] = 0.3
    legal[:, :2] = 1
    data = {
        "s": states,
        "pi": pi,
        "z": np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32),
        "legal_mask": legal,
        "ssl_piece": np.ones((4, 8, 8), dtype=np.float32),
        "value_weight": np.zeros((4,), dtype=np.float32),
        "meta_capped": np.array([1], dtype=np.int8),
        "meta_resigned": np.array([0], dtype=np.int8),
        "meta_terminal": np.array([0], dtype=np.int8),
        "meta_result_source": np.array(["capped"]),
        "meta_value_bootstrap": np.array([1], dtype=np.int8),
        "meta_value_weight": np.array([0.25], dtype=np.float32),
        "meta_final_piece_count": np.array([18], dtype=np.int16),
        "meta_final_halfmove_clock": np.array([72], dtype=np.int16),
        "meta_final_legal_count": np.array([31], dtype=np.int16),
        "meta_final_can_claim_draw": np.array([1], dtype=np.int8),
    }
    dm.add_selfplay_data(data, worker_id=0, game_id=0)

    metrics = summarize_npz_shards(tmp_path)

    assert metrics["shards"] == 1
    assert metrics["samples"] == 4
    assert metrics["policy_row_sum"]["mean"] == 1.0
    assert metrics["legal_policy_mass"]["mean"] == 1.0
    assert metrics["ssl_ranges"]["ssl_piece"] == {"min": 1.0, "max": 1.0}
    assert metrics["value_weight"]["mean"] == 0.0
    assert metrics["game_value_weight"]["mean"] == 0.25
    assert metrics["game_outcomes"]["capped"] == 1
    assert metrics["game_outcomes"]["value_bootstrap"] == 1
    assert metrics["game_outcomes"]["result_sources"] == {"capped": 1}
    assert metrics["final_position"]["piece_count"]["mean"] == 18.0
    assert metrics["final_position"]["halfmove_clock"]["mean"] == 72.0
    assert metrics["final_position"]["legal_count"]["mean"] == 31.0
    assert metrics["final_position"]["can_claim_draw"] == 1
    assert metrics["source_metrics"]["capped"]["shards"] == 1
    assert metrics["source_metrics"]["capped"]["samples"] == 4
    assert metrics["source_metrics"]["capped"]["final_piece_count"]["mean"] == 18.0
    assert np.isclose(metrics["source_metrics"]["capped"]["policy_top_prob"]["mean"], 0.7)


def test_fresh_quality_gate_rejects_high_capped_fraction():
    metrics = {
        "game_outcomes": {
            "capped": 18,
            "tablebase": 6,
            "terminal": 0,
            "adjudicated_draw": 0,
        }
    }

    gate = _fresh_quality_gate(metrics, 0.67)

    assert gate["verdict"] == "reject"
    assert gate["checks"][0]["name"] == "fresh_capped_fraction"
    assert np.isclose(gate["checks"][0]["value"], 0.75)


def test_source_pressure_diagnostics_reports_reuse_factor():
    metrics = {
        "source_metrics": {
            "terminal": {"samples": 100, "shards": 2},
            "tablebase": {"samples": 1000, "shards": 10},
            "capped": {"samples": 500, "shards": 5},
        }
    }

    diag = _source_pressure_diagnostics(
        metrics,
        source_mix_specs=["terminal=0.25", "tablebase=0.50", "capped=0.25"],
        batch_size=128,
        train_steps=60,
    )

    assert diag["enabled"] is True
    assert diag["sources"]["terminal"]["batch_allocation"] == 32
    assert diag["sources"]["terminal"]["expected_draws"] == 1920
    assert np.isclose(diag["sources"]["terminal"]["expected_reuse_factor"], 19.2)


def test_copy_anchor_shards_accepts_direct_npz_directory(tmp_path):
    source = tmp_path / "teacher_scenario"
    source.mkdir()
    np.savez_compressed(
        source / "teacher.npz",
        s=np.zeros((2, 19, 8, 8), dtype=np.float32),
        pi=np.full((2, 4672), 1.0 / 4672.0, dtype=np.float32),
        z=np.zeros((2,), dtype=np.float32),
        value_weight=np.ones((2,), dtype=np.float32),
        meta_result_source=np.array(["teacher:test"]),
    )
    run_data = tmp_path / "run" / "data"

    info = copy_anchor_shards([str(source)], run_data)

    assert info["copied_files"] == 1
    assert len(list((run_data / "replays").glob("*.npz"))) == 1


def test_copy_anchor_shards_can_filter_by_meta_result_source(tmp_path):
    source = tmp_path / "anchor"
    source.mkdir()
    for name, result_source in (("tablebase.npz", "tablebase"), ("capped.npz", "capped")):
        np.savez_compressed(
            source / name,
            s=np.zeros((2, 19, 8, 8), dtype=np.float32),
            pi=np.full((2, 4672), 1.0 / 4672.0, dtype=np.float32),
            z=np.zeros((2,), dtype=np.float32),
            value_weight=np.ones((2,), dtype=np.float32),
            meta_result_source=np.array([result_source]),
        )

    info = copy_anchor_shards([str(source)], tmp_path / "run" / "data", source_prefixes=["tablebase"])

    copied = list((tmp_path / "run" / "data" / "replays").glob("*.npz"))
    assert info["copied_files"] == 1
    assert copied[0].name.endswith("tablebase.npz")


def test_evaluate_checkpoint_reports_batch_metrics(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    dm = DataManager(base_dir=str(tmp_path))
    states = np.zeros((3, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((3, 4672), dtype=np.float32)
    legal = np.zeros((3, 4672), dtype=np.uint8)
    pi[:, 0] = 1.0
    legal[:, :2] = 1
    dm.add_training_data(
        {
            "s": states,
            "pi": pi,
            "z": np.zeros((3,), dtype=np.float32),
            "legal_mask": legal,
        },
        shard_id=0,
        source="selfplay",
    )
    model = PolicyValueNet.from_config(cfg.model())
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    metrics = evaluate_checkpoint(ckpt, cfg, tmp_path, "cpu", batch_size=2)

    assert metrics["batch_size"] == 2
    assert metrics["positions_per_second"] > 0
    assert metrics["policy_ce"] > 0
    assert "legal_policy_mass" in metrics
    assert "policy_legal_ce" in metrics
    assert "policy_legal_kl" in metrics


def test_evaluate_checkpoint_batches_reports_mean_and_std(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((2, 4672), dtype=np.float32)
    legal = np.zeros((2, 4672), dtype=np.uint8)
    pi[:, 0] = 1.0
    legal[:, 0] = 1
    batch = {"s": states, "pi": pi, "z": np.zeros((2,), dtype=np.float32), "legal_mask": legal}
    model = PolicyValueNet.from_config(cfg.model())
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=2,
        batches=2,
        fixed_batches=[batch, batch],
    )

    assert metrics["batches"] == 2
    assert "policy_ce" in metrics
    assert "policy_ce_std" in metrics
    assert "policy_legal_ce" in metrics


def test_evaluate_checkpoint_batches_uses_requested_checkpoint_state(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    batch = {
        "s": np.zeros((2, 19, 8, 8), dtype=np.float32),
        "pi": np.eye(4672, dtype=np.float32)[:2],
        "z": np.zeros((2,), dtype=np.float32),
    }
    model_a = PolicyValueNet.from_config(cfg.model())
    model_b = PolicyValueNet.from_config(cfg.model())
    with torch.no_grad():
        for param in model_a.parameters():
            param.zero_()
        for param in model_b.parameters():
            param.fill_(0.01)
    ckpt = tmp_path / "dual.pt"
    torch.save({"model_ema": model_a.state_dict(), "model": model_b.state_dict()}, ckpt)

    ema_metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=2,
        batches=1,
        fixed_batches=[batch],
        checkpoint_state="model_ema",
    )
    raw_metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=2,
        batches=1,
        fixed_batches=[batch],
        checkpoint_state="model",
    )

    assert raw_metrics["policy_ce"] != ema_metrics["policy_ce"]


def test_normalize_checkpoint_state_rewrites_legacy_loader_entries(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    model_a = PolicyValueNet.from_config(cfg.model())
    model_b = PolicyValueNet.from_config(cfg.model())
    with torch.no_grad():
        for param in model_a.parameters():
            param.zero_()
        for param in model_b.parameters():
            param.fill_(0.02)
    ckpt = tmp_path / "dual.pt"
    torch.save({"model_ema": model_a.state_dict(), "model": model_b.state_dict()}, ckpt)

    info = _normalize_checkpoint_state(ckpt, "model")
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)

    key = next(iter(model_b.state_dict()))
    assert info["source_key"] == "model"
    assert torch.equal(saved["model_ema"][key], model_b.state_dict()[key])
    assert torch.equal(saved["model_state_dict"][key], model_b.state_dict()[key])


def test_summarize_metric_records_weights_metric_means_by_batch_size():
    metrics = _summarize_metric_records(
        [
            {"batch_size": 1, "seconds": 1.0, "positions_per_second": 1.0, "value_mse": 10.0},
            {"batch_size": 3, "seconds": 1.0, "positions_per_second": 3.0, "value_mse": 2.0},
        ]
    )

    assert metrics["batches"] == 2
    assert np.isclose(metrics["batch_size"], 2.0)
    assert np.isclose(metrics["value_mse"], 4.0)
    assert np.isclose(metrics["positions_per_second"], 2.0)


def test_summarize_metric_records_weights_value_weighted_mse_by_value_weight_sum():
    metrics = _summarize_metric_records(
        [
            {"batch_size": 10, "value_weight_sum": 2.5, "value_weighted_mse": 8.0},
            {"batch_size": 10, "value_weight_sum": 10.0, "value_weighted_mse": 2.0},
        ]
    )

    assert np.isclose(metrics["value_weight_sum"], 12.5)
    assert np.isclose(metrics["value_weighted_mse"], 3.2)


def test_evaluate_checkpoint_batches_reports_source_metrics(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    states = np.zeros((4, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((4, 4672), dtype=np.float32)
    legal = np.zeros((4, 4672), dtype=np.uint8)
    pi[:, 0] = 1.0
    legal[:, 0] = 1
    batch = {
        "s": states,
        "pi": pi,
        "z": np.zeros((4,), dtype=np.float32),
        "legal_mask": legal,
        "result_source": np.asarray(["capped", "capped", "tablebase", "tablebase"]),
    }
    model = PolicyValueNet.from_config(cfg.model())
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=4,
        batches=1,
        fixed_batches=[batch],
    )

    assert metrics["source_metrics"]["capped"]["samples"] == 2
    assert metrics["source_metrics"]["tablebase"]["samples"] == 2
    assert "value_mse" in metrics["source_metrics"]["capped"]


def test_evaluate_checkpoint_batches_reports_value_weighted_mse(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((2, 4672), dtype=np.float32)
    pi[:, 0] = 1.0
    batch = {
        "s": states,
        "pi": pi,
        "z": np.array([1.0, 3.0], dtype=np.float32),
        "value_weight": np.array([1.0, 3.0], dtype=np.float32),
        "result_source": np.asarray(["capped", "terminal"]),
    }
    model = PolicyValueNet.from_config(cfg.model())
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=2,
        batches=1,
        fixed_batches=[batch],
    )

    before = metrics["value_mse"]
    weighted = metrics["value_weighted_mse"]
    assert "value_weight_mean" in metrics
    assert "value_weight_sum" in metrics
    assert weighted != before
    assert "value_weighted_mse" in metrics["source_metrics"]["capped"]


def test_evaluate_checkpoint_batches_omits_weighted_mse_when_all_value_weights_zero(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((2, 4672), dtype=np.float32)
    pi[:, 0] = 1.0
    batch = {
        "s": states,
        "pi": pi,
        "z": np.array([1.0, 3.0], dtype=np.float32),
        "value_weight": np.zeros((2,), dtype=np.float32),
        "result_source": np.asarray(["capped", "capped"]),
    }
    model = PolicyValueNet.from_config(cfg.model())
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    metrics = evaluate_checkpoint_batches(
        ckpt,
        cfg,
        tmp_path,
        "cpu",
        batch_size=2,
        batches=1,
        fixed_batches=[batch],
    )

    assert metrics["value_weight_sum"] == 0.0
    assert "value_weighted_mse" not in metrics
    assert "value_weighted_mse" not in metrics["source_metrics"]["capped"]


def test_source_configuration_diagnostics_warns_when_eval_source_not_trained():
    metrics = {
        "source_metrics": {
            "tablebase": {"samples": 10, "shards": 1},
            "capped": {"samples": 10, "shards": 1},
        }
    }
    args = Namespace(
        train_anchor_source_prefix=["tablebase", "capped"],
        eval_result_source=["tablebase", "draw_adjudication"],
        value_include_source=["tablebase", "capped"],
        train_result_source_mix=["tablebase=0.8", "capped=0.2"],
    )

    diag = _source_configuration_diagnostics(metrics, args)

    assert diag["missing_eval_sources_from_train"] == ["draw_adjudication"]
    assert diag["warnings"]


def test_copy_anchor_shards_imports_prior_data_into_replays(tmp_path):
    anchor = tmp_path / "anchor"
    dm = DataManager(base_dir=str(anchor))
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((2, 4672), dtype=np.float32)
    legal = np.zeros((2, 4672), dtype=np.uint8)
    pi[:, 0] = 1.0
    legal[:, 0] = 1
    dm.add_selfplay_data(
        {
            "s": states,
            "pi": pi,
            "z": np.zeros((2,), dtype=np.float32),
            "legal_mask": legal,
            "value_weight": np.ones((2,), dtype=np.float32),
        },
        worker_id=0,
        game_id=0,
    )

    info = copy_anchor_shards([str(anchor)], tmp_path / "run" / "data")
    metrics = summarize_npz_shards(tmp_path / "run" / "data")

    assert info["copied_files"] == 1
    assert metrics["shards"] == 1
    assert metrics["samples"] == 2


def test_limit_fresh_selfplay_shards_moves_excess_out_of_training_path(tmp_path):
    run_data = tmp_path / "run" / "data"
    dm = DataManager(base_dir=str(run_data))
    states = np.zeros((2, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((2, 4672), dtype=np.float32)
    legal = np.zeros((2, 4672), dtype=np.uint8)
    pi[:, 0] = 1.0
    legal[:, 0] = 1
    for game_id in range(5):
        dm.add_selfplay_data(
            {
                "s": states,
                "pi": pi,
                "z": np.zeros((2,), dtype=np.float32),
                "legal_mask": legal,
                "value_weight": np.zeros((2,), dtype=np.float32),
            },
            worker_id=0,
            game_id=game_id,
        )

    before = summarize_npz_shards(run_data)
    info = limit_fresh_selfplay_shards(run_data, max_files=2, seed=7)
    after = summarize_npz_shards(run_data)

    assert before["shards"] == 5
    assert info["kept_files"] == 2
    assert info["moved_files"] == 3
    assert info["metadata_rows_removed"] == 3
    assert after["shards"] == 2
    assert len(list((run_data / "excluded_selfplay").glob("*.npz"))) == 3

    with sqlite3.connect(run_data / "data_metadata.db") as conn:
        rows = conn.execute("SELECT path FROM shards").fetchall()
    db_paths = [path for (path,) in rows]
    assert len(db_paths) == 2
    assert all((run_data / "selfplay" / Path(path).name).exists() for path in db_paths)


def test_sample_eval_batch_can_filter_by_source_prefix(tmp_path):
    dm = DataManager(base_dir=str(tmp_path))
    states = np.zeros((3, 19, 8, 8), dtype=np.float32)
    pi = np.zeros((3, 4672), dtype=np.float32)
    pi[:, 0] = 1.0
    dm.add_training_data(
        {
            "s": states,
            "pi": pi,
            "z": np.ones((3,), dtype=np.float32),
        },
        shard_id=0,
        source="holdout:stable",
    )

    batch = _sample_eval_batch(tmp_path, batch_size=2, source_prefixes=["holdout:"])

    assert batch["s"].shape[0] == 2
    assert np.all(batch["z"] == 1.0)


def test_sample_eval_batches_seed_is_stable_and_restores_rng(tmp_path):
    dm = DataManager(base_dir=str(tmp_path))
    for shard_id in range(3):
        states = np.full((4, 19, 8, 8), float(shard_id), dtype=np.float32)
        pi = np.zeros((4, 4672), dtype=np.float32)
        pi[:, shard_id] = 1.0
        dm.add_training_data(
            {
                "s": states,
                "pi": pi,
                "z": np.full((4,), float(shard_id), dtype=np.float32),
            },
            shard_id=shard_id,
            source="stockfish:holdout",
        )
    np.random.seed(99)
    before = np.random.random()

    first = _sample_eval_batches(tmp_path, batch_size=2, batches=2, seed=123)
    after_seeded = np.random.random()
    np.random.seed(99)
    assert before == np.random.random()
    assert after_seeded == np.random.random()

    second = _sample_eval_batches(tmp_path, batch_size=2, batches=2, seed=123)

    assert [batch["z"].tolist() for batch in first] == [batch["z"].tolist() for batch in second]


def test_sample_eval_batches_can_filter_by_meta_result_source(tmp_path):
    for name, result_source, value in (("tablebase.npz", "tablebase", 1.0), ("capped.npz", "capped", -1.0)):
        np.savez_compressed(
            tmp_path / name,
            s=np.full((4, 19, 8, 8), value, dtype=np.float32),
            pi=np.full((4, 4672), 1.0 / 4672.0, dtype=np.float32),
            z=np.full((4,), value, dtype=np.float32),
            legal_mask=np.ones((4, 4672), dtype=np.uint8),
            meta_result_source=np.array([result_source]),
        )

    batches = _sample_eval_batches(
        tmp_path,
        batch_size=3,
        batches=2,
        result_source_prefixes=["tablebase"],
        seed=321,
    )

    assert all(np.all(batch["z"] == 1.0) for batch in batches)
    assert all(set(batch["result_source"].astype(str)) == {"tablebase"} for batch in batches)


def test_full_npz_eval_batches_iterates_all_matching_samples(tmp_path):
    for name, result_source, value in (("tablebase.npz", "tablebase", 1.0), ("terminal.npz", "terminal", -1.0)):
        np.savez_compressed(
            tmp_path / name,
            s=np.full((5, 19, 8, 8), value, dtype=np.float32),
            pi=np.full((5, 4672), 1.0 / 4672.0, dtype=np.float32),
            z=np.full((5,), value, dtype=np.float32),
            legal_mask=np.ones((5, 4672), dtype=np.uint8),
            meta_result_source=np.array([result_source]),
        )

    batches = _full_npz_eval_batches(tmp_path, batch_size=2, result_source_prefixes=["tablebase"])

    assert [batch["z"].shape[0] for batch in batches] == [2, 2, 1]
    assert sum(batch["z"].shape[0] for batch in batches) == 5
    assert all(np.all(batch["z"] == 1.0) for batch in batches)


def test_eval_delta_and_selection_policy_limits():
    before = {
        "policy_ce": 1.0,
        "policy_legal_ce": 0.5,
        "value_mse": 0.25,
        "legal_policy_mass": 0.99,
    }
    after = {
        "policy_ce": 1.0005,
        "policy_legal_ce": 0.500002,
        "value_mse": 0.24,
        "legal_policy_mass": 0.991,
    }
    args = Namespace(
        eval_select_max_policy_ce_delta=0.001,
        eval_select_max_policy_legal_ce_delta=1.0e-5,
        eval_select_max_source_value_mse_delta=None,
    )

    delta = _eval_delta(before, after)

    assert np.isclose(delta["value_mse"], -0.01)
    assert _selection_passes(delta, args)

    delta["policy_ce"] = 0.002
    assert not _selection_passes(delta, args)


def test_selection_can_reject_source_value_regression():
    delta = {
        "policy_ce": 0.0005,
        "policy_legal_ce": 0.000002,
        "value_mse": -0.01,
    }
    source_delta = {
        "capped": {"value_mse": -0.02},
        "terminal": {"value_mse": 0.00001},
    }
    args = Namespace(
        eval_select_max_policy_ce_delta=0.001,
        eval_select_max_policy_legal_ce_delta=1.0e-5,
        eval_select_max_source_value_mse_delta=0.0000005,
    )

    assert not _selection_passes(delta, args, source_delta)

    source_delta["terminal"]["value_mse"] = 0.0000001
    assert _selection_passes(delta, args, source_delta)


def test_selection_can_use_custom_source_metric():
    delta = {
        "policy_ce": 0.0005,
        "policy_legal_ce": 0.000002,
        "value_weighted_mse": -0.01,
    }
    source_delta = {
        "capped": {"value_weighted_mse": -0.02},
        "terminal": {"value_weighted_mse": 0.0000001},
    }
    args = Namespace(
        eval_select_max_policy_ce_delta=0.001,
        eval_select_max_policy_legal_ce_delta=1.0e-5,
        eval_select_max_source_value_mse_delta=0.0000005,
        eval_select_source_metric="value_weighted_mse",
    )

    assert _selection_passes(delta, args, source_delta)

    source_delta["terminal"]["value_weighted_mse"] = 0.00001
    assert not _selection_passes(delta, args, source_delta)


def test_prepare_initial_checkpoint_copies_and_validates_provided_checkpoint(tmp_path):
    cfg = Config({"model": _tiny_model_cfg()})
    model = PolicyValueNet.from_config(cfg.model())
    source = tmp_path / "source.pt"
    target = tmp_path / "run" / "checkpoints" / "local_loop_init.pt"
    torch.save({"model": model.state_dict()}, source)

    info = _prepare_initial_checkpoint(cfg, target, "cpu", str(source))

    assert info["mode"] == "provided"
    assert info["source"] == str(source)
    assert target.exists()
    copied = torch.load(target, map_location="cpu", weights_only=False)
    assert "model" in copied


def test_write_loop_config_applies_draw_overrides(tmp_path):
    cfg = Config(
        {
            "model": _tiny_model_cfg(),
            "selfplay": {"draw": {"enabled": True, "halfmove_cap": 100}},
        }
    )
    args = Namespace(
        workers=2,
        games=4,
        max_game_len=120,
        sims=8,
        capped_value_weight=0.25,
        min_resign_plies=90,
        resign_threshold=-0.95,
        resign_consecutive_bad=3,
        resign_window=6,
        resign_value_margin=0.08,
        resign_min_entropy=0.2,
        inference_batch_size=4,
        batch_size=16,
        eval_batches=4,
        lr=1e-4,
        warmup_steps=10,
        train_steps=20,
        dataloader_workers=0,
        train_result_source_mix=["terminal=0.25", "tablebase=0.50", "capped=0.25"],
        legal_mass_weight=0.05,
        legal_policy_weight=0.25,
        value_source_weight=["terminal=2.0", "capped=0.5"],
        ssl_weight=0.0,
        policy_label_smoothing=0.0,
        policy_target_temperature=0.5,
        seed=20260512,
        draw_halfmove_cap=80,
        draw_material_threshold=14,
        draw_min_plies=40,
        draw_window=10,
        draw_min_unique=5,
        draw_claim_min_plies=120,
        draw_disable_repetition_claims=True,
        draw_disable_fifty_move_claims=True,
    )

    path = write_loop_config(cfg, tmp_path, args)
    written = Config.load(str(path)).to_dict()

    draw = written["selfplay"]["draw"]
    assert written["seed"] == 20260512
    assert written["selfplay"]["min_resign_plies"] == 90
    assert written["selfplay"]["resign_threshold"] == -0.95
    assert written["selfplay"]["resign_consecutive_bad"] == 3
    assert written["selfplay"]["resign_window"] == 6
    assert written["selfplay"]["resign_value_margin"] == 0.08
    assert written["selfplay"]["resign_min_entropy"] == 0.2
    assert written["training"]["ssl_weight"] == 0.0
    assert written["training"]["policy_label_smoothing"] == 0.0
    assert written["training"]["legal_policy_weight"] == 0.25
    assert written["training"]["value_source_weights"] == ["terminal=2.0", "capped=0.5"]
    assert written["training"]["train_result_source_mix"] == ["terminal=0.25", "tablebase=0.50", "capped=0.25"]
    assert written["selfplay"]["policy_target_temperature"] == 0.5
    assert draw["enabled"] is True
    assert draw["halfmove_cap"] == 80
    assert draw["material_draw_threshold"] == 14
    assert draw["min_plies"] == 40
    assert draw["window"] == 10
    assert draw["min_unique"] == 5
    assert draw["claim_min_plies"] == 120
    assert draw["claim_repetition"] is False
    assert draw["claim_fifty_moves"] is False
