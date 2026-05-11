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
    _prepare_initial_checkpoint,
    _sample_eval_batch,
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
        legal_mass_weight=0.05,
        legal_policy_weight=0.25,
        ssl_weight=0.0,
        policy_label_smoothing=0.0,
        policy_target_temperature=0.5,
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
    assert written["selfplay"]["min_resign_plies"] == 90
    assert written["selfplay"]["resign_threshold"] == -0.95
    assert written["selfplay"]["resign_consecutive_bad"] == 3
    assert written["selfplay"]["resign_window"] == 6
    assert written["selfplay"]["resign_value_margin"] == 0.08
    assert written["selfplay"]["resign_min_entropy"] == 0.2
    assert written["training"]["ssl_weight"] == 0.0
    assert written["training"]["policy_label_smoothing"] == 0.0
    assert written["training"]["legal_policy_weight"] == 0.25
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
