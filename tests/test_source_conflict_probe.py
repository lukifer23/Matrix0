from __future__ import annotations

from argparse import Namespace

from azchess.tools.source_conflict_probe import (
    available_probe_specs,
    build_probe_command,
    classify_conflict,
    default_probe_specs,
)


def test_build_probe_command_uses_games_zero_and_source_mix(tmp_path):
    args = Namespace(
        config="config.yaml",
        seed=123,
        sims=50,
        max_game_len=240,
        train_steps=20,
        eval_select_interval=10,
        eval_select_metric="value_weighted_mse",
        eval_select_source_metric="value_weighted_mse",
        eval_select_max_policy_ce_delta=0.001,
        eval_select_max_policy_legal_ce_delta=1e-5,
        eval_select_max_source_value_mse_delta=2e-6,
        batch_size=128,
        eval_batch_size=512,
        eval_batches=1,
        anchor_data_dir=tmp_path / "anchor",
        mps_target_batch=4,
        capped_value_weight=0.25,
        lr=1.5e-8,
        warmup_steps=10,
        parent_checkpoint=tmp_path / "parent.pt",
        eval_full_dataset=True,
        trainable_scope="all",
        policy_distill_weight=1.0,
        value_distill_weight=0.0,
        value_mean_distill_weight=0.0,
    )
    spec = default_probe_specs()[0]

    cmd = build_probe_command(args, spec, tmp_path / "run")

    assert "--games" in cmd
    assert cmd[cmd.index("--games") + 1] == "0"
    assert "--eval-full-dataset" in cmd
    assert cmd[cmd.index("--eval-select-interval") + 1] == "10"
    assert cmd[cmd.index("--eval-select-max-source-value-mse-delta") + 1] == "2e-06"
    assert cmd.count("--train-result-source-mix") == 2
    assert "tablebase=0.70" in cmd
    assert "capped=0.30" in cmd
    assert "--value-include-source" in cmd
    assert "terminal" not in [
        cmd[idx + 1]
        for idx, item in enumerate(cmd[:-1])
        if item == "--value-include-source"
    ]


def test_build_probe_command_can_override_trainable_scope(tmp_path):
    args = Namespace(
        config="config.yaml",
        seed=123,
        sims=50,
        max_game_len=240,
        train_steps=20,
        eval_select_interval=0,
        eval_select_metric="value_weighted_mse",
        eval_select_source_metric="value_weighted_mse",
        eval_select_max_policy_ce_delta=0.001,
        eval_select_max_policy_legal_ce_delta=1e-5,
        eval_select_max_source_value_mse_delta=None,
        batch_size=128,
        eval_batch_size=512,
        eval_batches=1,
        anchor_data_dir=tmp_path / "anchor",
        mps_target_batch=4,
        capped_value_weight=0.25,
        lr=1.5e-8,
        warmup_steps=10,
        parent_checkpoint=tmp_path / "parent.pt",
        eval_full_dataset=True,
        trainable_scope="all",
        policy_distill_weight=1.0,
        value_distill_weight=0.5,
        value_mean_distill_weight=25.0,
    )

    cmd = build_probe_command(args, available_probe_specs()["balanced_value_head"], tmp_path / "run")

    assert cmd[cmd.index("--trainable-scope") + 1] == "value_head"
    assert cmd[cmd.index("--value-distill-weight") + 1] == "0.5"
    assert cmd[cmd.index("--value-mean-distill-weight") + 1] == "25.0"


def test_classify_conflict_detects_opposing_source_pulls():
    probes = {
        "tablebase_capped": {
            "source_delta": {
                "capped": {"value_weighted_mse": -0.1},
                "tablebase": {"value_weighted_mse": -0.2},
                "terminal": {"value_weighted_mse": 0.3},
            }
        },
        "terminal_only": {
            "source_delta": {
                "capped": {"value_weighted_mse": 0.4},
                "tablebase": {"value_weighted_mse": 0.5},
                "terminal": {"value_weighted_mse": -0.6},
            }
        },
        "balanced": {
            "source_delta": {
                "capped": {"value_weighted_mse": 0.7, "value_pred_mean": 0.01},
                "tablebase": {"value_pred_mean": -0.02},
                "terminal": {"value_weighted_mse": -0.8, "value_pred_mean": 0.03},
            }
        },
    }

    classification = classify_conflict(probes, "value_weighted_mse")

    assert classification["source_conflict_detected"] is True
    assert classification["balanced_overprotects_terminal"] is True
    assert classification["global_bias_like_updates"]["balanced"]["same_sign_value_pred_mean_shift"] is False
