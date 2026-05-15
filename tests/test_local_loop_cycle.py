from __future__ import annotations

from pathlib import Path

from azchess.tools.local_loop_cycle import (
    _eval_selection_summary,
    _prune_generated_artifacts,
    _replace_placeholders,
    evaluate_cycle_promotion,
)


def test_cycle_promotion_rejects_policy_drift():
    report = {
        "eval": {
            "delta": {
                "value_mse": -0.01,
                "policy_ce": 0.002,
                "policy_legal_ce": 0.0,
            }
        },
        "fresh_data": {"game_outcomes": {"capped": 0, "terminal": 0}},
    }

    gate = evaluate_cycle_promotion(
        report,
        max_value_mse_delta=0.0,
        max_policy_ce_delta=0.001,
        max_policy_legal_ce_delta=1.0e-5,
        max_fresh_capped_fraction=1.0,
    )

    assert gate["promote"] is False
    assert any(check["name"] == "heldout_policy_ce" and not check["passed"] for check in gate["checks"])


def test_cycle_promotion_promotes_value_improvement_with_policy_caps():
    report = {
        "eval": {
            "delta": {
                "value_mse": -0.001,
                "policy_ce": 0.0005,
                "policy_legal_ce": 0.000002,
            }
        },
        "fresh_data": {"game_outcomes": {"capped": 1, "terminal": 9}},
    }

    gate = evaluate_cycle_promotion(
        report,
        max_value_mse_delta=0.0,
        max_policy_ce_delta=0.001,
        max_policy_legal_ce_delta=1.0e-5,
        max_fresh_capped_fraction=0.2,
    )

    assert gate["promote"] is True


def test_cycle_promotion_rejects_source_value_regression():
    report = {
        "eval": {
            "delta": {
                "value_mse": -0.001,
                "policy_ce": 0.0005,
                "policy_legal_ce": 0.000002,
            },
            "source_delta": {
                "tablebase": {"value_mse": 0.00001},
                "terminal": {"value_mse": -0.002},
            },
        },
        "fresh_data": {"game_outcomes": {"capped": 0, "terminal": 1}},
    }

    gate = evaluate_cycle_promotion(
        report,
        max_value_mse_delta=0.0,
        max_policy_ce_delta=0.001,
        max_policy_legal_ce_delta=1.0e-5,
        max_fresh_capped_fraction=1.0,
        max_source_value_mse_delta=0.0,
    )

    assert gate["promote"] is False
    assert any(
        check["name"] == "heldout_source_value_mse:tablebase" and not check["passed"]
        for check in gate["checks"]
    )


def test_cycle_promotion_allows_source_value_within_cap():
    report = {
        "eval": {
            "delta": {
                "value_mse": -0.001,
                "policy_ce": 0.0005,
                "policy_legal_ce": 0.000002,
            },
            "source_delta": {
                "tablebase": {"value_mse": 0.0000001},
                "terminal": {"value_mse": -0.002},
            },
        },
        "fresh_data": {"game_outcomes": {"capped": 0, "terminal": 1}},
    }

    gate = evaluate_cycle_promotion(
        report,
        max_value_mse_delta=0.0,
        max_policy_ce_delta=0.001,
        max_policy_legal_ce_delta=1.0e-5,
        max_fresh_capped_fraction=1.0,
        max_source_value_mse_delta=0.000001,
    )

    assert gate["promote"] is True


def test_cycle_promotion_can_use_weighted_value_metric():
    report = {
        "eval": {
            "delta": {
                "value_mse": -0.0000001,
                "value_weighted_mse": -0.000001,
                "policy_ce": 0.0005,
                "policy_legal_ce": 0.000002,
            },
            "source_delta": {
                "capped": {"value_weighted_mse": -0.000001},
                "terminal": {"value_weighted_mse": 0.0000001},
            },
        },
        "fresh_data": {"game_outcomes": {"capped": 0, "terminal": 1}},
    }

    gate = evaluate_cycle_promotion(
        report,
        max_value_mse_delta=-0.0000003,
        max_policy_ce_delta=0.001,
        max_policy_legal_ce_delta=1.0e-5,
        max_fresh_capped_fraction=1.0,
        max_source_value_mse_delta=0.0000005,
        value_gate_metric="value_weighted_mse",
        source_value_gate_metric="value_weighted_mse",
    )

    assert gate["promote"] is True
    assert any(check["name"] == "heldout_value_weighted_mse" for check in gate["checks"])


def test_replace_placeholders():
    values = ["--policy-distill-checkpoint", "{parent}", "--run-name", "cycle_{cycle}", "{cycle_run_dir}/x"]

    replaced = _replace_placeholders(
        values,
        parent=Path("checkpoints/current.pt"),
        cycle_run_dir=Path("logs/cycles/cycle_0003"),
        cycle=3,
    )

    assert replaced == [
        "--policy-distill-checkpoint",
        "checkpoints/current.pt",
        "--run-name",
        "cycle_0003",
        "logs/cycles/cycle_0003/x",
    ]


def test_eval_selection_summary_includes_candidate_failures():
    report = {
        "eval": {
            "selection": {
                "enabled": True,
                "selected_chunk": None,
                "best_metric_value": None,
                "candidates": [
                    {
                        "chunk": 1,
                        "metric_value": -1.0e-6,
                        "passes_policy_limits": False,
                        "source_delta": {
                            "capped": {"value_weighted_mse": 6.8e-6},
                            "terminal": {"value_weighted_mse": -1.7e-6},
                        },
                        "selection_failures": [
                            {
                                "name": "source:terminal:value_weighted_mse",
                                "value": 1.1e-5,
                                "max": 2.0e-6,
                            }
                        ],
                    }
                ],
            }
        }
    }

    summary = _eval_selection_summary(report)

    assert summary["enabled"] is True
    assert summary["selected_chunk"] is None
    assert summary["candidates"][0]["chunk"] == 1
    assert summary["candidates"][0]["source_delta"]["capped"]["value_weighted_mse"] == 6.8e-6
    assert summary["candidates"][0]["selection_failures"][0]["name"] == "source:terminal:value_weighted_mse"


def test_prune_generated_artifacts_removes_checkpoints_and_events(tmp_path):
    keep = tmp_path / "local_loop_report.json"
    checkpoint = tmp_path / "checkpoints" / "local_loop_selected.pt"
    event = tmp_path / "logs" / "events.out.tfevents.fake"
    keep.write_text("{}")
    checkpoint.parent.mkdir()
    checkpoint.write_text("checkpoint")
    event.parent.mkdir()
    event.write_text("event")

    result = _prune_generated_artifacts(tmp_path)

    assert result["count"] == 2
    assert keep.exists()
    assert not checkpoint.exists()
    assert not event.exists()
