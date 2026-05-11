from __future__ import annotations

from azchess.tools.promotion_gate import evaluate_gate


def test_promotion_gate_rejects_without_match():
    report = evaluate_gate(
        eval_report={"delta_a_minus_b": {"policy_legal_ce": -0.1, "policy_legal_kl": -0.1, "value_mse": -0.1}},
        generator_report={"fresh_data": {"game_outcomes": {"capped": 1, "terminal": 9}}},
        match_report=None,
        max_policy_legal_ce_delta=0.0,
        max_policy_legal_kl_delta=0.0,
        max_value_mse_delta=0.0,
        max_capped_fraction=0.9,
        min_match_score=0.55,
        require_match=True,
    )

    assert report["promote"] is False
    assert report["verdict"] == "reject"
    assert any(check["name"] == "match_score" and not check["passed"] for check in report["checks"])


def test_promotion_gate_promotes_when_all_gates_pass():
    report = evaluate_gate(
        eval_report={"delta_a_minus_b": {"policy_legal_ce": -0.01, "policy_legal_kl": -0.01, "value_mse": 0.0}},
        generator_report={"fresh_data": {"game_outcomes": {"capped": 2, "terminal": 8}}},
        match_report={"wins": 6, "losses": 3, "draws": 1},
        max_policy_legal_ce_delta=0.0,
        max_policy_legal_kl_delta=0.0,
        max_value_mse_delta=0.0,
        max_capped_fraction=0.9,
        min_match_score=0.55,
        require_match=True,
    )

    assert report["promote"] is True
    assert report["verdict"] == "promote"
