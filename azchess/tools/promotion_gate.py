from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open() as f:
        return json.load(f)


def _delta(report: dict[str, Any], key: str) -> float | None:
    if "delta_a_minus_b" in report:
        value = report["delta_a_minus_b"].get(key)
    else:
        value = report.get("eval_delta", report.get("eval", {}).get("delta", {})).get(key)
    return None if value is None else float(value)


def _game_outcomes(report: dict[str, Any]) -> dict[str, Any]:
    data = report.get("fresh_data") or report.get("data_after_selfplay") or report.get("data") or {}
    return data.get("game_outcomes", {})


def _match_score(match_report: dict[str, Any] | None) -> float | None:
    if not match_report:
        return None
    if "score" in match_report:
        return float(match_report["score"])
    wins = float(match_report.get("wins", match_report.get("candidate_wins", 0)))
    losses = float(match_report.get("losses", match_report.get("candidate_losses", 0)))
    draws = float(match_report.get("draws", 0))
    total = wins + losses + draws
    if total <= 0:
        return None
    return (wins + 0.5 * draws) / total


def evaluate_gate(
    *,
    eval_report: dict[str, Any],
    generator_report: dict[str, Any] | None,
    match_report: dict[str, Any] | None,
    max_policy_legal_ce_delta: float,
    max_policy_legal_kl_delta: float,
    max_value_mse_delta: float,
    max_capped_fraction: float,
    min_match_score: float,
    require_match: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: dict[str, Any]) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    plc = _delta(eval_report, "policy_legal_ce")
    add(
        "heldout_policy_legal_ce",
        plc is not None and plc <= max_policy_legal_ce_delta,
        {"value": plc, "max": max_policy_legal_ce_delta},
    )
    plk = _delta(eval_report, "policy_legal_kl")
    add(
        "heldout_policy_legal_kl",
        plk is not None and plk <= max_policy_legal_kl_delta,
        {"value": plk, "max": max_policy_legal_kl_delta},
    )
    vmse = _delta(eval_report, "value_mse")
    add(
        "heldout_value_mse",
        vmse is not None and vmse <= max_value_mse_delta,
        {"value": vmse, "max": max_value_mse_delta},
    )

    if generator_report is not None:
        outcomes = _game_outcomes(generator_report)
        total = sum(int(outcomes.get(k, 0)) for k in ("capped", "terminal", "tablebase", "adjudicated_draw"))
        capped = int(outcomes.get("capped", 0))
        capped_fraction = (capped / total) if total > 0 else None
        add(
            "generator_capped_fraction",
            capped_fraction is not None and capped_fraction <= max_capped_fraction,
            {"value": capped_fraction, "max": max_capped_fraction, "outcomes": outcomes},
        )
    else:
        add("generator_report_present", False, {"value": None})

    score = _match_score(match_report)
    if score is None and not require_match:
        add("match_score", True, {"value": None, "min": min_match_score, "required": False})
    else:
        add(
            "match_score",
            score is not None and score >= min_match_score,
            {"value": score, "min": min_match_score, "required": require_match},
        )

    promote = all(check["passed"] for check in checks)
    return {
        "type": "matrix0_promotion_gate",
        "promote": promote,
        "verdict": "promote" if promote else "reject",
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Matrix0 checkpoint promotion criteria from reports.")
    parser.add_argument("--eval-report", required=True, help="eval_checkpoints JSON or local_loop_report JSON")
    parser.add_argument("--generator-report", default=None, help="local_loop_report JSON for candidate generator quality")
    parser.add_argument("--match-report", default=None, help="Candidate-vs-parent match JSON with wins/losses/draws or score")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-policy-legal-ce-delta", type=float, default=0.0)
    parser.add_argument("--max-policy-legal-kl-delta", type=float, default=0.0)
    parser.add_argument("--max-value-mse-delta", type=float, default=0.0)
    parser.add_argument("--max-capped-fraction", type=float, default=0.9)
    parser.add_argument("--min-match-score", type=float, default=0.55)
    parser.add_argument("--allow-missing-match", action="store_true", help="Diagnostic mode only; promotion normally requires a match report.")
    args = parser.parse_args()

    report = evaluate_gate(
        eval_report=_load_json(args.eval_report) or {},
        generator_report=_load_json(args.generator_report),
        match_report=_load_json(args.match_report),
        max_policy_legal_ce_delta=float(args.max_policy_legal_ce_delta),
        max_policy_legal_kl_delta=float(args.max_policy_legal_kl_delta),
        max_value_mse_delta=float(args.max_value_mse_delta),
        max_capped_fraction=float(args.max_capped_fraction),
        min_match_score=float(args.min_match_score),
        require_match=not bool(args.allow_missing_match),
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
