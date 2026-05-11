from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _delta(report: dict[str, Any], key: str) -> float | None:
    value = report.get("eval", {}).get("delta", {}).get(key)
    return None if value is None else float(value)


def _source_deltas(report: dict[str, Any], key: str) -> list[tuple[str, float]]:
    source_delta = report.get("eval", {}).get("source_delta", {})
    if not isinstance(source_delta, dict):
        return []
    values: list[tuple[str, float]] = []
    for source, metrics in source_delta.items():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(key)
        if value is not None:
            values.append((str(source), float(value)))
    return values


def _fresh_game_total(report: dict[str, Any]) -> int:
    outcomes = report.get("fresh_data", report.get("data_after_selfplay", {})).get("game_outcomes", {})
    return sum(int(outcomes.get(k, 0)) for k in ("capped", "terminal", "tablebase", "adjudicated_draw"))


def _fresh_capped_fraction(report: dict[str, Any]) -> float | None:
    outcomes = report.get("fresh_data", report.get("data_after_selfplay", {})).get("game_outcomes", {})
    total = _fresh_game_total(report)
    if total <= 0:
        return None
    return float(outcomes.get("capped", 0)) / float(total)


def evaluate_cycle_promotion(
    report: dict[str, Any],
    *,
    max_value_mse_delta: float,
    max_policy_ce_delta: float,
    max_policy_legal_ce_delta: float,
    max_fresh_capped_fraction: float,
    max_source_value_mse_delta: float | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, **detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **detail})

    value_mse = _delta(report, "value_mse")
    add(
        "heldout_value_mse",
        value_mse is not None and value_mse <= max_value_mse_delta,
        value=value_mse,
        max=max_value_mse_delta,
    )

    policy_ce = _delta(report, "policy_ce")
    add(
        "heldout_policy_ce",
        policy_ce is not None and policy_ce <= max_policy_ce_delta,
        value=policy_ce,
        max=max_policy_ce_delta,
    )

    policy_legal_ce = _delta(report, "policy_legal_ce")
    add(
        "heldout_policy_legal_ce",
        policy_legal_ce is not None and policy_legal_ce <= max_policy_legal_ce_delta,
        value=policy_legal_ce,
        max=max_policy_legal_ce_delta,
    )

    capped_fraction = _fresh_capped_fraction(report)
    if capped_fraction is not None:
        add(
            "fresh_capped_fraction",
            capped_fraction <= max_fresh_capped_fraction,
            value=capped_fraction,
            max=max_fresh_capped_fraction,
        )

    if max_source_value_mse_delta is not None:
        for source, source_value_mse in _source_deltas(report, "value_mse"):
            add(
                f"heldout_source_value_mse:{source}",
                source_value_mse <= max_source_value_mse_delta,
                value=source_value_mse,
                max=max_source_value_mse_delta,
            )

    promote = all(check["passed"] for check in checks)
    return {
        "type": "matrix0_cycle_promotion_gate",
        "promote": promote,
        "verdict": "promote" if promote else "reject",
        "checks": checks,
    }


def _replace_placeholders(values: list[str], *, parent: Path, cycle_run_dir: Path, cycle: int) -> list[str]:
    parent_text = str(parent)
    run_text = str(cycle_run_dir)
    cycle_text = f"{cycle:04d}"
    return [
        value.replace("{parent}", parent_text).replace("{cycle_run_dir}", run_text).replace("{cycle}", cycle_text)
        for value in values
    ]


def _archive_existing(path: Path, archive_dir: Path) -> str | None:
    if not path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
    shutil.copy2(path, archived)
    return str(archived)


def run_cycles(args: argparse.Namespace, bench_args: list[str]) -> dict[str, Any]:
    base_run_dir = Path(args.base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = Path(args.best_checkpoint)
    current_parent = Path(args.initial_checkpoint)
    if not current_parent.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {current_parent}")

    if not best_checkpoint.exists() and bool(args.seed_best_checkpoint):
        best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current_parent, best_checkpoint)
        current_parent = best_checkpoint

    cycle_reports: list[dict[str, Any]] = []
    for cycle in range(1, int(args.cycles) + 1):
        cycle_run_dir = base_run_dir / f"cycle_{cycle:04d}"
        cycle_bench_args = _replace_placeholders(bench_args, parent=current_parent, cycle_run_dir=cycle_run_dir, cycle=cycle)
        cmd = [
            sys.executable,
            "-m",
            "azchess.tools.bench_local_loop",
            *cycle_bench_args,
            "--run-dir",
            str(cycle_run_dir),
            "--init-checkpoint",
            str(current_parent),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path.cwd())
        env["MATRIX0_STRICT_DATA"] = "1"
        env["MATRIX0_STRICT_CHECKPOINT"] = "1"
        proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"cycle {cycle} failed with code {proc.returncode}\n{proc.stderr[-4000:]}")

        report_path = cycle_run_dir / "local_loop_report.json"
        report = _load_json(report_path)
        gate = evaluate_cycle_promotion(
            report,
            max_value_mse_delta=float(args.max_value_mse_delta),
            max_policy_ce_delta=float(args.max_policy_ce_delta),
            max_policy_legal_ce_delta=float(args.max_policy_legal_ce_delta),
            max_fresh_capped_fraction=float(args.max_fresh_capped_fraction),
            max_source_value_mse_delta=(
                None if args.max_source_value_mse_delta is None else float(args.max_source_value_mse_delta)
            ),
        )
        candidate = Path(report["checkpoints"]["final"])
        promotion: dict[str, Any] = {
            "cycle": cycle,
            "parent_checkpoint": str(current_parent),
            "candidate_checkpoint": str(candidate),
            "gate": gate,
            "promoted_checkpoint": None,
            "archived_previous_best": None,
        }
        if gate["promote"]:
            best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            promotion["archived_previous_best"] = _archive_existing(best_checkpoint, base_run_dir / "archives")
            shutil.copy2(candidate, best_checkpoint)
            current_parent = best_checkpoint
            promotion["promoted_checkpoint"] = str(best_checkpoint)
        elif bool(args.stop_on_reject):
            cycle_reports.append({"report": str(report_path), "promotion": promotion, "stdout_tail": proc.stdout[-4000:]})
            break

        cycle_reports.append({"report": str(report_path), "promotion": promotion, "stdout_tail": proc.stdout[-4000:]})

    summary = {
        "type": "matrix0_local_loop_cycles",
        "timestamp": datetime.now().isoformat(),
        "base_run_dir": str(base_run_dir),
        "best_checkpoint": str(best_checkpoint),
        "final_parent_checkpoint": str(current_parent),
        "cycles": cycle_reports,
    }
    out_path = Path(args.output) if args.output else base_run_dir / "cycle_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["output"] = str(out_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated gated Matrix0 local-loop cycles. Put bench_local_loop args after --."
    )
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed-best-checkpoint", action="store_true", help="Copy the initial checkpoint to --best-checkpoint if missing.")
    parser.add_argument("--stop-on-reject", action="store_true", help="Stop immediately after the first rejected cycle.")
    parser.add_argument("--max-value-mse-delta", type=float, default=-1.0e-12, help="Largest allowed value MSE delta; default requires a strict improvement.")
    parser.add_argument("--max-policy-ce-delta", type=float, default=0.001)
    parser.add_argument("--max-policy-legal-ce-delta", type=float, default=1.0e-5)
    parser.add_argument("--max-fresh-capped-fraction", type=float, default=1.0)
    parser.add_argument(
        "--max-source-value-mse-delta",
        type=float,
        default=None,
        help="Optional largest allowed per-source held-out value MSE delta for every eval.source_delta entry.",
    )
    raw_args = sys.argv[1:]
    if "-h" in raw_args or "--help" in raw_args:
        parser.parse_args(raw_args)
        return
    if "--" not in raw_args:
        raise SystemExit("Pass bench_local_loop arguments after --")
    split_at = raw_args.index("--")
    args = parser.parse_args(raw_args[:split_at])
    bench_args = raw_args[split_at + 1 :]
    if not bench_args:
        raise SystemExit("Pass bench_local_loop arguments after --")

    summary = run_cycles(args, bench_args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
