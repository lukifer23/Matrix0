from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    description: str
    train_result_source_mix: tuple[str, ...]
    value_include_source: tuple[str, ...]
    value_source_weight: tuple[str, ...] = ()
    trainable_scope: Optional[str] = None


def available_probe_specs() -> Dict[str, ProbeSpec]:
    specs = [
        ProbeSpec(
            name="tablebase_capped",
            description="Train value on tablebase+capped only; terminal heldout reveals zero-terminal conflict.",
            train_result_source_mix=("tablebase=0.70", "capped=0.30"),
            value_include_source=("tablebase", "capped"),
            value_source_weight=("capped=2.0",),
        ),
        ProbeSpec(
            name="terminal_only",
            description="Train value on terminal only; tablebase/capped heldout reveal terminal-zero pull.",
            train_result_source_mix=("terminal=1.0",),
            value_include_source=("terminal",),
        ),
        ProbeSpec(
            name="balanced",
            description="Current balanced scout shape without fresh self-play noise.",
            train_result_source_mix=("terminal=0.30", "tablebase=0.50", "capped=0.20"),
            value_include_source=("tablebase", "terminal", "capped"),
            value_source_weight=("terminal=2.5", "capped=2.0"),
        ),
        ProbeSpec(
            name="balanced_value_head",
            description="Balanced scout shape with only value_head trainable; separates global/trunk drift from head calibration.",
            train_result_source_mix=("terminal=0.30", "tablebase=0.50", "capped=0.20"),
            value_include_source=("tablebase", "terminal", "capped"),
            value_source_weight=("terminal=2.5", "capped=2.0"),
            trainable_scope="value_head",
        ),
        ProbeSpec(
            name="tbprotect",
            description="Tablebase-protected scout shape that previously promoted but can underprotect terminal.",
            train_result_source_mix=("terminal=0.25", "tablebase=0.60", "capped=0.15"),
            value_include_source=("tablebase", "terminal", "capped"),
            value_source_weight=("terminal=2.0", "capped=1.5"),
        ),
    ]
    return {spec.name: spec for spec in specs}


def default_probe_specs(include_value_head_probe: bool = False) -> List[ProbeSpec]:
    specs = available_probe_specs()
    names = ["tablebase_capped", "terminal_only", "balanced"]
    if include_value_head_probe:
        names.append("balanced_value_head")
    return [specs[name] for name in names]


def _source_metric(report: Dict[str, Any], source: str, metric: str) -> Optional[float]:
    value = (
        report.get("eval", {})
        .get("source_delta", {})
        .get(source, {})
        .get(metric)
    )
    return None if value is None else float(value)


def summarize_probe(report: Dict[str, Any], metric: str) -> Dict[str, Any]:
    delta = report.get("eval", {}).get("delta", {})
    selection = report.get("eval", {}).get("selection", {})
    source_pressure = report.get("source_pressure", {})
    candidates = selection.get("candidates", []) if isinstance(selection, dict) else []
    return {
        "output": report.get("output"),
        "run_dir": report.get("run_dir"),
        "selection": {
            "enabled": selection.get("enabled") if isinstance(selection, dict) else None,
            "selected_chunk": selection.get("selected_chunk") if isinstance(selection, dict) else None,
            "best_metric_value": selection.get("best_metric_value") if isinstance(selection, dict) else None,
            "candidate_failures": [
                {
                    "chunk": candidate.get("chunk"),
                    "metric_value": candidate.get("metric_value"),
                    "selection_failures": candidate.get("selection_failures", []),
                }
                for candidate in candidates
            ],
        },
        "aggregate": {
            "value_weighted_mse": delta.get("value_weighted_mse"),
            "value_mse": delta.get("value_mse"),
            "policy_ce": delta.get("policy_ce"),
            "policy_legal_ce": delta.get("policy_legal_ce"),
        },
        "source_delta": {
            source: {
                metric: _source_metric(report, source, metric),
                "value_mse": _source_metric(report, source, "value_mse"),
                "value_pred_mean": _source_metric(report, source, "value_pred_mean"),
            }
            for source in ("capped", "tablebase", "terminal")
        },
        "source_pressure": {
            source: source_pressure.get("sources", {}).get(source, {})
            for source in ("capped", "tablebase", "terminal")
        },
    }


def classify_conflict(probes: Dict[str, Dict[str, Any]], metric: str) -> Dict[str, Any]:
    def val(probe: str, source: str) -> Optional[float]:
        return probes.get(probe, {}).get("source_delta", {}).get(source, {}).get(metric)

    tb_cap_terminal = val("tablebase_capped", "terminal")
    terminal_only_capped = val("terminal_only", "capped")
    terminal_only_tablebase = val("terminal_only", "tablebase")
    balanced_capped = val("balanced", "capped")
    balanced_terminal = val("balanced", "terminal")

    conflict = (
        tb_cap_terminal is not None
        and tb_cap_terminal > 0.0
        and (
            (terminal_only_capped is not None and terminal_only_capped > 0.0)
            or (terminal_only_tablebase is not None and terminal_only_tablebase > 0.0)
        )
    )
    balanced_overprotects_terminal = (
        balanced_terminal is not None
        and balanced_terminal < 0.0
        and balanced_capped is not None
        and balanced_capped > 0.0
    )
    bias_like_updates: Dict[str, Any] = {}
    for name, probe in probes.items():
        shifts = []
        for source in ("capped", "tablebase", "terminal"):
            shift = probe.get("source_delta", {}).get(source, {}).get("value_pred_mean")
            if shift is not None:
                shifts.append(float(shift))
        if shifts:
            same_sign = all(shift >= 0.0 for shift in shifts) or all(shift <= 0.0 for shift in shifts)
            abs_shifts = [abs(shift) for shift in shifts if abs(shift) > 0.0]
            spread = (max(abs_shifts) / min(abs_shifts)) if len(abs_shifts) >= 2 else None
            bias_like_updates[name] = {
                "same_sign_value_pred_mean_shift": bool(same_sign),
                "abs_shift_spread": spread,
                "shifts": shifts,
            }

    return {
        "metric": metric,
        "source_conflict_detected": bool(conflict),
        "balanced_overprotects_terminal": bool(balanced_overprotects_terminal),
        "global_bias_like_updates": bias_like_updates,
        "notes": [
            "Positive delta means heldout regression; negative delta means heldout improvement.",
            "A conflict is indicated when tablebase+capped training regresses terminal, while terminal-only training regresses tablebase or capped.",
            "same_sign_value_pred_mean_shift across sources means the update is behaving like a global value bias shift, not source-specific value learning.",
            "If balanced_overprotects_terminal is true, reduce terminal pressure, lower LR, or change value-head capacity/objective before running more promotion cycles.",
        ],
    }


def _append_repeated(cmd: List[str], flag: str, values: Iterable[str]) -> None:
    for value in values:
        cmd.extend([flag, str(value)])


def build_probe_command(args: argparse.Namespace, spec: ProbeSpec, run_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "azchess.tools.bench_local_loop",
        "--config",
        str(args.config),
        "--run-dir",
        str(run_dir),
        "--seed",
        str(int(args.seed)),
        "--games",
        "0",
        "--workers",
        "1",
        "--sims",
        str(int(args.sims)),
        "--max-game-len",
        str(int(args.max_game_len)),
        "--train-steps",
        str(int(args.train_steps)),
        "--eval-select-interval",
        str(int(args.eval_select_interval)),
        "--eval-select-metric",
        str(args.eval_select_metric),
        "--eval-select-source-metric",
        str(args.eval_select_source_metric),
        "--eval-select-max-policy-ce-delta",
        str(float(args.eval_select_max_policy_ce_delta)),
        "--eval-select-max-policy-legal-ce-delta",
        str(float(args.eval_select_max_policy_legal_ce_delta)),
        "--batch-size",
        str(int(args.batch_size)),
        "--eval-batch-size",
        str(int(args.eval_batch_size)),
        "--eval-batches",
        str(int(args.eval_batches)),
        "--eval-data-dir",
        str(args.anchor_data_dir),
        "--eval-result-source",
        "tablebase",
        "--eval-result-source",
        "terminal",
        "--eval-result-source",
        "capped",
        "--dataloader-workers",
        "0",
        "--mps-target-batch",
        str(int(args.mps_target_batch)),
        "--checkpoint-state",
        "model",
        "--initial-checkpoint-state",
        "model_ema",
        "--trainable-scope",
        str(spec.trainable_scope or args.trainable_scope),
        "--legal-mass-weight",
        "0.0",
        "--legal-policy-weight",
        "0.0",
        "--capped-value-weight",
        str(float(args.capped_value_weight)),
        "--policy-label-smoothing",
        "0.0",
        "--lr",
        str(float(args.lr)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--ssl-weight",
        "0.0",
        "--init-checkpoint",
        str(args.parent_checkpoint),
        "--train-anchor-data-dir",
        str(args.anchor_data_dir),
        "--train-anchor-source-prefix",
        "tablebase",
        "--train-anchor-source-prefix",
        "terminal",
        "--train-anchor-source-prefix",
        "capped",
        "--policy-include-source",
        "__none__",
        "--policy-distill-checkpoint",
        str(args.parent_checkpoint),
        "--policy-distill-weight",
        str(float(args.policy_distill_weight)),
        "--policy-distill-temperature",
        "1.0",
    ]
    if args.eval_select_max_source_value_mse_delta is not None:
        cmd.extend([
            "--eval-select-max-source-value-mse-delta",
            str(float(args.eval_select_max_source_value_mse_delta)),
        ])
    if float(getattr(args, "value_mean_distill_weight", 0.0) or 0.0) > 0.0:
        cmd.extend(["--value-mean-distill-weight", str(float(args.value_mean_distill_weight))])
    if float(getattr(args, "value_distill_weight", 0.0) or 0.0) > 0.0:
        cmd.extend(["--value-distill-weight", str(float(args.value_distill_weight))])
    if bool(args.eval_full_dataset):
        cmd.append("--eval-full-dataset")
    _append_repeated(cmd, "--train-result-source-mix", spec.train_result_source_mix)
    _append_repeated(cmd, "--value-include-source", spec.value_include_source)
    _append_repeated(cmd, "--value-source-weight", spec.value_source_weight)
    return cmd


def run_probe(args: argparse.Namespace, spec: ProbeSpec) -> Dict[str, Any]:
    run_dir = Path(args.base_run_dir) / spec.name
    cmd = build_probe_command(args, spec, run_dir)
    if bool(args.dry_run):
        return {"name": spec.name, "description": spec.description, "run_dir": str(run_dir), "command": cmd}
    proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe {spec.name} failed with code {proc.returncode}\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-4000:]}"
        )
    report_path = run_dir / "local_loop_report.json"
    with report_path.open("r") as f:
        report = json.load(f)
    summary = summarize_probe(report, str(args.metric))
    summary.update(
        {
            "name": spec.name,
            "description": spec.description,
            "command": cmd,
            "stdout_tail": proc.stdout[-2000:],
        }
    )
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.probe:
        available = available_probe_specs()
        specs = [available[name] for name in args.probe]
    else:
        specs = default_probe_specs(include_value_head_probe=bool(args.include_value_head_probe))
    probes = [run_probe(args, spec) for spec in specs]
    by_name = {str(probe["name"]): probe for probe in probes}
    report = {
        "type": "matrix0_source_conflict_probe",
        "timestamp": datetime.now().isoformat(),
        "base_run_dir": str(args.base_run_dir),
        "anchor_data_dir": str(args.anchor_data_dir),
        "parent_checkpoint": str(args.parent_checkpoint),
        "metric": str(args.metric),
        "dry_run": bool(args.dry_run),
        "probes": by_name,
        "classification": classify_conflict(by_name, str(args.metric)) if not bool(args.dry_run) else None,
    }
    out_path = Path(args.output) if args.output else Path(args.base_run_dir) / "source_conflict_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["output"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anchor-only source conflict probes for Matrix0 value training.")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--anchor-data-dir", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=2026051400)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--max-game-len", type=int, default=240)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-select-interval", type=int, default=0)
    parser.add_argument("--eval-select-metric", default="value_weighted_mse")
    parser.add_argument("--eval-select-source-metric", default="value_weighted_mse")
    parser.add_argument("--eval-select-max-policy-ce-delta", type=float, default=0.001)
    parser.add_argument("--eval-select-max-policy-legal-ce-delta", type=float, default=1.0e-5)
    parser.add_argument("--eval-select-max-source-value-mse-delta", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--eval-full-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mps-target-batch", type=int, default=4)
    parser.add_argument("--trainable-scope", default="all", choices=["all", "value_head", "moves_left_head", "value_and_moves_left"])
    parser.add_argument("--probe", action="append", choices=sorted(available_probe_specs().keys()), help="Run only this named probe. Repeatable. Defaults to the three core probes.")
    parser.add_argument("--include-value-head-probe", action="store_true", help="Also run the balanced_value_head probe.")
    parser.add_argument("--capped-value-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1.5e-8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--policy-distill-weight", type=float, default=1.0)
    parser.add_argument("--value-distill-weight", type=float, default=0.0)
    parser.add_argument("--value-mean-distill-weight", type=float, default=0.0)
    parser.add_argument("--metric", default="value_weighted_mse")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
