#!/usr/bin/env python3
"""Run and audit the Solo12 upward-from-scratch S1-S4 curriculum."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("TRAINS_WORKSPACE_ROOT", PROJECT_ROOT.parents[1]))
LOG_ROOT = ROOT / "outputs/quadruped-jumping-ada/logs"
RUN_ROOT = LOG_ROOT / "test_solo12_v3_1"
HOST_OUTPUT = LOG_ROOT / "upward_retrain"
CONTAINER_OUTPUT = Path("/workspace/project/legged_gym/logs/upward_retrain")
ARTIFACT_OUTPUT = ROOT / "artifacts/solo12_upward_retrain"
STATE_PATH = ARTIFACT_OUTPUT / "state.json"
TASK = "solo12_v3_1_upwards"

BUCKETS = {
    "A": (0.010, 0.000, 0.010),
    "B": (0.005, 0.000, 0.005),
    "C": (0.010, -0.002, 0.010),
}

REWARD_ARGS = (
    "--takeoff_pitch_angular_impulse_scale_override", "-200",
)

FORMAL_STAGES = ("s1", "s2", "s3", "s4")


def cli(kind: str, *args: str) -> list[str]:
    return [sys.executable, "-m", "trains.cli", kind,
            "quadruped-jumping-ada", *args]


def contact_args(bucket: str) -> list[str]:
    contact, rest, threshold = BUCKETS[bucket]
    return [
        "--physx_contact_offset_override", str(contact),
        "--physx_rest_offset_override", str(rest),
        "--physx_friction_offset_threshold_override", str(threshold),
    ]


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"status": "in_progress", "stages": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    ARTIFACT_OUTPUT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def newest_run(tag: str, before: set[str]) -> str:
    candidates = [
        path for path in RUN_ROOT.glob(f"*{tag}*")
        if path.name not in before
    ]
    if not candidates:
        raise RuntimeError(f"No new run found for tag {tag}")
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def train_stage(args: argparse.Namespace) -> None:
    tag = args.tag or f"solo12_upward_retrain_{args.stage}_seed{args.seed}"
    before = {path.name for path in RUN_ROOT.iterdir() if path.is_dir()}
    command = cli(
        "train", "--task", TASK,
        "--seed", str(args.seed),
        "--max_iterations", str(args.iterations),
        "--run_name", tag,
        "--effective_config_output",
        str(CONTAINER_OUTPUT / f"{tag}_effective.json"),
        *REWARD_ARGS,
    )
    if args.parent_run is not None:
        if args.parent_checkpoint is None:
            raise ValueError("--parent-checkpoint is required with --parent-run")
        command.extend([
            "--resume", "--load_run", args.parent_run,
            "--checkpoint", str(args.parent_checkpoint),
        ])
    elif args.parent_checkpoint is not None:
        raise ValueError("--parent-run is required with --parent-checkpoint")
    run(command, ARTIFACT_OUTPUT / "logs" / f"{tag}.log")
    run_name = newest_run(tag, before)
    state = load_state()
    stage_runs = state["stages"].setdefault(args.stage, {}).setdefault("runs", [])
    stage_runs.append({
        "seed": args.seed,
        "run": run_name,
        "start_checkpoint": args.parent_checkpoint,
        "iterations": args.iterations,
        "final_checkpoint": (args.parent_checkpoint or 0) + args.iterations,
        "from_random_initialization": args.parent_run is None,
        "effective_config": str(HOST_OUTPUT / f"{tag}_effective.json"),
    })
    save_state(state)
    print(json.dumps(stage_runs[-1], indent=2, sort_keys=True))


def field_mean(result: dict, name: str) -> float:
    field = result.get("diagnostics", {}).get("fields", {}).get(name, {})
    value = field.get("mean")
    return float(value) if value is not None else math.nan


def result_summary(result: dict) -> dict:
    metrics = result["metrics"]
    jump_audit = result.get("diagnostics", {}).get("jump_detection_audit", {})
    return {
        "height_mean": float(metrics["height_mean"]),
        "height_p10": float(metrics["height_p10"]),
        "success_rate": float(metrics["success_rate"]),
        "termination_rate": float(metrics["termination_rate"]),
        "physical_takeoff_rate": float(
            jump_audit.get("physical_takeoff_rate", 0.0)
        ),
        "torque_saturation_fraction": float(
            metrics["torque_saturation_fraction"]
        ),
        "takeoff_pitch_abs_deg_mean": field_mean(
            result, "takeoff_pitch_abs_deg"
        ),
        "contact_mismatch_seconds_mean": field_mean(
            result, "pre_takeoff_contact_mismatch_seconds"
        ),
        "pitch_angular_impulse_normalized_mean": field_mean(
            result, "takeoff_pitch_angular_impulse_normalized"
        ),
        "orientation_termination_rate": field_mean(
            result, "termination_reason_orientation"
        ),
        "action_rate_termination_rate": field_mean(
            result, "termination_reason_action_rate"
        ),
    }


def evaluation_cases(suite: str) -> list[dict]:
    nominal = [
        {
            "name": "nominal_A", "mode": "nominal", "bucket": "A",
            "extra": ["--fixed_friction", "0.85", "--fixed_latency_ms", "0"],
        }
    ]
    if suite == "s1":
        return nominal
    abc = [
        {"name": f"robust_{bucket}", "mode": "robust", "bucket": bucket,
         "extra": []}
        for bucket in BUCKETS
    ]
    if suite in {"s2", "s3"}:
        return nominal + abc
    endpoints = [
        {
            "name": "endpoint_low", "mode": "nominal", "bucket": "B",
            "extra": [
                "--fixed_friction", "0.3", "--fixed_restitution", "0",
                "--fixed_joint_friction", "0", "--fixed_joint_damping", "0",
                "--fixed_latency_ms", "0", "--fixed_com_xyz=-0.02,-0.02,-0.02",
            ],
        },
        {
            "name": "endpoint_high", "mode": "nominal", "bucket": "C",
            "extra": [
                "--fixed_friction", "1.5", "--fixed_restitution", "0.4",
                "--fixed_joint_friction", "0.04",
                "--fixed_joint_damping", "0.01", "--fixed_latency_ms", "40",
                "--fixed_com_xyz=0.02,0.02,0.02",
            ],
        },
    ]
    return nominal + abc + endpoints


def evaluate_case(
    run_name: str,
    checkpoint: int,
    seed: int,
    episodes: int,
    case: dict,
    prefix: str,
) -> dict:
    stem = f"{prefix}_{case['name']}_seed{seed}"
    host_json = HOST_OUTPUT / "evaluations" / f"{stem}.json"
    container_json = CONTAINER_OUTPUT / "evaluations" / f"{stem}.json"
    if not host_json.exists():
        command = cli(
            "eval", "--task", TASK,
            "--load_run", run_name, "--checkpoint", str(checkpoint),
            "--eval_mode", case["mode"],
            "--target_x", "0", "--target_y", "0",
            "--eval_episodes", str(episodes), "--num_envs", str(episodes),
            "--seed", str(seed), "--diagnostics",
            "--filter_freq_override", "8", "--clip_actions_override", "100",
            "--output_json", str(container_json),
            *contact_args(case["bucket"]), *case["extra"],
        )
        run(command, ARTIFACT_OUTPUT / "logs" / f"{stem}.log")
    result = json.loads(host_json.read_text(encoding="utf-8"))
    summary = result_summary(result)
    summary.update({
        "case": case["name"], "seed": seed,
        "bucket": case["bucket"], "mode": case["mode"],
        "result": str(host_json),
    })
    return summary


def finite_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.inf


def aggregate_case(rows: list[dict]) -> dict:
    numeric = (
        "height_mean", "height_p10", "success_rate", "termination_rate",
        "physical_takeoff_rate",
        "torque_saturation_fraction", "takeoff_pitch_abs_deg_mean",
        "contact_mismatch_seconds_mean",
        "pitch_angular_impulse_normalized_mean",
        "orientation_termination_rate", "action_rate_termination_rate",
    )
    return {
        name: sum(row[name] for row in rows if math.isfinite(row[name]))
        / max(1, sum(math.isfinite(row[name]) for row in rows))
        for name in numeric
    }


def gate_suite(suite: str, aggregates: dict) -> dict:
    nominal = aggregates["nominal_A"]
    if suite == "s1":
        limits = {
            "height_mean_min": 0.50, "success_rate_min": 0.75,
            "physical_takeoff_rate_min": 0.90,
            "termination_rate_max": 0.25, "takeoff_pitch_deg_max": 15.0,
            "contact_mismatch_s_max": 0.04, "angular_impulse_max": 0.25,
        }
    else:
        limits = {
            "height_mean_min": 0.62, "success_rate_min": 0.80,
            "physical_takeoff_rate_min": 0.95,
            "termination_rate_max": 0.20, "takeoff_pitch_deg_max": 15.0,
            "contact_mismatch_s_max": 0.04, "angular_impulse_max": 0.22,
        }
    checks = {
        "nominal_height": nominal["height_mean"] >= limits["height_mean_min"],
        "nominal_success": nominal["success_rate"] >= limits["success_rate_min"],
        "nominal_physical_takeoff": (
            nominal["physical_takeoff_rate"]
            >= limits["physical_takeoff_rate_min"]
        ),
        "nominal_termination": nominal["termination_rate"] <= limits["termination_rate_max"],
        "nominal_pitch": nominal["takeoff_pitch_abs_deg_mean"] <= limits["takeoff_pitch_deg_max"],
        "nominal_contact_mismatch": nominal["contact_mismatch_seconds_mean"] <= limits["contact_mismatch_s_max"],
        "nominal_angular_impulse": nominal["pitch_angular_impulse_normalized_mean"] <= limits["angular_impulse_max"],
    }
    if suite in {"s2", "s3", "s4"}:
        robust = [aggregates[f"robust_{bucket}"] for bucket in BUCKETS]
        checks.update({
            "abc_height_retention": min(
                row["height_mean"] / max(nominal["height_mean"], 1e-6)
                for row in robust
            ) >= (0.85 if suite == "s2" else 0.90),
            "abc_success_gap": max(
                nominal["success_rate"] - row["success_rate"] for row in robust
            ) <= (0.20 if suite == "s2" else 0.12),
            "abc_pitch": finite_max([
                row["takeoff_pitch_abs_deg_mean"] for row in robust
            ]) <= 18.0,
        })
    if suite == "s4":
        endpoints = [aggregates["endpoint_low"], aggregates["endpoint_high"]]
        checks.update({
            "endpoint_height_retention": min(
                row["height_mean"] / max(nominal["height_mean"], 1e-6)
                for row in endpoints
            ) >= 0.85,
            "endpoint_success": min(row["success_rate"] for row in endpoints) >= 0.70,
            "endpoint_pitch": finite_max([
                row["takeoff_pitch_abs_deg_mean"] for row in endpoints
            ]) <= 20.0,
            "endpoint_angular_impulse": finite_max([
                row["pitch_angular_impulse_normalized_mean"] for row in endpoints
            ]) <= 0.30,
        })
    return {"pass": all(checks.values()), "checks": checks, "limits": limits}


def evaluate_suite(args: argparse.Namespace) -> None:
    prefix = args.prefix or f"{args.suite}_{args.run}_c{args.checkpoint}"
    rows = [
        evaluate_case(
            args.run, args.checkpoint, seed, args.episodes, case, prefix
        )
        for case in evaluation_cases(args.suite)
        for seed in args.seeds
    ]
    aggregates = {
        case["name"]: aggregate_case([
            row for row in rows if row["case"] == case["name"]
        ])
        for case in evaluation_cases(args.suite)
    }
    report = {
        "suite": args.suite, "run": args.run,
        "checkpoint": args.checkpoint, "seeds": args.seeds,
        "episodes_per_seed": args.episodes, "rows": rows,
        "aggregates": aggregates,
        "gate": gate_suite(args.suite, aggregates),
        "scope": {
            "simulator": "PhysX",
            "target": "upward x=0 y=0",
            "excluded": ["MuJoCo", "locomotion", "forward", "landing error"],
        },
    }
    report_path = ARTIFACT_OUTPUT / "reports" / f"{prefix}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    state = load_state()
    evaluations = state["stages"].setdefault(args.suite, {}).setdefault(
        "evaluations", []
    )
    evaluations.append({
        "run": args.run, "checkpoint": args.checkpoint,
        "report": str(report_path), "pass": report["gate"]["pass"],
    })
    save_state(state)
    print(json.dumps({
        "report": str(report_path), "aggregates": aggregates,
        "gate": report["gate"],
    }, indent=2, sort_keys=True))


def show_status(_: argparse.Namespace) -> None:
    print(json.dumps(load_state(), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--stage", choices=FORMAL_STAGES, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--iterations", type=int, required=True)
    train_parser.add_argument("--parent-run")
    train_parser.add_argument("--parent-checkpoint", type=int)
    train_parser.add_argument("--tag")
    train_parser.set_defaults(function=train_stage)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--suite", choices=("s1", "s2", "s3", "s4"), required=True)
    eval_parser.add_argument("--run", required=True)
    eval_parser.add_argument("--checkpoint", type=int, required=True)
    eval_parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    eval_parser.add_argument("--episodes", type=int, default=128)
    eval_parser.add_argument("--prefix")
    eval_parser.set_defaults(function=evaluate_suite)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(function=show_status)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
