#!/usr/bin/env python3
"""Execute Solo12 contact-robustness stages 0, 1, and 2."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("TRAINS_WORKSPACE_ROOT", PROJECT_ROOT.parents[1]))
LOG_ROOT = ROOT / "outputs/quadruped-jumping-ada/logs"
RUN_ROOT = LOG_ROOT / "test_solo12_v3_1"
HOST_OUTPUT = LOG_ROOT / "diagnostics/solo12_contact_stages"
ARTIFACT_OUTPUT = ROOT / "artifacts/solo12_contact_stages"
CONTAINER_OUTPUT = Path(
    "/workspace/project/legged_gym/logs/diagnostics/solo12_contact_stages"
)
STATE_PATH = ARTIFACT_OUTPUT / "state.json"
BASE_RUN = "Aug17_10-01-20_solver8_forward_from_upward11050_3000"
BASE_CHECKPOINT = 13900
UPWARD_RUN = "Aug17_09-39-13_solver8_adapt_original_from10550_500"
UPWARD_CHECKPOINT = 11050
FRICTIONS = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
BUCKETS = {
    "A": (0.010, 0.000, 0.010),
    "B": (0.005, 0.000, 0.005),
    "C": (0.010, -0.002, 0.010),
}


def cli(kind, *args):
    return [sys.executable, "-m", "trains.cli", kind, "quadruped-jumping-ada", *args]


def run(command, log_name):
    ARTIFACT_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_OUTPUT / log_name
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )


def contact_args(bucket):
    contact, rest, threshold = BUCKETS[bucket]
    return [
        "--physx_contact_offset_override", str(contact),
        "--physx_rest_offset_override", str(rest),
        "--physx_friction_offset_threshold_override", str(threshold),
    ]


def container_path(name):
    return str(CONTAINER_OUTPUT / name)


def host_path(name):
    return HOST_OUTPUT / name


def trace_metrics(path):
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    if not records:
        return {}
    dt = float(records[1]["time_s"] - records[0]["time_s"]) if len(records) > 1 else 0.02
    settled = next((i for i, row in enumerate(records) if row["settled_after_init"]), 0)
    contact = next((i for i in range(settled, len(records)) if all(records[i]["foot_contacts"])), None)
    if contact is None:
        return {"valid_takeoff": False}
    takeoff = next((i for i in range(contact + 1, len(records) - 1)
                    if not any(records[i]["foot_contacts"])
                    and not any(records[i + 1]["foot_contacts"])), None)
    if takeoff is None:
        return {"valid_takeoff": False}
    front_only_steps = 0
    rear_takeoff = None
    for i in range(contact + 1, takeoff + 1):
        feet = records[i]["foot_contacts"]
        rear_off = not feet[2] and not feet[3]
        if rear_off and rear_takeoff is None:
            rear_takeoff = i
        if rear_off and (feet[0] or feet[1]):
            front_only_steps += 1
    row = records[takeoff]
    return {
        "valid_takeoff": True,
        "takeoff_time_s": float(row["time_s"]),
        "front_only_support_s": front_only_steps * dt,
        "takeoff_pitch_deg": math.degrees(float(row["euler_rpy"][1])),
        "takeoff_pitch_rate_rad_s": float(row["angular_velocity"][1]),
        "rear_to_all_off_s": ((takeoff - rear_takeoff) * dt if rear_takeoff is not None else None),
    }


def score_case(metrics, trace):
    score = metrics["success_rate"] - metrics["termination_rate"]
    score -= max(0.0, 0.95 - metrics["height_mean"] / 0.68)
    if trace.get("valid_takeoff"):
        score -= max(0.0, abs(trace["takeoff_pitch_deg"]) - 35.0) / 35.0
        score -= max(0.0, abs(trace["takeoff_pitch_rate_rad_s"]) - 3.0) / 3.0
        score -= max(0.0, trace["front_only_support_s"] - 0.06) / 0.06
    else:
        score -= 1.0
    return score


def evaluate(
    run_name,
    checkpoint,
    bucket,
    friction,
    latency,
    prefix,
    episodes=16,
    task="solo12_v3_1_forward",
):
    stem = f"{prefix}_b{bucket}_f{friction:g}_l{latency:g}"
    result_name = stem + ".json"
    trace_name = stem + "_trace.json"
    if host_path(result_name).exists() and host_path(trace_name).exists():
        result = json.loads(host_path(result_name).read_text(encoding="utf-8"))
        trace = trace_metrics(host_path(trace_name))
        return {"bucket": bucket, "friction": friction, "latency_ms": latency,
                "metrics": result["metrics"], "trace": trace,
                "score": score_case(result["metrics"], trace)}
    command = cli(
        "eval", "--task", task, "--load_run", run_name,
        "--checkpoint", str(checkpoint), "--eval_mode", "nominal",
        "--target_x", "0.5", "--target_y", "0",
        "--eval_episodes", str(episodes), "--num_envs", str(episodes),
        "--seed", "43", "--fixed_friction", str(friction),
        "--fixed_latency_ms", str(latency), "--landing_stability_seconds", "1.0",
        "--diagnostics", "--output_json", container_path(result_name),
        "--trajectory_output_json", container_path(trace_name), *contact_args(bucket),
    )
    run(command, stem + ".log")
    result = json.loads(host_path(result_name).read_text(encoding="utf-8"))
    trace = trace_metrics(host_path(trace_name))
    return {"bucket": bucket, "friction": friction, "latency_ms": latency,
            "metrics": result["metrics"], "trace": trace,
            "score": score_case(result["metrics"], trace)}


def load_state():
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def save_state(state):
    ARTIFACT_OUTPUT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage0():
    config_name = "c13900_effective_config.json"
    run(cli(
        "train", "--task", "solo12_v3_1_forward", "--resume",
        "--load_run", UPWARD_RUN, "--checkpoint", str(UPWARD_CHECKPOINT),
        "--num_envs", "4096", "--max_iterations", "3000", "--seed", "1",
        "--run_name", "solver8_forward_from_upward11050_3000",
        "--group_name", "solo_solver8_forward",
        "--com_range_half_width_xyz", "0.02,0.02,0.02",
        "--latency_range_max_ms", "40",
        "--effective_config_output", container_path(config_name), "--config_only",
    ), "stage0_config.log")
    cases = [evaluate(BASE_RUN, BASE_CHECKPOINT, bucket, friction, 0,
                      "stage0_c13900", episodes=16)
             for bucket in BUCKETS for friction in FRICTIONS]
    aggregate = {friction: sum(case["score"] for case in cases
                               if case["friction"] == friction) / len(BUCKETS)
                 for friction in FRICTIONS}
    boundary = max(zip(FRICTIONS[:-1], FRICTIONS[1:]),
                   key=lambda pair: abs(aggregate[pair[1]] - aggregate[pair[0]]))
    state = load_state()
    state["stage0"] = {
        "status": "complete", "base_run": BASE_RUN,
        "base_checkpoint": BASE_CHECKPOINT,
        "effective_config": str(host_path(config_name)),
        "config_provenance": {
            "proven": ["parent upward checkpoint 11050", "COM xyz half width 0.02 m", "latency range 0-40 ms"],
            "inherited_from_source": ["8 Hz action filter", "PhysX TGS with 8 position iterations", "zero/max latency probabilities 0.5/0.25"],
        },
        "friction_boundary": list(boundary), "cases": cases,
    }
    save_state(state)
    print(json.dumps(state["stage0"], indent=2, sort_keys=True))


def newest_run(run_tag, before):
    candidates = [path for path in RUN_ROOT.glob(f"*{run_tag}*") if path.name not in before]
    if not candidates:
        raise RuntimeError(f"No new run found for tag {run_tag}")
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def stage1():
    state = load_state()
    if state.get("stage0", {}).get("status") != "complete":
        raise RuntimeError("Run stage0 before stage1")
    boundary = state["stage0"]["friction_boundary"]
    segments = state.get("stage1", {}).get("segments", [])
    if segments:
        current_run = segments[-1]["run"]
        checkpoint = segments[-1]["checkpoint"]
    else:
        current_run, checkpoint = BASE_RUN, BASE_CHECKPOINT
    schedule = list(zip(("A", "B", "C", "A", "B", "C"),
                        (0.8, 0.8, 0.4, 0.4, 0.0, 0.0)))
    for index, (bucket, push_probability) in enumerate(schedule, 1):
        if index <= len(segments):
            continue
        tag = f"contact_robust_s{index}_b{bucket}_p{push_probability:g}"
        next_checkpoint = checkpoint + 50
        completed_runs = [
            path for path in RUN_ROOT.glob(f"*{tag}*")
            if (path / f"model_{next_checkpoint}.pt").exists()
        ]
        if completed_runs:
            current_run = max(
                completed_runs, key=lambda path: path.stat().st_mtime
            ).name
        else:
            before = {path.name for path in RUN_ROOT.iterdir() if path.is_dir()}
            run(cli(
                "train", "--task", "solo12_v3_1_forward", "--num_envs", "4096",
                "--max_iterations", "50", "--seed", "1", "--resume",
                "--load_run", current_run, "--checkpoint", str(checkpoint),
                "--run_name", tag, "--group_name", "solo_contact_robustness",
                "--fixed_target_x", "0.5", "--fixed_target_y", "0",
                "--com_range_half_width_xyz", "0.02,0.02,0.02",
                "--latency_range_max_ms", "40", "--friction_range_override", "0.1,3.0",
                "--friction_boundary_range", f"{boundary[0]},{boundary[1]}",
                "--friction_boundary_probability", "0.6",
                "--push_towards_goal_probability_override", str(push_probability),
                "--push_towards_goal_final_probability_override", str(push_probability),
                "--effective_config_output", container_path(tag + "_config.json"),
                *contact_args(bucket),
            ), tag + "_train.log")
            current_run = newest_run(tag, before)
        checkpoint = next_checkpoint
        midpoint = sum(boundary) / 2.0
        evaluations = [evaluate(current_run, checkpoint, eval_bucket, midpoint, 20,
                                f"stage1_c{checkpoint}", episodes=16)
                       for eval_bucket in BUCKETS]
        segments.append({"index": index, "training_bucket": bucket,
                         "push_probability": push_probability, "run": current_run,
                         "checkpoint": checkpoint, "evaluations": evaluations})
        state["stage1"] = {"status": "in_progress", "segments": segments}
        save_state(state)
    state["stage1"].update({"status": "complete", "final_run": current_run,
                            "final_checkpoint": checkpoint})
    save_state(state)
    print(json.dumps(state["stage1"], indent=2, sort_keys=True))


def matrix(
    run_name,
    checkpoint,
    prefix,
    frictions,
    task="solo12_v3_1_forward",
):
    conditions = [
        (bucket, friction, latency)
        for bucket in BUCKETS
        for friction in frictions
        for latency in (0, 20, 40)
    ]
    def evaluate_condition(condition):
        bucket, friction, latency = condition
        return evaluate(
            run_name,
            checkpoint,
            bucket,
            friction,
            latency,
            prefix,
            episodes=32,
            task=task,
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        return list(executor.map(evaluate_condition, conditions))


def summarize_matrix(cases):
    successes = [case["metrics"]["success_rate"] for case in cases]
    terminations = [case["metrics"]["termination_rate"] for case in cases]
    heights = [case["metrics"]["height_mean"] for case in cases]
    scores = [case["score"] for case in cases]
    traces = [case["trace"] for case in cases if case["trace"].get("valid_takeoff")]
    average = lambda key: (sum(row[key] for row in traces) / len(traces) if traces else None)
    return {
        "success_mean": sum(successes) / len(successes), "success_min": min(successes),
        "termination_mean": sum(terminations) / len(terminations),
        "height_mean": sum(heights) / len(heights),
        "score_mean": sum(scores) / len(scores), "score_min": min(scores),
        "valid_takeoff_fraction": len(traces) / len(cases),
        "takeoff_time_mean_s": average("takeoff_time_s"),
        "takeoff_time_max_s": (
            max(row["takeoff_time_s"] for row in traces) if traces else None
        ),
        "front_only_support_mean_s": average("front_only_support_s"),
        "takeoff_pitch_abs_mean_deg": (sum(abs(row["takeoff_pitch_deg"]) for row in traces) / len(traces) if traces else None),
        "takeoff_pitch_rate_abs_mean_rad_s": (sum(abs(row["takeoff_pitch_rate_rad_s"]) for row in traces) / len(traces) if traces else None),
    }


def upward_audit():
    cases = matrix(
        UPWARD_RUN,
        UPWARD_CHECKPOINT,
        "upward_u0_c11050",
        (0.7, 0.85, 1.0),
        task="solo12_v3_1_upwards",
    )
    summary = summarize_matrix(cases)
    by_bucket = {
        bucket: summarize_matrix(
            [case for case in cases if case["bucket"] == bucket]
        )
        for bucket in BUCKETS
    }
    gates = {
        "valid_takeoff_at_least_95pct": (
            summary["valid_takeoff_fraction"] >= 0.95
        ),
        "takeoff_time_at_most_0p6s": (
            summary["takeoff_time_max_s"] is not None
            and summary["takeoff_time_max_s"] <= 0.6
        ),
        "front_support_at_most_60ms": (
            summary["front_only_support_mean_s"] is not None
            and summary["front_only_support_mean_s"] <= 0.06
        ),
        "takeoff_pitch_at_most_15deg": (
            summary["takeoff_pitch_abs_mean_deg"] is not None
            and summary["takeoff_pitch_abs_mean_deg"] <= 15.0
        ),
        "takeoff_pitch_rate_at_most_3rad_s": (
            summary["takeoff_pitch_rate_abs_mean_rad_s"] is not None
            and summary["takeoff_pitch_rate_abs_mean_rad_s"] <= 3.0
        ),
        "termination_mean_at_most_20pct": (
            summary["termination_mean"] <= 0.2
        ),
        "no_systematic_bucket_failure": summary["success_min"] >= 0.7,
    }
    qualified = all(gates.values())
    audit = {
        "status": "complete",
        "run": UPWARD_RUN,
        "checkpoint": UPWARD_CHECKPOINT,
        "qualified_for_forward_restart": qualified,
        "decision": (
            "start_forward_curriculum"
            if qualified
            else "repair_upward_before_forward"
        ),
        "gates": gates,
        "summary": summary,
        "by_bucket": by_bucket,
        "cases": cases,
    }
    state = load_state()
    state["upward_audit"] = audit
    save_state(state)
    json_path = ARTIFACT_OUTPUT / "upward_c11050_audit.json"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = ARTIFACT_OUTPUT / "upward_c11050_audit.md"
    lines = [
        "# Solo12 upward c11050 audit",
        "",
        f"Decision: `{audit['decision']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in gates.items()
    )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, indent=2, sort_keys=True),
            "```",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


def stage2():
    state = load_state()
    if state.get("stage1", {}).get("status") != "complete":
        raise RuntimeError("Run stage1 before stage2")
    lo, hi = state["stage0"]["friction_boundary"]
    selected_friction = sorted({lo, (lo + hi) / 2.0, hi})
    final_run, final_checkpoint = state["stage1"]["final_run"], state["stage1"]["final_checkpoint"]
    baseline_cases = matrix(BASE_RUN, BASE_CHECKPOINT, "stage2_baseline", selected_friction)
    candidate_cases = matrix(final_run, final_checkpoint, "stage2_candidate", selected_friction)
    baseline, candidate = summarize_matrix(baseline_cases), summarize_matrix(candidate_cases)
    gates = {
        "mean_success_not_regressed": candidate["success_mean"] >= baseline["success_mean"] - 0.05,
        "worst_success_not_regressed": candidate["success_min"] >= baseline["success_min"] - 0.10,
        "mean_termination_not_worse": candidate["termination_mean"] <= baseline["termination_mean"] + 0.03,
        "height_retained": candidate["height_mean"] >= 0.95 * baseline["height_mean"],
        "robust_score_improved": candidate["score_mean"] > baseline["score_mean"],
        "valid_takeoff": candidate["valid_takeoff_fraction"] >= 0.95,
        "front_support_gate": candidate["front_only_support_mean_s"] is not None and candidate["front_only_support_mean_s"] <= 0.06,
        "pitch_gate": candidate["takeoff_pitch_abs_mean_deg"] is not None and candidate["takeoff_pitch_abs_mean_deg"] <= 35.0,
        "pitch_rate_gate": candidate["takeoff_pitch_rate_abs_mean_rad_s"] is not None and candidate["takeoff_pitch_rate_abs_mean_rad_s"] <= 3.0,
    }
    decision = "continue_c13900_lineage" if all(gates.values()) else "restart_from_upward_c11050"
    state["stage2"] = {"status": "complete", "decision": decision,
                       "baseline": baseline, "candidate": candidate, "gates": gates,
                       "baseline_cases": baseline_cases, "candidate_cases": candidate_cases}
    save_state(state)
    report = ARTIFACT_OUTPUT / "stage2_decision.md"
    lines = ["# Solo12 contact robustness stage 2 decision", "", f"Decision: `{decision}`", "",
             f"Baseline: `{BASE_RUN}/model_{BASE_CHECKPOINT}.pt`",
             f"Candidate: `{final_run}/model_{final_checkpoint}.pt`", "", "## Gates", ""]
    lines.extend(f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in gates.items())
    lines.extend(["", "## Aggregate metrics", "", "```json",
                  json.dumps({"baseline": baseline, "candidate": candidate}, indent=2, sort_keys=True),
                  "```", "", "MuJoCo remains an external holdout. Check this decision against the existing MuJoCo A/B results before real-hardware deployment."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("stage0", "stage1", "stage2", "upward-audit", "all"),
    )
    args = parser.parse_args()
    if args.stage == "upward-audit":
        upward_audit()
        return
    if args.stage in ("stage0", "all"):
        stage0()
    if args.stage in ("stage1", "all"):
        stage1()
    if args.stage in ("stage2", "all"):
        stage2()


if __name__ == "__main__":
    main()
