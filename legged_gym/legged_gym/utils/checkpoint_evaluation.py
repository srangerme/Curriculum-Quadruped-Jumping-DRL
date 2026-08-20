"""Pure helpers for deterministic checkpoint evaluation and Go2-relative gates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


PROTOCOL_VERSION = "jump-deterministic-v2"
PROTOCOL_FIELDS = (
    "protocol_version",
    "jump_type",
    "mode",
    "seed",
    "num_envs",
    "eval_episodes",
    "target",
    "success_height",
    "action_noise_scale",
    "fixed_physics",
    "episode_sampling",
    "action_source",
)


def build_episode_quotas(num_envs: int, eval_episodes: int) -> np.ndarray:
    """Assign a deterministic, balanced episode quota to every environment."""

    if num_envs <= 0 or eval_episodes <= 0:
        raise ValueError("num_envs and eval_episodes must be positive")
    quotient, remainder = divmod(eval_episodes, num_envs)
    quotas = np.full(num_envs, quotient, dtype=np.int64)
    quotas[:remainder] += 1
    return quotas


def summarize_episodes(
    heights: Iterable[float],
    landing_errors: Iterable[float],
    success_height: float,
) -> Dict[str, Any]:
    heights_array = np.asarray(list(heights), dtype=np.float64)
    if heights_array.size == 0:
        raise ValueError("at least one completed episode is required")

    landing_array = np.asarray(list(landing_errors), dtype=np.float64)
    successful_landing = landing_array[np.isfinite(landing_array)]

    return {
        "height_mean": float(np.mean(heights_array)),
        "height_median": float(np.median(heights_array)),
        "height_p10": float(np.percentile(heights_array, 10)),
        "height_p90": float(np.percentile(heights_array, 90)),
        "success_rate": float(np.mean(heights_array > success_height)),
        "successful_landing_error_mean": (
            float(np.mean(successful_landing)) if successful_landing.size else None
        ),
        "successful_landing_error_median": (
            float(np.median(successful_landing)) if successful_landing.size else None
        ),
    }


def _protocol_mismatches(candidate: Dict[str, Any], reference: Dict[str, Any]) -> List[str]:
    mismatches = []
    for field in PROTOCOL_FIELDS:
        if field == "action_noise_scale":
            default = 0.0
        elif field == "fixed_physics":
            default = {
                "restitution": None,
                "joint_friction": None,
                "joint_damping": None,
            }
        elif field == "episode_sampling":
            default = "first-completed"
        elif field == "action_source":
            default = "policy"
        else:
            default = None
        if candidate.get(field, default) != reference.get(field, default):
            mismatches.append(
                "{}: candidate={!r}, reference={!r}".format(
                    field,
                    candidate.get(field, default),
                    reference.get(field, default),
                )
            )
    return mismatches


def compare_evaluations(
    candidate: Dict[str, Any],
    reference: Dict[str, Any],
    min_height_ratio: float = 0.95,
    max_success_gap: float = 0.05,
) -> Dict[str, Any]:
    if not 0.0 < min_height_ratio <= 1.0:
        raise ValueError("min_height_ratio must be in (0, 1]")
    if not 0.0 <= max_success_gap <= 1.0:
        raise ValueError("max_success_gap must be in [0, 1]")

    mismatches = _protocol_mismatches(candidate, reference)
    if mismatches:
        raise ValueError("evaluation protocol mismatch: " + "; ".join(mismatches))

    candidate_metrics = candidate["metrics"]
    reference_metrics = reference["metrics"]
    checks = []
    for metric_name in ("height_mean", "height_median"):
        reference_value = float(reference_metrics[metric_name])
        candidate_value = float(candidate_metrics[metric_name])
        if reference_value <= 0.0:
            raise ValueError("reference {} must be positive".format(metric_name))
        ratio = candidate_value / reference_value
        checks.append(
            {
                "name": metric_name + "_ratio",
                "candidate": candidate_value,
                "reference": reference_value,
                "required": min_height_ratio,
                "actual": ratio,
                "pass": ratio >= min_height_ratio,
            }
        )

    candidate_success = float(candidate_metrics["success_rate"])
    reference_success = float(reference_metrics["success_rate"])
    required_success = max(0.0, reference_success - max_success_gap)
    checks.append(
        {
            "name": "success_rate",
            "candidate": candidate_success,
            "reference": reference_success,
            "required": required_success,
            "actual": candidate_success,
            "pass": candidate_success >= required_success,
        }
    )

    return {
        "pass": all(check["pass"] for check in checks),
        "reference_task": reference.get("task"),
        "reference_run": reference.get("load_run"),
        "reference_checkpoint": reference.get("checkpoint"),
        "thresholds": {
            "min_height_ratio": min_height_ratio,
            "max_success_gap": max_success_gap,
        },
        "checks": checks,
    }
