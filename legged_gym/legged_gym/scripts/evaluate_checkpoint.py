"""Deterministic, reproducible checkpoint evaluator for jumping policies."""

from __future__ import annotations

import json
import os
import sys

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch.
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.checkpoint_evaluation import (
    PROTOCOL_VERSION,
    build_episode_quotas,
    compare_evaluations,
    summarize_episodes,
)


EVALUATOR_ARGS = [
    {
        "name": "--eval_mode",
        "type": str,
        "default": "robust",
        "choices": [
            "nominal",
            "robust",
            "observation",
            "latency",
            "actuation",
            "body",
            "base_mass",
            "com",
            "link_mass",
            "mass",
            "contact",
            "friction",
            "restitution",
            "joint_passive",
            "latency_contact",
            "latency_friction",
            "latency_restitution",
            "latency_joint_passive",
            "without_observation",
            "without_latency",
            "without_actuation",
            "without_body",
            "without_contact",
        ],
    },
    {"name": "--target_x", "type": float, "default": 0.0},
    {"name": "--target_y", "type": float, "default": 0.0},
    {"name": "--eval_episodes", "type": int, "default": 512},
    {"name": "--eval_max_steps", "type": int, "default": 2000},
    {
        "name": "--episode_sampling",
        "type": str,
        "default": "balanced-per-environment",
        "choices": ["balanced-per-environment", "first-completed"],
    },
    {"name": "--output_json", "type": str},
    {"name": "--trajectory_output_json", "type": str},
    {"name": "--trajectory_env_id", "type": int, "default": 0},
    {"name": "--reference_json", "type": str},
    {"name": "--min_height_ratio", "type": float, "default": 0.95},
    {"name": "--max_success_gap", "type": float, "default": 0.05},
    {"name": "--action_noise_scale", "type": float, "default": 0.0},
    {
        "name": "--action_source",
        "type": str,
        "default": "policy",
        "choices": ["policy", "zero"],
    },
    {"name": "--fixed_restitution", "type": float},
    {"name": "--fixed_joint_friction", "type": float},
    {"name": "--fixed_joint_damping", "type": float},
    {"name": "--fixed_latency_ms", "type": float},
    {"name": "--fixed_com_xyz", "type": str},
    {"name": "--com_range_half_width", "type": float},
    {"name": "--com_range_half_width_xyz", "type": str},
    {"name": "--latency_range_max_ms", "type": float},
    {"name": "--control_stiffness", "type": float},
    {"name": "--control_damping", "type": float},
    {"name": "--action_scale_override", "type": float},
    {"name": "--filter_freq_override", "type": float},
    {"name": "--base_height_override", "type": float},
    {"name": "--ignore_termination_contact_pattern", "type": str},
    {
        "name": "--terrain_mesh_type_override",
        "type": str,
        "choices": ["plane", "trimesh"],
    },
    {"name": "--sim_substeps_override", "type": int},
    {"name": "--physx_position_iterations_override", "type": int},
    {"name": "--physx_max_depenetration_velocity_override", "type": float},
    {
        "name": "--observation_noise_profile",
        "type": str,
        "default": "all",
        "choices": ["all", "contacts", "continuous"],
    },
    {"name": "--diagnostics", "action": "store_true", "default": False},
    {"name": "--landing_stability_seconds", "type": float, "default": 0.0},
    {"name": "--landing_contact_grace_seconds", "type": float, "default": 1.0},
    {
        "name": "--landing_min_all_feet_contact_ratio",
        "type": float,
        "default": 0.95,
    },
    {"name": "--landing_max_leg_torque_cv", "type": float, "default": 0.25},
    {"name": "--landing_max_linear_speed_rms", "type": float, "default": 0.1},
    {"name": "--landing_max_angular_speed_rms", "type": float, "default": 0.5},
    {
        "name": "--landing_max_orientation_error_rms",
        "type": float,
        "default": 0.2,
    },
]

UNCERTAINTY_GROUPS = {
    "latency": ("sim_latency", "sim_pd_latency", "randomize_lag_timesteps"),
    "actuation": (
        "randomize_spring_params",
        "randomize_motor_strength",
        "randomize_PD_gains",
        "randomize_motor_offset",
    ),
    "body": ("randomize_base_mass", "randomize_com", "randomize_link_mass"),
    "base_mass": ("randomize_base_mass",),
    "com": ("randomize_com",),
    "link_mass": ("randomize_link_mass",),
    "mass": ("randomize_base_mass", "randomize_link_mass"),
    "contact": (
        "randomize_friction",
        "randomize_restitution",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
    ),
    "friction": ("randomize_friction",),
    "restitution": ("randomize_restitution",),
    "joint_passive": (
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
    ),
    "latency_contact": (
        "sim_latency",
        "sim_pd_latency",
        "randomize_lag_timesteps",
        "randomize_friction",
        "randomize_restitution",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
    ),
    "latency_friction": (
        "sim_latency",
        "sim_pd_latency",
        "randomize_lag_timesteps",
        "randomize_friction",
    ),
    "latency_restitution": (
        "sim_latency",
        "sim_pd_latency",
        "randomize_lag_timesteps",
        "randomize_restitution",
    ),
    "latency_joint_passive": (
        "sim_latency",
        "sim_pd_latency",
        "randomize_lag_timesteps",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
    ),
}


def _set_if_present(config, name, value):
    if hasattr(config, name):
        setattr(config, name, value)


def _disable_initial_state_and_assistance(env_cfg):
    domain_rand = env_cfg.domain_rand
    for name in (
        "push_robots",
        "push_upwards",
        "randomize_robot_pos",
        "randomize_robot_vel",
        "randomize_robot_ori",
        "randomize_dof_pos",
        "randomize_has_jumped",
        "push_towards_goal",
    ):
        _set_if_present(domain_rand, name, False)
    _set_if_present(domain_rand, "pos_vel_random_prob", 0.0)
    _set_if_present(domain_rand, "curriculum", False)


def _disable_physical_and_sensor_uncertainty(env_cfg):
    env_cfg.noise.add_noise = False
    domain_rand = env_cfg.domain_rand
    for name in (
        "randomize_friction",
        "randomize_spring_params",
        "randomize_motor_strength",
        "randomize_PD_gains",
        "sim_latency",
        "sim_pd_latency",
        "randomize_lag_timesteps",
        "randomize_motor_offset",
        "randomize_base_mass",
        "randomize_com",
        "randomize_restitution",
        "randomize_link_mass",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
        "randomize_gravity",
    ):
        _set_if_present(domain_rand, name, False)


def configure_evaluation(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs or 128
    env_cfg.env.continuous_jumping = False
    env_cfg.env.continuous_jumping_reset_probability = 0.0
    env_cfg.env.debug_draw = False
    env_cfg.env.throttle_to_real_time = False
    env_cfg.terrain.curriculum = False

    # Diagnostic-only single-factor overrides. These are deliberately applied
    # to the in-memory evaluation config and never mutate a task/training file.
    if args.control_stiffness is not None:
        env_cfg.control.stiffness = {"joint": args.control_stiffness}
    if args.control_damping is not None:
        env_cfg.control.damping = {"joint": args.control_damping}
    if args.action_scale_override is not None:
        env_cfg.control.action_scale = args.action_scale_override
    if args.filter_freq_override is not None:
        env_cfg.control.filter_freq = args.filter_freq_override
    if args.base_height_override is not None:
        env_cfg.init_state.pos[2] = args.base_height_override
    if args.terrain_mesh_type_override is not None:
        env_cfg.terrain.mesh_type = args.terrain_mesh_type_override
    if args.sim_substeps_override is not None:
        if args.sim_substeps_override < 1:
            raise ValueError("--sim_substeps_override must be at least 1")
        env_cfg.sim.substeps = args.sim_substeps_override
    if args.physx_position_iterations_override is not None:
        if args.physx_position_iterations_override < 1:
            raise ValueError(
                "--physx_position_iterations_override must be at least 1"
            )
        env_cfg.sim.physx.num_position_iterations = (
            args.physx_position_iterations_override
        )
    if args.physx_max_depenetration_velocity_override is not None:
        if args.physx_max_depenetration_velocity_override <= 0.0:
            raise ValueError(
                "--physx_max_depenetration_velocity_override must be positive"
            )
        env_cfg.sim.physx.max_depenetration_velocity = (
            args.physx_max_depenetration_velocity_override
        )
    com_override_count = sum(
        value is not None
        for value in (
            args.fixed_com_xyz,
            args.com_range_half_width,
            args.com_range_half_width_xyz,
        )
    )
    if com_override_count > 1:
        raise ValueError(
            "Use only one of --fixed_com_xyz, --com_range_half_width, and "
            "--com_range_half_width_xyz"
        )
    if args.com_range_half_width_xyz is not None:
        try:
            half_widths = [
                float(part.strip())
                for part in args.com_range_half_width_xyz.split(",")
            ]
        except ValueError as exc:
            raise ValueError(
                "--com_range_half_width_xyz must be three comma-separated numbers"
            ) from exc
        if len(half_widths) != 3 or any(width < 0.0 for width in half_widths):
            raise ValueError(
                "--com_range_half_width_xyz must contain three non-negative values"
            )
        env_cfg.domain_rand.ranges.com_displacement_range = [
            [-width for width in half_widths],
            half_widths,
        ]
    elif args.com_range_half_width is not None:
        half_width = args.com_range_half_width
        if half_width < 0.0:
            raise ValueError("--com_range_half_width must be non-negative")
        env_cfg.domain_rand.ranges.com_displacement_range = [
            -half_width,
            half_width,
        ]
    if args.latency_range_max_ms is not None:
        latency_max_ms = args.latency_range_max_ms
        if latency_max_ms < 0.0:
            raise ValueError("--latency_range_max_ms must be non-negative")
        env_cfg.domain_rand.ranges.latency_range = [0.0, latency_max_ms]
    if args.ignore_termination_contact_pattern:
        pattern = args.ignore_termination_contact_pattern.lower()
        env_cfg.asset.terminate_after_contacts_on = [
            name
            for name in env_cfg.asset.terminate_after_contacts_on
            if pattern not in name.lower()
        ]
    noise_scales = env_cfg.noise.noise_scales
    if args.observation_noise_profile == "contacts":
        for name in (
            "lin_vel",
            "ang_vel",
            "dof_pos",
            "dof_vel",
            "height",
            "quat",
            "ori_error",
            "error_quat",
            "height_measurements",
        ):
            _set_if_present(noise_scales, name, 0.0)
    elif args.observation_noise_profile == "continuous":
        _set_if_present(noise_scales, "contacts_noise_prob", 0.0)

    commands = env_cfg.commands
    commands.curriculum = False
    commands.randomize_commands = False
    commands.randomize_yaw = False
    commands.upward_jump_probability = 0.0
    commands.distances.x = args.target_x
    commands.distances.y = args.target_y
    commands.distances.z = 0.0
    commands.ranges.pos_dx_ini = [args.target_x, args.target_x]
    commands.ranges.pos_dy_ini = [args.target_y, args.target_y]
    commands.ranges.pos_dz_ini = [0.0, 0.0]

    _disable_initial_state_and_assistance(env_cfg)
    configured_noise = env_cfg.noise.add_noise
    configured_flags = {
        name: getattr(env_cfg.domain_rand, name)
        for names in UNCERTAINTY_GROUPS.values()
        for name in names
        if hasattr(env_cfg.domain_rand, name)
    }
    if args.eval_mode.startswith("without_"):
        excluded_group = args.eval_mode[len("without_") :]
        if excluded_group == "observation":
            env_cfg.noise.add_noise = False
        else:
            for name in UNCERTAINTY_GROUPS[excluded_group]:
                _set_if_present(env_cfg.domain_rand, name, False)
    elif args.eval_mode != "robust":
        _disable_physical_and_sensor_uncertainty(env_cfg)
        if args.eval_mode == "observation":
            env_cfg.noise.add_noise = configured_noise
        elif args.eval_mode in UNCERTAINTY_GROUPS:
            for name in UNCERTAINTY_GROUPS[args.eval_mode]:
                if name in configured_flags:
                    setattr(env_cfg.domain_rand, name, configured_flags[name])

    ranges = env_cfg.domain_rand.ranges
    if args.fixed_restitution is not None:
        env_cfg.domain_rand.randomize_restitution = True
        ranges.restitution_range = [args.fixed_restitution, args.fixed_restitution]
    if args.fixed_joint_friction is not None:
        env_cfg.domain_rand.randomize_joint_friction = True
        ranges.joint_friction_range = [
            args.fixed_joint_friction,
            args.fixed_joint_friction,
        ]
    if args.fixed_joint_damping is not None:
        env_cfg.domain_rand.randomize_joint_damping = True
        ranges.joint_damping_range = [args.fixed_joint_damping, args.fixed_joint_damping]
    if args.fixed_latency_ms is not None:
        env_cfg.domain_rand.sim_latency = True
        _set_if_present(env_cfg.domain_rand, "zero_latency_probability", 0.0)
        _set_if_present(env_cfg.domain_rand, "max_latency_probability", 0.0)
        ranges.latency_range = [args.fixed_latency_ms, args.fixed_latency_ms]
        ranges.additional_latency_range = [0.0, 0.0]
    if args.fixed_com_xyz is not None:
        try:
            fixed_com = [
                float(part.strip()) for part in args.fixed_com_xyz.split(",")
            ]
        except ValueError as exc:
            raise ValueError(
                "--fixed_com_xyz must be three comma-separated numbers"
            ) from exc
        if len(fixed_com) != 3:
            raise ValueError(
                "--fixed_com_xyz must contain exactly three values"
            )
        env_cfg.domain_rand.randomize_com = True
        ranges.com_displacement_range = [fixed_com, fixed_com]

    if args.seed is None:
        args.seed = 1
    env_cfg.seed = args.seed
    return env_cfg


def _snapshot_episode_parameters(env):
    """Copy per-environment parameters before reset resamples them."""

    scalar_sources = {
        "payload_mass": "payload_masses",
        "friction": "friction_coeffs",
        "restitution": "restitutions",
        "joint_friction": "joint_friction_coeffs",
        "joint_damping": "joint_damping_coeffs",
        "latency": "episodic_latency",
    }
    vector_sources = {
        "com_x": ("com_displacements", 0),
        "com_y": ("com_displacements", 1),
        "com_z": ("com_displacements", 2),
    }
    mean_sources = {
        "link_mass_factor_mean": "link_masses",
        "motor_strength_mean": "motor_strengths",
        "p_gain_mean": "p_gains",
        "d_gain_mean": "d_gains",
        "motor_offset_abs_mean": "motor_offsets",
    }

    snapshot = {}
    for name, source in scalar_sources.items():
        if hasattr(env, source):
            snapshot[name] = getattr(env, source).reshape(env.num_envs, -1).mean(1).clone()
    for name, (source, index) in vector_sources.items():
        if hasattr(env, source):
            snapshot[name] = getattr(env, source)[:, index].clone()
    for name, source in mean_sources.items():
        if hasattr(env, source):
            values = getattr(env, source).reshape(env.num_envs, -1)
            if name == "motor_offset_abs_mean":
                values = torch.abs(values)
            snapshot[name] = values.mean(1).clone()
    if hasattr(env, "link_masses"):
        link_mass_factors = env.link_masses.reshape(env.num_envs, -1)
        snapshot["link_mass_factor_std"] = link_mass_factors.std(
            dim=1, unbiased=False
        )
        for link_index, body_name in enumerate(env.body_names[1:]):
            snapshot["link_mass_factor_{}".format(body_name)] = (
                link_mass_factors[:, link_index].clone()
            )
    return snapshot


def _audit_physics_randomization(env, max_envs=64, tolerance=1e-5):
    """Round-trip sampled parameters through the live PhysX actors."""

    sample_count = min(env.num_envs, max_envs)
    sampled_env_ids = np.linspace(
        0, env.num_envs - 1, num=sample_count, dtype=np.int64
    )
    max_errors = {
        "shape_friction": 0.0,
        "shape_restitution": 0.0,
        "joint_friction": 0.0,
        "joint_damping": 0.0,
        "joint_armature": 0.0,
        "body_mass": 0.0,
        "base_com": 0.0,
    }
    shape_counts = []
    body_counts = []
    nominal_masses = env.nominal_body_masses
    nominal_com = env.nominal_base_com

    for env_id in sampled_env_ids:
        env_handle = env.envs[int(env_id)]
        actor_handle = env.actor_handles[int(env_id)]
        shape_props = env.gym.get_actor_rigid_shape_properties(
            env_handle, actor_handle
        )
        shape_counts.append(len(shape_props))
        expected_friction = float(env.friction_coeffs[env_id, 0].item())
        expected_restitution = float(env.restitutions[env_id, 0].item())
        for prop in shape_props:
            max_errors["shape_friction"] = max(
                max_errors["shape_friction"],
                abs(float(prop.friction) - expected_friction),
            )
            max_errors["shape_restitution"] = max(
                max_errors["shape_restitution"],
                abs(float(prop.restitution) - expected_restitution),
            )

        dof_props = env.gym.get_actor_dof_properties(env_handle, actor_handle)
        expected_joint_friction = float(
            env.joint_friction_coeffs[env_id, 0].item()
        )
        expected_joint_damping = float(
            env.joint_damping_coeffs[env_id, 0].item()
        )
        expected_joint_armature = float(env.joint_armatures[env_id, 0].item())
        max_errors["joint_friction"] = max(
            max_errors["joint_friction"],
            float(np.max(np.abs(dof_props["friction"] - expected_joint_friction))),
        )
        max_errors["joint_damping"] = max(
            max_errors["joint_damping"],
            float(np.max(np.abs(dof_props["damping"] - expected_joint_damping))),
        )
        max_errors["joint_armature"] = max(
            max_errors["joint_armature"],
            float(np.max(np.abs(dof_props["armature"] - expected_joint_armature))),
        )

        body_props = env.gym.get_actor_rigid_body_properties(
            env_handle, actor_handle
        )
        body_counts.append(len(body_props))
        for body_index, prop in enumerate(body_props):
            if body_index == 0:
                expected_mass = nominal_masses[0]
                if env.cfg.domain_rand.randomize_base_mass:
                    expected_mass += float(env.payload_masses[env_id].item())
            else:
                expected_mass = nominal_masses[body_index]
                if env.cfg.domain_rand.randomize_link_mass:
                    expected_mass *= float(
                        env.link_masses[env_id, body_index - 1].item()
                    )
            max_errors["body_mass"] = max(
                max_errors["body_mass"], abs(float(prop.mass) - expected_mass)
            )
        com_displacement = (
            tuple(float(value.item()) for value in env.com_displacements[env_id])
            if env.cfg.domain_rand.randomize_com
            else (0.0, 0.0, 0.0)
        )
        expected_com = tuple(
            nominal + displacement
            for nominal, displacement in zip(nominal_com, com_displacement)
        )
        actual_com = (
            float(body_props[0].com.x),
            float(body_props[0].com.y),
            float(body_props[0].com.z),
        )
        max_errors["base_com"] = max(
            max_errors["base_com"],
            max(abs(actual - expected) for actual, expected in zip(actual_com, expected_com)),
        )

    passed = all(error <= tolerance for error in max_errors.values())
    return {
        "pass": passed,
        "sampled_env_count": sample_count,
        "sampled_env_ids": sampled_env_ids.tolist(),
        "shape_count_min": min(shape_counts),
        "shape_count_max": max(shape_counts),
        "body_count_min": min(body_counts),
        "body_count_max": max(body_counts),
        "tolerance": tolerance,
        "max_absolute_errors": max_errors,
    }


def _summarize_diagnostics(records, success_height):
    if not records:
        return {}
    heights = np.asarray([record["height"] for record in records], dtype=float)
    success = heights > success_height
    fields = sorted(set().union(*(record.keys() for record in records)) - {"height"})
    summary = {
        "count": len(records),
        "successful_count": int(np.count_nonzero(success)),
        "failed_count": int(np.count_nonzero(~success)),
        "fields": {},
    }
    for field in fields:
        values = np.asarray([record.get(field, np.nan) for record in records], dtype=float)
        finite = np.isfinite(values)
        successful_values = values[finite & success]
        failed_values = values[finite & ~success]
        all_values = values[finite]
        if not all_values.size:
            continue
        standard_deviation = float(np.std(all_values))
        successful_mean = (
            float(np.mean(successful_values)) if successful_values.size else None
        )
        failed_mean = float(np.mean(failed_values)) if failed_values.size else None
        standardized_difference = None
        if successful_mean is not None and failed_mean is not None and standard_deviation > 0.0:
            standardized_difference = (successful_mean - failed_mean) / standard_deviation
        correlation = None
        if standard_deviation > 0.0 and np.unique(success[finite]).size > 1:
            correlation = float(np.corrcoef(all_values, success[finite].astype(float))[0, 1])
        summary["fields"][field] = {
            "mean": float(np.mean(all_values)),
            "median": float(np.median(all_values)),
            "successful_mean": successful_mean,
            "failed_mean": failed_mean,
            "standardized_difference": standardized_difference,
            "success_correlation": correlation,
        }
    return summary


def _summarize_jump_detection(records):
    """Compare the task state machine with contact/kinematics-only take-off."""

    if not records:
        return {}
    physical_takeoff = np.asarray(
        [
            record["physical_max_all_feet_off_steps"] >= 2
            and record["physical_takeoff_max_vz"] > 0.0
            for record in records
        ],
        dtype=bool,
    )
    event_takeoff = np.asarray(
        [record["first_air_step"] >= 0 for record in records], dtype=bool
    )
    event_landing = np.asarray(
        [record["event_landing_observed"] > 0.5 for record in records],
        dtype=bool,
    )
    return {
        "physical_takeoff_rate": float(np.mean(physical_takeoff)),
        "event_takeoff_rate": float(np.mean(event_takeoff)),
        "event_landing_rate": float(np.mean(event_landing)),
        "takeoff_false_negative_rate": float(
            np.mean(physical_takeoff & ~event_takeoff)
        ),
        "takeoff_false_positive_rate": float(
            np.mean(~physical_takeoff & event_takeoff)
        ),
        "agreement_rate": float(np.mean(physical_takeoff == event_takeoff)),
    }


def evaluate(args):
    if args.load_run is None or args.checkpoint is None:
        raise ValueError("--load_run and --checkpoint are required")
    if args.eval_episodes <= 0 or args.eval_max_steps <= 0:
        raise ValueError("--eval_episodes and --eval_max_steps must be positive")
    if args.action_noise_scale < 0.0:
        raise ValueError("--action_noise_scale must be non-negative")
    for name in (
        "control_stiffness",
        "control_damping",
        "action_scale_override",
        "filter_freq_override",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise ValueError("--{} must be positive".format(name))
    if args.base_height_override is not None and args.base_height_override <= 0.0:
        raise ValueError("--base_height_override must be positive")
    if args.landing_stability_seconds < 0.0:
        raise ValueError("--landing_stability_seconds must be non-negative")
    if args.landing_contact_grace_seconds < 0.0:
        raise ValueError("--landing_contact_grace_seconds must be non-negative")
    if not 0.0 <= args.landing_min_all_feet_contact_ratio <= 1.0:
        raise ValueError("--landing_min_all_feet_contact_ratio must be in [0, 1]")
    if args.landing_max_leg_torque_cv < 0.0:
        raise ValueError("--landing_max_leg_torque_cv must be non-negative")
    if args.landing_max_linear_speed_rms < 0.0:
        raise ValueError("--landing_max_linear_speed_rms must be non-negative")
    if args.landing_max_angular_speed_rms < 0.0:
        raise ValueError("--landing_max_angular_speed_rms must be non-negative")
    if args.landing_max_orientation_error_rms < 0.0:
        raise ValueError("--landing_max_orientation_error_rms must be non-negative")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = configure_evaluation(env_cfg, args)
    configured_episode_length_s = env_cfg.env.episode_length_s
    if args.landing_stability_seconds:
        # Leave the normal jump window untouched, then keep the policy alive
        # long enough to observe a full post-landing standing interval. The
        # grace window excludes the impact/bounce transient from contact and
        # torque stability statistics.
        env_cfg.env.episode_length_s += (
            args.landing_stability_seconds
            + args.landing_contact_grace_seconds
            + 0.1
        )
    train_cfg.seed = args.seed
    train_cfg.runner.resume = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    physics_audit = None
    if args.diagnostics:
        physics_audit = _audit_physics_randomization(env)
        if not physics_audit["pass"]:
            raise RuntimeError(
                "sampled domain-randomization buffers do not match live PhysX "
                "actor properties: {}".format(
                    physics_audit["max_absolute_errors"]
                )
            )
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    # BaseTask.reset(), used during construction, advances one zero-action
    # physics step after reset_idx(). Later training episodes start directly
    # from reset_idx() inside post_physics_step(). Normalize evaluation to the
    # latter, recurrent path so the first batch is not artificially shifted by
    # one control step relative to every following episode. Rebuild stored and
    # delayed state too: compute_observations() consumes the delayed tensors,
    # and reset_idx() alone intentionally leaves their refresh to
    # post_physics_step().
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(all_env_ids)
    # Match the recurrent training-reset path exactly. reset_idx() clears the
    # flattened state histories, while post_physics_step() separately fills
    # every observation-delay storage slot from the new physical state. If
    # this fill is omitted, non-zero-latency evaluation reads stale/zero slots
    # on the first decision and tests an observation that training never sees.
    env._reset_stored_states(all_env_ids)
    # post_physics_step() also snapshots the newly reset root velocity at the
    # end of a recurrent training reset. Direct reset_idx() leaves it at zero,
    # which turns the first velocity sample into a spurious acceleration and
    # can trip the two-sample impact termination before the policy acts.
    env.last_root_vel[all_env_ids] = env.root_states[all_env_ids, 7:13]
    env._compute_state_history()
    env._store_states()
    latency_indices, sampled_latency = env._sample_latency()
    env._get_delayed_states(latency_indices, sampled_latency)
    env.compute_observations()
    obs = env.get_observations()
    actor_critic = runner.alg.actor_critic
    actor_critic.eval()
    actor_critic.to(env.device)
    checkpoint_std = actor_critic.std.detach().clone()
    action_generator = torch.Generator(device=env.device)
    action_generator.manual_seed(args.seed + 1000003)

    peaks = env.root_states[:, 2].clone()
    completed_heights = []
    completed_landing_errors = []
    completed_terminations = []
    completed_landing_observed = []
    completed_post_landing_seconds = []
    completed_all_feet_contact_ratios = []
    completed_final_all_feet_contacts = []
    completed_leg_torque_cvs = []
    completed_final_linear_speeds = []
    completed_final_angular_speeds = []
    completed_final_orientation_errors = []
    completed_linear_speed_rms = []
    completed_angular_speed_rms = []
    completed_orientation_error_rms = []
    completed_stable_standing = []
    diagnostic_records = []
    trajectory_records = []
    episode_quotas = torch.as_tensor(
        build_episode_quotas(env.num_envs, args.eval_episodes),
        dtype=torch.long,
        device=env.device,
    )
    completed_per_env = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    landing_observed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    landing_step = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    post_landing_samples = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    post_landing_all_feet_contacts = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    post_landing_torque_sum = torch.zeros(env.num_envs, device=env.device)
    post_landing_torque_sq_sum = torch.zeros(env.num_envs, device=env.device)
    post_landing_linear_speed_sq_sum = torch.zeros(env.num_envs, device=env.device)
    post_landing_angular_speed_sq_sum = torch.zeros(env.num_envs, device=env.device)
    post_landing_orientation_error_sq_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    final_all_feet_contact = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    final_linear_speed = torch.full((env.num_envs,), torch.nan, device=env.device)
    final_angular_speed = torch.full((env.num_envs,), torch.nan, device=env.device)
    final_orientation_error = torch.full(
        (env.num_envs,), torch.nan, device=env.device
    )
    landing_grace_steps = int(
        np.ceil(args.landing_contact_grace_seconds / env.dt)
    )
    required_stability_steps = int(
        np.ceil(args.landing_stability_seconds / env.dt)
    )
    if args.diagnostics:
        episode_generation = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        episode_parameters = _snapshot_episode_parameters(env)
        diagnostic_steps = tuple(range(25))
        thigh_indices = torch.as_tensor(
            [index for index, name in enumerate(env.dof_names) if "thigh" in name],
            dtype=torch.long,
            device=env.device,
        )
        calf_indices = torch.as_tensor(
            [index for index, name in enumerate(env.dof_names) if "calf" in name],
            dtype=torch.long,
            device=env.device,
        )
        landing_joint_indices = {
            "{}_{}".format(leg, joint): next(
                index
                for index, name in enumerate(env.dof_names)
                if leg in name and joint in name
            )
            for leg in ("FL", "FR", "RL", "RR")
            for joint in ("thigh", "calf")
        }
        landing_state_values = {
            name: torch.full((env.num_envs,), torch.nan, device=env.device)
            for name in (
                "landing_height",
                "landing_vz",
                "landing_projected_gravity_x",
                "landing_projected_gravity_y",
                "landing_action_abs_mean",
                *("landing_{}_pos".format(name) for name in landing_joint_indices),
                *("landing_foot_{}_contact".format(leg) for leg in ("FL", "FR", "RL", "RR")),
            )
        }
        episode_step_values = {
            "{}_step_{}".format(signal, step): torch.full(
                (env.num_envs,), torch.nan, device=env.device
            )
            for step in diagnostic_steps
            for signal in (
                "height",
                "vz",
                "action_abs_mean",
                "torque_ratio",
                "thigh_pos_mean",
                "calf_pos_mean",
                "thigh_action_mean",
                "calf_action_mean",
            )
        }
        episode_min_height = env.root_states[:, 2].clone()
        episode_max_vz = env.root_states[:, 9].clone()
        episode_max_action_abs_mean = torch.zeros(env.num_envs, device=env.device)
        episode_max_torque_ratio = torch.zeros(env.num_envs, device=env.device)
        episode_first_air_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        episode_takeoff_height = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_vz = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_action_abs_mean = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_dof_vel_abs_mean = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_torque_abs_mean = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        physical_contact_observed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        physical_first_all_feet_contact_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        physical_first_takeoff_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
        physical_first_contact_height = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        physical_first_contact_vz = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        physical_first_contact_action_abs_mean = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        physical_current_all_feet_off_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        physical_max_all_feet_off_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        physical_takeoff_max_height = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        physical_takeoff_max_vz = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        event_landing_observed = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        # Capture the observation/control state at the first policy decision
        # of every episode. The initial env.reset() performs a zero-action
        # physics step, whereas resets inside post_physics_step() do not; these
        # values make any first-episode/reset-generation mismatch explicit.
        episode_initial_captured = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        episode_initial_values = {
            name: torch.full((env.num_envs,), torch.nan, device=env.device)
            for name in (
                "initial_episode_length",
                "initial_height",
                "initial_vz",
                "initial_contact_fraction",
                "initial_obs_abs_mean",
                "initial_action_abs_mean",
                "initial_thigh_action_mean",
                "initial_calf_action_mean",
                "initial_filter_abs_mean",
                "initial_episodic_latency_ms",
                "initial_sampled_latency_ms",
            )
        }
        episode_sampled_latency_sum_ms = torch.zeros(
            env.num_envs, device=env.device
        )
        episode_sampled_latency_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )
        episode_sampled_latency_max_ms = torch.zeros(
            env.num_envs, device=env.device
        )
    torque_saturated = 0
    torque_samples = 0
    action_abs_sum = 0.0
    action_samples = 0

    with torch.inference_mode():
        for evaluation_step in range(args.eval_max_steps):
            actions = actor_critic.act_inference(obs.detach())
            if args.action_source == "zero":
                actions = torch.zeros_like(actions)
            if args.action_noise_scale:
                actions = actions + args.action_noise_scale * checkpoint_std * torch.randn(
                    actions.shape,
                    dtype=actions.dtype,
                    device=actions.device,
                    generator=action_generator,
                )
            if args.trajectory_output_json:
                trace_env_id = args.trajectory_env_id
                if not 0 <= trace_env_id < env.num_envs:
                    raise ValueError(
                        "--trajectory_env_id must identify an active environment"
                    )
                if completed_per_env[trace_env_id] == 0:
                    dof_pos_policy = env._dof_to_policy(env.dof_pos)[trace_env_id]
                    dof_vel_policy = env._dof_to_policy(env.dof_vel)[trace_env_id]
                    foot_positions = env.feet_pos[trace_env_id]
                    calf_body_indices = [
                        index
                        for index, name in enumerate(env.body_names)
                        if "calf" in name.lower()
                    ]
                    trajectory_records.append(
                        {
                            "evaluation_step": evaluation_step,
                            "episode_step": int(
                                env.episode_length_buf[trace_env_id].item()
                            ),
                            "time_s": float(
                                env.episode_length_buf[trace_env_id].item()
                                * env.dt
                            ),
                            "height": float(
                                env.root_states[trace_env_id, 2].item()
                            ),
                            "linear_velocity": env.root_states[
                                trace_env_id, 7:10
                            ].cpu().tolist(),
                            "angular_velocity": env.root_states[
                                trace_env_id, 10:13
                            ].cpu().tolist(),
                            "euler_rpy": env.euler[trace_env_id].cpu().tolist(),
                            "projected_gravity": env.projected_gravity[
                                trace_env_id
                            ].cpu().tolist(),
                            "settled_after_init": bool(
                                env.settled_after_init[trace_env_id].item()
                            ),
                            "mid_air": bool(env.mid_air[trace_env_id].item()),
                            "was_in_flight": bool(
                                env.was_in_flight[trace_env_id].item()
                            ),
                            "has_jumped": bool(
                                env.has_jumped[trace_env_id].item()
                            ),
                            "foot_contacts": env.contacts[
                                trace_env_id
                            ].cpu().tolist(),
                            "foot_positions_world": foot_positions.cpu().tolist(),
                            "calf_contact_force_norms": torch.linalg.norm(
                                env.contact_forces[
                                    trace_env_id, calf_body_indices, :
                                ],
                                dim=-1,
                            ).cpu().tolist(),
                            "dof_pos_policy": dof_pos_policy.cpu().tolist(),
                            "dof_vel_policy": dof_vel_policy.cpu().tolist(),
                            "actions_policy": actions[trace_env_id].cpu().tolist(),
                        }
                    )
            if args.diagnostics:
                first_decision = ~episode_initial_captured
                if torch.any(first_decision):
                    episode_initial_values["initial_episode_length"][first_decision] = (
                        env.episode_length_buf[first_decision].float()
                    )
                    episode_initial_values["initial_height"][first_decision] = (
                        env.root_states[first_decision, 2]
                    )
                    episode_initial_values["initial_vz"][first_decision] = (
                        env.root_states[first_decision, 9]
                    )
                    episode_initial_values["initial_contact_fraction"][first_decision] = (
                        env.contacts[first_decision].float().mean(1)
                    )
                    episode_initial_values["initial_obs_abs_mean"][first_decision] = (
                        torch.abs(obs[first_decision]).mean(1)
                    )
                    episode_initial_values["initial_action_abs_mean"][first_decision] = (
                        torch.abs(actions[first_decision]).mean(1)
                    )
                    episode_initial_values["initial_thigh_action_mean"][first_decision] = (
                        actions[first_decision][:, thigh_indices].mean(1)
                    )
                    episode_initial_values["initial_calf_action_mean"][first_decision] = (
                        actions[first_decision][:, calf_indices].mean(1)
                    )
                    if hasattr(env.action_filter, "filtered_values"):
                        episode_initial_values["initial_filter_abs_mean"][first_decision] = (
                            torch.abs(env.action_filter.filtered_values[first_decision]).mean(1)
                        )
                    episode_initial_values["initial_episodic_latency_ms"][first_decision] = (
                        1000.0 * env.episodic_latency[first_decision]
                    )
                    episode_initial_values["initial_sampled_latency_ms"][first_decision] = (
                        1000.0 * env.sampled_observation_latency[first_decision]
                    )
                    episode_initial_captured[first_decision] = True
                episode_max_action_abs_mean = torch.maximum(
                    episode_max_action_abs_mean, torch.abs(actions).mean(1)
                )
            action_abs_sum += torch.abs(actions).sum().item()
            action_samples += actions.numel()

            obs, _, _, dones, _ = env.step(actions.detach())

            torque_limits = env.torque_limits.view(1, -1)
            if args.diagnostics:
                torque_ratio = torch.abs(env.torques) / torque_limits
                episode_max_torque_ratio = torch.maximum(
                    episode_max_torque_ratio, torque_ratio.max(1).values
                )
            torque_saturated += torch.count_nonzero(
                torch.abs(env.torques) >= 0.98 * torque_limits
            ).item()
            torque_samples += env.torques.numel()

            done_mask = dones.bool()
            active_mask = ~done_mask
            peaks[active_mask] = torch.maximum(
                peaks[active_mask], env.root_states[active_mask, 2]
            )
            just_landed = active_mask & env.has_jumped & ~landing_observed
            landing_observed[just_landed] = True
            landing_step[just_landed] = evaluation_step
            if args.diagnostics and torch.any(just_landed):
                landing_state_values["landing_height"][just_landed] = (
                    env.root_states[just_landed, 2]
                )
                landing_state_values["landing_vz"][just_landed] = (
                    env.root_states[just_landed, 9]
                )
                landing_state_values["landing_projected_gravity_x"][just_landed] = (
                    env.projected_gravity[just_landed, 0]
                )
                landing_state_values["landing_projected_gravity_y"][just_landed] = (
                    env.projected_gravity[just_landed, 1]
                )
                landing_state_values["landing_action_abs_mean"][just_landed] = (
                    torch.abs(actions[just_landed]).mean(1)
                )
                for name, joint_index in landing_joint_indices.items():
                    landing_state_values[
                        "landing_{}_pos".format(name)
                    ][just_landed] = env.dof_pos[just_landed, joint_index]
                for foot_index, leg in enumerate(("FL", "FR", "RL", "RR")):
                    landing_state_values[
                        "landing_foot_{}_contact".format(leg)
                    ][just_landed] = env.contacts[just_landed, foot_index].float()
            post_landing_age = evaluation_step - landing_step
            standing_sample = (
                active_mask
                & landing_observed
                & (post_landing_age >= landing_grace_steps)
            )
            if torch.any(standing_sample):
                all_feet_contact = torch.all(env.contacts, dim=1)
                mean_abs_leg_torque = torch.abs(env.torques).mean(1)
                linear_speed = torch.linalg.norm(env.base_lin_vel, dim=1)
                angular_speed = torch.linalg.norm(env.base_ang_vel, dim=1)
                orientation_error = env.ori_error.reshape(-1)
                post_landing_samples[standing_sample] += 1
                post_landing_all_feet_contacts[standing_sample] += (
                    all_feet_contact[standing_sample].long()
                )
                post_landing_torque_sum[standing_sample] += (
                    mean_abs_leg_torque[standing_sample]
                )
                post_landing_torque_sq_sum[standing_sample] += (
                    mean_abs_leg_torque[standing_sample] ** 2
                )
                post_landing_linear_speed_sq_sum[standing_sample] += (
                    linear_speed[standing_sample] ** 2
                )
                post_landing_angular_speed_sq_sum[standing_sample] += (
                    angular_speed[standing_sample] ** 2
                )
                post_landing_orientation_error_sq_sum[standing_sample] += (
                    orientation_error[standing_sample] ** 2
                )
                final_all_feet_contact[standing_sample] = all_feet_contact[
                    standing_sample
                ]
                final_linear_speed[standing_sample] = torch.linalg.norm(
                    env.base_lin_vel[standing_sample], dim=1
                )
                final_angular_speed[standing_sample] = torch.linalg.norm(
                    env.base_ang_vel[standing_sample], dim=1
                )
                final_orientation_error[standing_sample] = env.ori_error[
                    standing_sample
                ].reshape(-1)
            if args.diagnostics:
                sampled_latency_ms = 1000.0 * env.sampled_observation_latency
                episode_sampled_latency_sum_ms[active_mask] += sampled_latency_ms[
                    active_mask
                ]
                episode_sampled_latency_count[active_mask] += 1
                episode_sampled_latency_max_ms[active_mask] = torch.maximum(
                    episode_sampled_latency_max_ms[active_mask],
                    sampled_latency_ms[active_mask],
                )
                episode_min_height[active_mask] = torch.minimum(
                    episode_min_height[active_mask], env.root_states[active_mask, 2]
                )
                episode_max_vz[active_mask] = torch.maximum(
                    episode_max_vz[active_mask], env.root_states[active_mask, 9]
                )
                all_feet_contact = torch.all(env.contacts, dim=1)
                first_physical_contact = (
                    active_mask & all_feet_contact & ~physical_contact_observed
                )
                physical_contact_observed[first_physical_contact] = True
                physical_first_all_feet_contact_step[first_physical_contact] = (
                    env.episode_length_buf[first_physical_contact]
                )
                physical_first_contact_height[first_physical_contact] = (
                    env.root_states[first_physical_contact, 2]
                )
                physical_first_contact_vz[first_physical_contact] = (
                    env.root_states[first_physical_contact, 9]
                )
                physical_first_contact_action_abs_mean[first_physical_contact] = (
                    torch.abs(actions[first_physical_contact]).mean(1)
                )
                all_feet_off_after_contact = (
                    active_mask
                    & physical_contact_observed
                    & torch.all(~env.contacts, dim=1)
                )
                physical_current_all_feet_off_steps[all_feet_off_after_contact] += 1
                physical_current_all_feet_off_steps[
                    active_mask & ~all_feet_off_after_contact
                ] = 0
                physical_max_all_feet_off_steps = torch.maximum(
                    physical_max_all_feet_off_steps,
                    physical_current_all_feet_off_steps,
                )
                first_physical_takeoff = (
                    all_feet_off_after_contact
                    & (physical_first_takeoff_step < 0)
                )
                physical_first_takeoff_step[first_physical_takeoff] = (
                    env.episode_length_buf[first_physical_takeoff]
                )
                previous_takeoff_height = physical_takeoff_max_height[
                    all_feet_off_after_contact
                ]
                current_takeoff_height = env.root_states[
                    all_feet_off_after_contact, 2
                ]
                physical_takeoff_max_height[all_feet_off_after_contact] = torch.where(
                    torch.isnan(previous_takeoff_height),
                    current_takeoff_height,
                    torch.maximum(previous_takeoff_height, current_takeoff_height),
                )
                previous_takeoff_vz = physical_takeoff_max_vz[
                    all_feet_off_after_contact
                ]
                current_takeoff_vz = env.root_states[
                    all_feet_off_after_contact, 9
                ]
                physical_takeoff_max_vz[all_feet_off_after_contact] = torch.where(
                    torch.isnan(previous_takeoff_vz),
                    current_takeoff_vz,
                    torch.maximum(previous_takeoff_vz, current_takeoff_vz),
                )
                event_landing_observed[active_mask] |= env.has_jumped[active_mask]
                relative_step = (
                    env.episode_length_buf - env.settled_after_init_timer
                )
                for diagnostic_step in diagnostic_steps:
                    step_mask = (
                        active_mask
                        & env.settled_after_init
                        & (relative_step == diagnostic_step)
                    )
                    if not torch.any(step_mask):
                        continue
                    episode_step_values[
                        "height_step_{}".format(diagnostic_step)
                    ][step_mask] = env.root_states[step_mask, 2]
                    episode_step_values[
                        "vz_step_{}".format(diagnostic_step)
                    ][step_mask] = env.root_states[step_mask, 9]
                    episode_step_values[
                        "action_abs_mean_step_{}".format(diagnostic_step)
                    ][step_mask] = torch.abs(actions[step_mask]).mean(1)
                    episode_step_values[
                        "torque_ratio_step_{}".format(diagnostic_step)
                    ][step_mask] = torque_ratio[step_mask].max(1).values
                    episode_step_values[
                        "thigh_pos_mean_step_{}".format(diagnostic_step)
                    ][step_mask] = env.dof_pos[step_mask][:, thigh_indices].mean(1)
                    episode_step_values[
                        "calf_pos_mean_step_{}".format(diagnostic_step)
                    ][step_mask] = env.dof_pos[step_mask][:, calf_indices].mean(1)
                    episode_step_values[
                        "thigh_action_mean_step_{}".format(diagnostic_step)
                    ][step_mask] = actions[step_mask][:, thigh_indices].mean(1)
                    episode_step_values[
                        "calf_action_mean_step_{}".format(diagnostic_step)
                    ][step_mask] = actions[step_mask][:, calf_indices].mean(1)
                # Exclude the intentional airborne initialization. A real
                # take-off occurs only after the robot has first settled and
                # then breaks all foot contacts while moving upward.
                first_air = (
                    active_mask
                    & env.settled_after_init
                    & env.mid_air
                    & (env.root_states[:, 9] > 0.0)
                    & (episode_first_air_step < 0)
                )
                if torch.any(first_air):
                    episode_first_air_step[first_air] = (
                        env.episode_length_buf[first_air]
                        - env.settled_after_init_timer[first_air]
                    )
                    episode_takeoff_height[first_air] = env.root_states[first_air, 2]
                    episode_takeoff_vz[first_air] = env.root_states[first_air, 9]
                    episode_takeoff_action_abs_mean[first_air] = torch.abs(actions[first_air]).mean(1)
                    episode_takeoff_dof_vel_abs_mean[first_air] = torch.abs(
                        env.dof_vel[first_air]
                    ).mean(1)
                    episode_takeoff_torque_abs_mean[first_air] = torch.abs(
                        env.torques[first_air]
                    ).mean(1)

            # Record a balanced, deterministic quota per environment instead
            # of the globally first completed episodes.  The latter
            # over-counted rapidly terminating environments and biased robust
            # results toward failure, especially when eval_episodes was less
            # than num_envs.
            if args.episode_sampling == "balanced-per-environment":
                recordable_done = done_mask & (completed_per_env < episode_quotas)
                done_ids = recordable_done.nonzero(as_tuple=False).flatten()
            else:
                done_ids = done_mask.nonzero(as_tuple=False).flatten()
                remaining = args.eval_episodes - len(completed_heights)
                done_ids = done_ids[:remaining]
            if done_ids.numel():
                done_peaks = peaks[done_ids]
                done_errors = env.tracking_error_store[done_ids]
                done_success = done_peaks > env_cfg.rewards.jump_success_height

                completed_heights.extend(done_peaks.cpu().tolist())
                for successful, error in zip(
                    done_success.cpu().tolist(), done_errors.cpu().tolist()
                ):
                    completed_landing_errors.append(
                        float(error) if successful and error > 0.0 else float("nan")
                    )
                completed_terminations.extend(
                    (~env.time_out_buf[done_ids]).float().cpu().tolist()
                )
                sample_counts = post_landing_samples[done_ids]
                observed = landing_observed[done_ids]
                contact_ratios = torch.where(
                    sample_counts > 0,
                    post_landing_all_feet_contacts[done_ids].float()
                    / torch.clamp(sample_counts.float(), min=1.0),
                    torch.zeros_like(sample_counts, dtype=torch.float),
                )
                torque_means = torch.where(
                    sample_counts > 0,
                    post_landing_torque_sum[done_ids]
                    / torch.clamp(sample_counts.float(), min=1.0),
                    torch.zeros_like(sample_counts, dtype=torch.float),
                )
                torque_variances = torch.where(
                    sample_counts > 0,
                    post_landing_torque_sq_sum[done_ids]
                    / torch.clamp(sample_counts.float(), min=1.0)
                    - torque_means**2,
                    torch.zeros_like(sample_counts, dtype=torch.float),
                )
                torque_cvs = torch.where(
                    sample_counts > 0,
                    torch.sqrt(torch.clamp(torque_variances, min=0.0))
                    / torch.clamp(torque_means, min=1e-6),
                    torch.full_like(torque_means, torch.nan),
                )
                sample_denominator = torch.clamp(sample_counts.float(), min=1.0)
                missing_sample = torch.full_like(sample_denominator, torch.nan)
                linear_speed_rms = torch.where(
                    sample_counts > 0,
                    torch.sqrt(
                        post_landing_linear_speed_sq_sum[done_ids]
                        / sample_denominator
                    ),
                    missing_sample,
                )
                angular_speed_rms = torch.where(
                    sample_counts > 0,
                    torch.sqrt(
                        post_landing_angular_speed_sq_sum[done_ids]
                        / sample_denominator
                    ),
                    missing_sample,
                )
                orientation_error_rms = torch.where(
                    sample_counts > 0,
                    torch.sqrt(
                        post_landing_orientation_error_sq_sum[done_ids]
                        / sample_denominator
                    ),
                    missing_sample,
                )
                natural_completion = env.time_out_buf[done_ids]
                stable_standing = (
                    done_success
                    & natural_completion
                    & observed
                    & (sample_counts >= required_stability_steps)
                    & (
                        contact_ratios
                        >= args.landing_min_all_feet_contact_ratio
                    )
                    & final_all_feet_contact[done_ids]
                    & (torque_cvs <= args.landing_max_leg_torque_cv)
                    & (linear_speed_rms <= args.landing_max_linear_speed_rms)
                    & (angular_speed_rms <= args.landing_max_angular_speed_rms)
                    & (
                        orientation_error_rms
                        <= args.landing_max_orientation_error_rms
                    )
                )
                completed_landing_observed.extend(observed.float().cpu().tolist())
                completed_post_landing_seconds.extend(
                    (sample_counts.float() * env.dt).cpu().tolist()
                )
                completed_all_feet_contact_ratios.extend(
                    contact_ratios.cpu().tolist()
                )
                completed_final_all_feet_contacts.extend(
                    final_all_feet_contact[done_ids].float().cpu().tolist()
                )
                completed_leg_torque_cvs.extend(torque_cvs.cpu().tolist())
                completed_final_linear_speeds.extend(
                    final_linear_speed[done_ids].cpu().tolist()
                )
                completed_final_angular_speeds.extend(
                    final_angular_speed[done_ids].cpu().tolist()
                )
                completed_final_orientation_errors.extend(
                    final_orientation_error[done_ids].cpu().tolist()
                )
                completed_linear_speed_rms.extend(linear_speed_rms.cpu().tolist())
                completed_angular_speed_rms.extend(angular_speed_rms.cpu().tolist())
                completed_orientation_error_rms.extend(
                    orientation_error_rms.cpu().tolist()
                )
                completed_stable_standing.extend(
                    stable_standing.float().cpu().tolist()
                )
                completed_per_env[done_ids] += 1

                if args.diagnostics:
                    diagnostic_rows = zip(
                        done_ids.cpu().tolist(),
                        done_peaks.cpu().tolist(),
                        done_errors.cpu().tolist(),
                    )
                    for local_index, (env_id, height, landing_error) in enumerate(
                        diagnostic_rows
                    ):
                        record = {
                            "height": float(height),
                            "env_id": float(env_id),
                            "episode_generation": float(
                                episode_generation[env_id].item()
                            ),
                            "terminated": float(
                                (~env.time_out_buf[env_id]).item()
                            ),
                            "timed_out": float(env.time_out_buf[env_id].item()),
                            "landing_error": float(landing_error),
                            "completion_step": float(evaluation_step),
                            "min_height": float(episode_min_height[env_id].item()),
                            "max_vz": float(episode_max_vz[env_id].item()),
                            "max_action_abs_mean": float(
                                episode_max_action_abs_mean[env_id].item()
                            ),
                            "max_torque_ratio": float(
                                episode_max_torque_ratio[env_id].item()
                            ),
                            "first_air_step": float(
                                episode_first_air_step[env_id].item()
                            ),
                            "takeoff_height": float(
                                episode_takeoff_height[env_id].item()
                            ),
                            "takeoff_vz": float(episode_takeoff_vz[env_id].item()),
                            "takeoff_action_abs_mean": float(
                                episode_takeoff_action_abs_mean[env_id].item()
                            ),
                            "takeoff_dof_vel_abs_mean": float(
                                episode_takeoff_dof_vel_abs_mean[env_id].item()
                            ),
                            "takeoff_torque_abs_mean": float(
                                episode_takeoff_torque_abs_mean[env_id].item()
                            ),
                            "physical_first_all_feet_contact_step": float(
                                physical_first_all_feet_contact_step[env_id].item()
                            ),
                            "physical_first_takeoff_step": float(
                                physical_first_takeoff_step[env_id].item()
                            ),
                            "physical_first_contact_height": float(
                                physical_first_contact_height[env_id].item()
                            ),
                            "physical_first_contact_vz": float(
                                physical_first_contact_vz[env_id].item()
                            ),
                            "physical_first_contact_action_abs_mean": float(
                                physical_first_contact_action_abs_mean[env_id].item()
                            ),
                            "physical_max_all_feet_off_steps": float(
                                physical_max_all_feet_off_steps[env_id].item()
                            ),
                            "physical_takeoff_max_height": float(
                                physical_takeoff_max_height[env_id].item()
                            ),
                            "physical_takeoff_max_vz": float(
                                physical_takeoff_max_vz[env_id].item()
                            ),
                            "event_landing_observed": float(
                                event_landing_observed[env_id].item()
                            ),
                            **{
                                name: float(values[env_id].item())
                                for name, values in landing_state_values.items()
                            },
                            "landing_observed": float(observed[local_index].item()),
                            "post_landing_seconds": float(
                                sample_counts[local_index].item() * env.dt
                            ),
                            "post_landing_all_feet_contact_ratio": float(
                                contact_ratios[local_index].item()
                            ),
                            "post_landing_final_all_feet_contact": float(
                                final_all_feet_contact[env_id].item()
                            ),
                            "post_landing_leg_torque_cv": float(
                                torque_cvs[local_index].item()
                            ),
                            "post_landing_linear_speed_rms": float(
                                linear_speed_rms[local_index].item()
                            ),
                            "post_landing_angular_speed_rms": float(
                                angular_speed_rms[local_index].item()
                            ),
                            "post_landing_orientation_error_rms": float(
                                orientation_error_rms[local_index].item()
                            ),
                            "stable_standing": float(
                                stable_standing[local_index].item()
                            ),
                            "termination_contact_body_index": float(
                                env.termination_contact_body_index[env_id].item()
                            ),
                            "termination_contact_force_max": float(
                                env.termination_contact_force_max[env_id].item()
                            ),
                            "sampled_latency_mean_ms": float(
                                (
                                    episode_sampled_latency_sum_ms[env_id]
                                    / torch.clamp(
                                        episode_sampled_latency_count[env_id], min=1
                                    )
                                ).item()
                            ),
                            "sampled_latency_max_ms": float(
                                episode_sampled_latency_max_ms[env_id].item()
                            ),
                        }
                        for name, values in episode_parameters.items():
                            record[name] = float(values[env_id].item())
                        for name, values in episode_step_values.items():
                            record[name] = float(values[env_id].item())
                        for name, values in episode_initial_values.items():
                            record[name] = float(values[env_id].item())
                        for name, values in env.termination_reason_bufs.items():
                            record["termination_reason_{}".format(name)] = float(
                                values[env_id].item()
                            )
                        diagnostic_records.append(record)

                peaks[done_mask] = env.root_states[done_mask, 2]
                landing_observed[done_mask] = False
                landing_step[done_mask] = -1
                post_landing_samples[done_mask] = 0
                post_landing_all_feet_contacts[done_mask] = 0
                post_landing_torque_sum[done_mask] = 0.0
                post_landing_torque_sq_sum[done_mask] = 0.0
                post_landing_linear_speed_sq_sum[done_mask] = 0.0
                post_landing_angular_speed_sq_sum[done_mask] = 0.0
                post_landing_orientation_error_sq_sum[done_mask] = 0.0
                final_all_feet_contact[done_mask] = False
                final_linear_speed[done_mask] = torch.nan
                final_angular_speed[done_mask] = torch.nan
                final_orientation_error[done_mask] = torch.nan
                if args.diagnostics:
                    episode_min_height[done_mask] = env.root_states[done_mask, 2]
                    episode_max_vz[done_mask] = env.root_states[done_mask, 9]
                    episode_max_action_abs_mean[done_mask] = 0.0
                    episode_max_torque_ratio[done_mask] = 0.0
                    episode_first_air_step[done_mask] = -1
                    episode_takeoff_height[done_mask] = torch.nan
                    episode_takeoff_vz[done_mask] = torch.nan
                    episode_takeoff_action_abs_mean[done_mask] = torch.nan
                    episode_takeoff_dof_vel_abs_mean[done_mask] = torch.nan
                    episode_takeoff_torque_abs_mean[done_mask] = torch.nan
                    physical_contact_observed[done_mask] = False
                    physical_first_all_feet_contact_step[done_mask] = -1
                    physical_first_takeoff_step[done_mask] = -1
                    physical_first_contact_height[done_mask] = torch.nan
                    physical_first_contact_vz[done_mask] = torch.nan
                    physical_first_contact_action_abs_mean[done_mask] = torch.nan
                    physical_current_all_feet_off_steps[done_mask] = 0
                    physical_max_all_feet_off_steps[done_mask] = 0
                    physical_takeoff_max_height[done_mask] = torch.nan
                    physical_takeoff_max_vz[done_mask] = torch.nan
                    event_landing_observed[done_mask] = False
                    episode_sampled_latency_sum_ms[done_mask] = 0.0
                    episode_sampled_latency_count[done_mask] = 0
                    episode_sampled_latency_max_ms[done_mask] = 0.0
                    for values in episode_step_values.values():
                        values[done_mask] = torch.nan
                    for values in landing_state_values.values():
                        values[done_mask] = torch.nan
                    episode_initial_captured[done_mask] = False
                    for values in episode_initial_values.values():
                        values[done_mask] = torch.nan
                    episode_generation[done_mask] += 1
                    current_parameters = _snapshot_episode_parameters(env)
                    for name, values in current_parameters.items():
                        if name not in episode_parameters:
                            episode_parameters[name] = values
                        else:
                            episode_parameters[name][done_mask] = values[done_mask]
                if len(completed_heights) >= args.eval_episodes:
                    break

    if len(completed_heights) < args.eval_episodes:
        raise RuntimeError(
            "only {} of {} episodes completed in {} steps".format(
                len(completed_heights), args.eval_episodes, args.eval_max_steps
            )
        )

    metrics = summarize_episodes(
        completed_heights,
        completed_landing_errors,
        env_cfg.rewards.jump_success_height,
    )
    metrics.update(
        {
            "termination_rate": float(np.mean(completed_terminations)),
            "torque_saturation_fraction": torque_saturated / torque_samples,
            "action_abs_mean": action_abs_sum / action_samples,
        }
    )
    if args.landing_stability_seconds:
        metrics.update(
            {
                "landing_observed_rate": float(np.mean(completed_landing_observed)),
                "post_landing_observation_seconds_min": float(
                    np.min(completed_post_landing_seconds)
                ),
                "post_landing_observation_seconds_mean": float(
                    np.mean(completed_post_landing_seconds)
                ),
                "post_landing_all_feet_contact_ratio_mean": float(
                    np.mean(completed_all_feet_contact_ratios)
                ),
                "post_landing_all_feet_contact_ratio_median": float(
                    np.median(completed_all_feet_contact_ratios)
                ),
                "post_landing_final_all_feet_contact_rate": float(
                    np.mean(completed_final_all_feet_contacts)
                ),
                "post_landing_leg_torque_cv_mean": float(
                    np.nanmean(completed_leg_torque_cvs)
                ),
                "post_landing_leg_torque_cv_median": float(
                    np.nanmedian(completed_leg_torque_cvs)
                ),
                "post_landing_final_linear_speed_mean": float(
                    np.nanmean(completed_final_linear_speeds)
                ),
                "post_landing_final_angular_speed_mean": float(
                    np.nanmean(completed_final_angular_speeds)
                ),
                "post_landing_final_orientation_error_mean": float(
                    np.nanmean(completed_final_orientation_errors)
                ),
                "post_landing_linear_speed_rms_mean": float(
                    np.nanmean(completed_linear_speed_rms)
                ),
                "post_landing_angular_speed_rms_mean": float(
                    np.nanmean(completed_angular_speed_rms)
                ),
                "post_landing_orientation_error_rms_mean": float(
                    np.nanmean(completed_orientation_error_rms)
                ),
                "stable_standing_success_rate": float(
                    np.mean(completed_stable_standing)
                ),
            }
        )

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "task": args.task,
        "jump_type": env_cfg.env.jump_type,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "mode": args.eval_mode,
        "seed": args.seed,
        "num_envs": env_cfg.env.num_envs,
        "eval_episodes": args.eval_episodes,
        "target": {"x": args.target_x, "y": args.target_y, "z": 0.0},
        "success_height": env_cfg.rewards.jump_success_height,
        "deterministic_actor": args.action_noise_scale == 0.0,
        "action_source": args.action_source,
        "action_noise_scale": args.action_noise_scale,
        "checkpoint_action_std": {
            "mean": float(checkpoint_std.mean().item()),
            "min": float(checkpoint_std.min().item()),
            "max": float(checkpoint_std.max().item()),
        },
        "fixed_physics": {
            "latency_ms": args.fixed_latency_ms,
            "com_xyz": args.fixed_com_xyz,
            "restitution": args.fixed_restitution,
            "joint_friction": args.fixed_joint_friction,
            "joint_damping": args.fixed_joint_damping,
        },
        "diagnostic_randomization_overrides": {
            "com_range_half_width": args.com_range_half_width,
            "com_range_half_width_xyz": args.com_range_half_width_xyz,
            "latency_range_max_ms": args.latency_range_max_ms,
        },
        "evaluation_control": {
            "stiffness": dict(env_cfg.control.stiffness),
            "damping": dict(env_cfg.control.damping),
            "action_scale": env_cfg.control.action_scale,
            "filter_freq": env_cfg.control.filter_freq,
            "base_height": env_cfg.init_state.pos[2],
        },
        "observation_noise_profile": args.observation_noise_profile,
        "ignored_termination_contact_pattern": (
            args.ignore_termination_contact_pattern
        ),
        "episode_sampling": args.episode_sampling,
        "training_assistance_disabled": True,
        "metrics": metrics,
    }
    if args.landing_stability_seconds:
        result["landing_stability_protocol"] = {
            "configured_episode_length_s": configured_episode_length_s,
            "required_stability_seconds": args.landing_stability_seconds,
            "contact_grace_seconds": args.landing_contact_grace_seconds,
            "min_all_feet_contact_ratio": args.landing_min_all_feet_contact_ratio,
            "require_final_all_feet_contact": True,
            "max_leg_torque_cv": args.landing_max_leg_torque_cv,
            "max_linear_speed_rms": args.landing_max_linear_speed_rms,
            "max_angular_speed_rms": args.landing_max_angular_speed_rms,
            "max_orientation_error_rms": args.landing_max_orientation_error_rms,
            "require_natural_timeout": True,
        }
    if args.diagnostics:
        result["diagnostics"] = _summarize_diagnostics(
            diagnostic_records, env_cfg.rewards.jump_success_height
        )
        result["diagnostics"]["physics_audit"] = physics_audit
        result["diagnostics"]["body_names"] = list(env.body_names)
        result["diagnostics"]["jump_detection_audit"] = (
            _summarize_jump_detection(diagnostic_records)
        )
        # Keep the per-episode rows only in explicit diagnostics output so
        # asynchronous resets can be separated by generation during analysis.
        result["diagnostics"]["records"] = diagnostic_records

    exit_code = 0
    if args.reference_json:
        with open(args.reference_json, "r", encoding="utf-8") as stream:
            reference = json.load(stream)
        result["gate"] = compare_evaluations(
            result,
            reference,
            min_height_ratio=args.min_height_ratio,
            max_success_gap=args.max_success_gap,
        )
        if not result["gate"]["pass"]:
            exit_code = 3

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    if args.trajectory_output_json:
        trajectory_path = os.path.abspath(args.trajectory_output_json)
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
        trajectory_result = {
            "task": args.task,
            "load_run": args.load_run,
            "checkpoint": args.checkpoint,
            "target": {"x": args.target_x, "y": args.target_y},
            "env_id": args.trajectory_env_id,
            "dt": env.dt,
            "foot_names": list(env.cfg.asset.policy_foot_names),
            "dof_names_policy": list(env.cfg.asset.policy_dof_names),
            "ignored_termination_contact_pattern": (
                args.ignore_termination_contact_pattern
            ),
            "records": trajectory_records,
        }
        with open(trajectory_path, "w", encoding="utf-8") as stream:
            json.dump(trajectory_result, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate(get_args(additional_parameters=EVALUATOR_ARGS)))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print("evaluation error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
