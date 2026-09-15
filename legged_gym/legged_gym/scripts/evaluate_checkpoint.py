"""Deterministic, reproducible checkpoint evaluator for jumping policies."""

from __future__ import annotations

import json
import os
import sys

import isaacgym  # noqa: F401; Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from legged_gym.envs import task_registry
from legged_gym.envs.base.legged_robot import LeggedRobot
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
    {"name": "--target_yaw_deg", "type": float},
    {"name": "--fixed_initial_yaw_deg", "type": float},
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
    {"name": "--record_frames_dir", "type": str},
    {"name": "--record_viewer_frames_dir", "type": str},
    {"name": "--record_state_npz", "type": str},
    {"name": "--camera_pos_override", "type": str},
    {"name": "--camera_lookat_override", "type": str},
    {"name": "--asset_file_override", "type": str},
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
    {
        "name": "--initial_dof_pos_delta_policy",
        "type": str,
        "help": "Diagnostic-only 12-value policy-order DOF position delta.",
    },
    {
        "name": "--initial_dof_vel_delta_policy",
        "type": str,
        "help": "Diagnostic-only 12-value policy-order DOF velocity delta.",
    },
    {
        "name": "--action_replay_csv",
        "type": str,
        "help": "Diagnostic-only CSV containing one 12-value policy action per row.",
    },
    {"name": "--fixed_friction", "type": float},
    {"name": "--friction_range_override", "type": str},
    {"name": "--fixed_restitution", "type": float},
    {"name": "--restitution_range_override", "type": str},
    {"name": "--fixed_joint_friction", "type": float},
    {"name": "--joint_friction_range_override", "type": str},
    {"name": "--fixed_joint_damping", "type": float},
    {"name": "--fixed_joint_armature", "type": float},
    {"name": "--fixed_latency_ms", "type": float},
    {"name": "--fixed_com_xyz", "type": str},
    {"name": "--fixed_added_mass", "type": float},
    {"name": "--fixed_link_mass_factor", "type": float},
    {"name": "--fixed_motor_strength", "type": float},
    {"name": "--fixed_p_gain_multiplier", "type": float},
    {"name": "--fixed_d_gain_multiplier", "type": float},
    {"name": "--fixed_motor_offset", "type": float},
    {"name": "--com_range_half_width", "type": float},
    {"name": "--com_range_half_width_xyz", "type": str},
    {"name": "--latency_range_max_ms", "type": float},
    {"name": "--control_stiffness", "type": float},
    {"name": "--control_damping", "type": float},
    {"name": "--physical_dof_velocity_limit_override", "type": float},
    {"name": "--physical_dof_velocity_limit_scale", "type": float},
    {"name": "--velocity_torque_limit_scale", "type": float},
    {"name": "--velocity_torque_envelope_blend", "type": float},
    {"name": "--velocity_torque_envelope", "type": int, "choices": [0, 1]},
    {"name": "--takeoff_angvel_kick_max_override", "type": str},
    {"name": "--takeoff_angvel_kick_probability_override", "type": float},
    {"name": "--takeoff_contact_force_perturb_max_override", "type": str},
    {"name": "--takeoff_contact_force_perturb_probability_override", "type": float},
    {"name": "--action_scale_override", "type": float},
    {"name": "--filter_freq_override", "type": float},
    {"name": "--clip_actions_override", "type": float},
    {"name": "--torque_limit_scale_override", "type": float},
    {"name": "--max_action_delta_override", "type": float},
    {"name": "--base_height_override", "type": float},
    {"name": "--initial_stance_height_range_override", "type": str},
    {
        "name": "--initial_stance_kinematic_margin_override",
        "type": float,
        "default": 0.0,
    },
    {
        "name": "--initial_stance_height_probability_override",
        "type": float,
        "default": 1.0,
    },
    {"name": "--reset_landing_error_override", "type": float},
    {"name": "--command_position_observation_scale_override", "type": float},
    {"name": "--observe_joint_friction_in_command_y", "type": int},
    {"name": "--forward_position_tolerance", "type": float, "default": 0.05},
    {"name": "--forward_yaw_tolerance_deg", "type": float, "default": 10.0},
    {
        "name": "--counterfactual_target_x",
        "type": float,
        "help": "Evaluate actor sensitivity to another target_x without advancing physics",
    },
    {
        "name": "--initial_contact_history_probability",
        "type": float,
        "default": 0.0,
    },
    {
        "name": "--settled_contact_count",
        "type": int,
        "help": "Required contacting feet before the jump state machine is armed",
    },
    {
        "name": "--contact_observation_delay_max_steps",
        "type": int,
        "default": 0,
        "help": (
            "Diagnostic per-foot episodic contact-observation delay, sampled "
            "uniformly from 0..N policy steps"
        ),
    },
    {"name": "--initial_contact_settle_steps", "type": int, "default": 1},
    {
        "name": "--fixed_action_lag_timesteps",
        "type": int,
        "help": (
            "Diagnostic-only fixed action-application lag in policy steps; "
            "does not enable observation or PD latency"
        ),
    },
    {
        "name": "--initial_contact_base_height_offset",
        "type": float,
        "default": 0.0,
    },
    {"name": "--ignore_termination_contact_pattern", "type": str},
    {
        "name": "--terrain_mesh_type_override",
        "type": str,
        "choices": ["plane", "trimesh"],
    },
    {"name": "--sim_substeps_override", "type": int},
    {"name": "--physx_position_iterations_override", "type": int},
    {"name": "--physx_max_depenetration_velocity_override", "type": float},
    {"name": "--physx_contact_offset_override", "type": float},
    {"name": "--physx_rest_offset_override", "type": float},
    {
        "name": "--physx_friction_offset_threshold_override",
        "type": float,
    },
    {
        "name": "--observation_noise_profile",
        "type": str,
        "default": "all",
        "choices": ["all", "contacts", "continuous", "none"],
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


def _install_solo12_stance_height_randomization(
    env_cfg, height_range, probability, kinematic_margin=0.0
):
    """Use the same contact-consistent stance-height reset as training."""
    if height_range is None:
        return
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "--initial_stance_height_probability_override must be in [0, 1]"
        )
    try:
        bounds = [float(part.strip()) for part in height_range.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "--initial_stance_height_range_override must be min,max"
        ) from exc
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(
            "--initial_stance_height_range_override must be ordered as min,max"
        )
    lower, upper = bounds
    foot_radius = 0.021
    link_length = 0.2
    max_height = foot_radius + 2.0 * link_length
    if kinematic_margin < 0.0:
        raise ValueError(
            "--initial_stance_kinematic_margin_override must be non-negative"
        )
    if lower <= foot_radius or upper + kinematic_margin >= max_height:
        raise ValueError(
            "Solo12 stance-height range must lie strictly inside "
            f"({foot_radius}, {max_height}) m"
        )

    env_cfg.env.initial_stance_height_range = [lower, upper]
    env_cfg.env.initial_stance_height_probability = probability
    original_reset_dofs = LeggedRobot._reset_dofs
    original_reset_root_states = LeggedRobot._reset_root_states

    def _physical_reset_ids(self, env_ids):
        if self.cfg.env.continuous_jumping:
            return self.cont_jump_reset_env_ids
        return env_ids

    def _reset_dofs_with_stance_height(self, env_ids):
        original_reset_dofs(self, env_ids)
        if not hasattr(self, "_sampled_initial_stance_height"):
            self._sampled_initial_stance_height = torch.zeros(
                self.num_envs, device=self.device
            )
            self._sampled_initial_stance_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        self._sampled_initial_stance_active[env_ids] = False
        reset_ids = _physical_reset_ids(self, env_ids)
        count = reset_ids.numel()
        if count == 0:
            return
        sampled_height = torch.empty(count, device=self.device).uniform_(
            lower, upper
        )
        active = torch.rand(count, device=self.device) < probability
        self._sampled_initial_stance_height[reset_ids] = sampled_height
        self._sampled_initial_stance_active[reset_ids] = active
        active_ids = reset_ids[active]
        if active_ids.numel() == 0:
            return

        active_height = sampled_height[active]
        kinematic_height = active_height + kinematic_margin
        ratio = (kinematic_height - foot_radius) / (2.0 * link_length)
        thigh = torch.acos(torch.clamp(ratio, -1.0, 1.0))
        calf = -2.0 * thigh
        for dof_index, dof_name in enumerate(self.dof_names):
            upper_name = dof_name.upper()
            if "HAA" in upper_name or "HIP" in upper_name:
                self.dof_pos[active_ids, dof_index] = 0.0
            elif "HFE" in upper_name or "THIGH" in upper_name:
                self.dof_pos[active_ids, dof_index] = thigh
            elif "KFE" in upper_name or "CALF" in upper_name:
                self.dof_pos[active_ids, dof_index] = calf
        self.dof_vel[active_ids] = 0.0
        active_ids_int32 = active_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(active_ids_int32),
            len(active_ids_int32),
        )

    def _reset_root_states_with_stance_height(self, env_ids):
        original_reset_root_states(self, env_ids)
        if not hasattr(self, "_sampled_initial_stance_active"):
            return
        reset_ids = _physical_reset_ids(self, env_ids)
        active = self._sampled_initial_stance_active[reset_ids]
        active_ids = reset_ids[active]
        if active_ids.numel() == 0:
            return
        self.root_states[active_ids, 2] = (
            self.env_origins[active_ids, 2]
            + self._sampled_initial_stance_height[active_ids]
        )
        self.initial_root_states[active_ids, 2] = self.root_states[active_ids, 2]
        self.initial_root_states_nonrandomised[active_ids, 2] = self.root_states[
            active_ids, 2
        ]
        active_ids_int32 = active_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(active_ids_int32),
            len(active_ids_int32),
        )

    LeggedRobot._reset_dofs = _reset_dofs_with_stance_height
    LeggedRobot._reset_root_states = _reset_root_states_with_stance_height


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
    if args.asset_file_override is not None:
        env_cfg.asset.file = args.asset_file_override
    if args.record_frames_dir is not None:
        env_cfg.viewer.simulate_camera = True
    if (
        args.record_frames_dir is not None
        or args.record_viewer_frames_dir is not None
    ):
        for argument_name, config_name in (
            ("camera_pos_override", "pos"),
            ("camera_lookat_override", "lookat"),
        ):
            value = getattr(args, argument_name)
            if value is None:
                continue
            components = [float(component) for component in value.split(",")]
            if len(components) != 3:
                raise ValueError(
                    f"--{argument_name} must contain three comma-separated values"
                )
            setattr(env_cfg.viewer, config_name, components)

    # Diagnostic-only single-factor overrides. These are deliberately applied
    # to the in-memory evaluation config and never mutate a task/training file.
    if args.control_stiffness is not None:
        env_cfg.control.stiffness = {"joint": args.control_stiffness}
    if args.control_damping is not None:
        env_cfg.control.damping = {"joint": args.control_damping}
    if args.physical_dof_velocity_limit_override is not None:
        if args.physical_dof_velocity_limit_override <= 0.0:
            raise ValueError("--physical_dof_velocity_limit_override must be positive")
        env_cfg.asset.physical_dof_velocity_limit_override = (
            args.physical_dof_velocity_limit_override
        )
    if args.physical_dof_velocity_limit_scale is not None:
        if args.physical_dof_velocity_limit_scale < 1.0:
            raise ValueError("--physical_dof_velocity_limit_scale must be >= 1")
        env_cfg.asset.physical_dof_velocity_limit_scale = (
            args.physical_dof_velocity_limit_scale
        )
    if args.velocity_torque_envelope is not None:
        env_cfg.control.velocity_torque_envelope = bool(args.velocity_torque_envelope)
    if args.velocity_torque_limit_scale is not None:
        if args.velocity_torque_limit_scale <= 0.0:
            raise ValueError("--velocity_torque_limit_scale must be positive")
        env_cfg.control.velocity_torque_limit_scale = args.velocity_torque_limit_scale
    if args.velocity_torque_envelope_blend is not None:
        if not 0.0 <= args.velocity_torque_envelope_blend <= 1.0:
            raise ValueError("--velocity_torque_envelope_blend must be in [0, 1]")
        env_cfg.control.velocity_torque_envelope_blend = (
            args.velocity_torque_envelope_blend
        )
    if args.takeoff_angvel_kick_max_override is not None:
        values = [float(value) for value in args.takeoff_angvel_kick_max_override.split(",")]
        if len(values) != 3 or any(value < 0.0 for value in values):
            raise ValueError(
                "--takeoff_angvel_kick_max_override must be three "
                "non-negative values: roll,pitch,yaw"
            )
        env_cfg.domain_rand.takeoff_angvel_kick_max = values
    if args.takeoff_angvel_kick_probability_override is not None:
        probability = args.takeoff_angvel_kick_probability_override
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "--takeoff_angvel_kick_probability_override must be in [0,1]"
            )
        env_cfg.domain_rand.takeoff_angvel_kick_probability = probability
    if args.takeoff_contact_force_perturb_max_override is not None:
        values = [float(value) for value in args.takeoff_contact_force_perturb_max_override.split(",")]
        if len(values) != 2 or any(value < 0.0 for value in values):
            raise ValueError("--takeoff_contact_force_perturb_max_override must be roll,pitch non-negative force maxima")
        env_cfg.domain_rand.takeoff_contact_force_perturb_max = values
    if args.takeoff_contact_force_perturb_probability_override is not None:
        probability = args.takeoff_contact_force_perturb_probability_override
        if not 0.0 <= probability <= 1.0:
            raise ValueError("--takeoff_contact_force_perturb_probability_override must be in [0,1]")
        env_cfg.domain_rand.takeoff_contact_force_perturb_probability = probability
    if args.action_scale_override is not None:
        env_cfg.control.action_scale = args.action_scale_override
    if args.filter_freq_override is not None:
        env_cfg.control.filter_freq = args.filter_freq_override
    if args.clip_actions_override is not None:
        env_cfg.normalization.clip_actions = args.clip_actions_override
    if args.max_action_delta_override is not None:
        env_cfg.control.max_action_delta = args.max_action_delta_override
    if args.base_height_override is not None:
        env_cfg.init_state.pos[2] = args.base_height_override
    if args.reset_landing_error_override is not None:
        if args.reset_landing_error_override <= 0.0:
            raise ValueError("--reset_landing_error_override must be positive")
        env_cfg.env.reset_landing_error = args.reset_landing_error_override
    if args.command_position_observation_scale_override is not None:
        if args.command_position_observation_scale_override <= 0.0:
            raise ValueError(
                "--command_position_observation_scale_override must be positive"
            )
        env_cfg.env.command_position_observation_scale = (
            args.command_position_observation_scale_override
        )
    if args.observe_joint_friction_in_command_y is not None:
        env_cfg.env.observe_joint_friction_in_command_y = bool(
            args.observe_joint_friction_in_command_y
        )
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
    contact_overrides = {
        "contact_offset": args.physx_contact_offset_override,
        "rest_offset": args.physx_rest_offset_override,
        "friction_offset_threshold": (
            args.physx_friction_offset_threshold_override
        ),
    }
    for name, value in contact_overrides.items():
        if value is None:
            continue
        if name != "rest_offset" and value <= 0.0:
            raise ValueError(f"PhysX {name} must be positive")
        setattr(env_cfg.sim.physx, name, value)
    if env_cfg.sim.physx.rest_offset > env_cfg.sim.physx.contact_offset:
        raise ValueError("PhysX rest_offset cannot exceed contact_offset")
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
    if args.observation_noise_profile == "none":
        env_cfg.noise.add_noise = False
    elif args.observation_noise_profile == "contacts":
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
    commands.distances.des_yaw = (
        None if args.target_yaw_deg is None else np.deg2rad(args.target_yaw_deg)
    )
    commands.ranges.pos_dx_ini = [args.target_x, args.target_x]
    commands.ranges.pos_dy_ini = [args.target_y, args.target_y]
    commands.ranges.pos_dz_ini = [0.0, 0.0]

    _disable_initial_state_and_assistance(env_cfg)
    if args.fixed_initial_yaw_deg is not None:
        initial_yaw = float(np.deg2rad(args.fixed_initial_yaw_deg))
        env_cfg.domain_rand.randomize_robot_ori = True
        env_cfg.domain_rand.ranges.min_ori_euler = [0.0, 0.0, initial_yaw]
        env_cfg.domain_rand.ranges.max_ori_euler = [0.0, 0.0, initial_yaw]
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
    if args.friction_range_override is not None:
        friction_range = [
            float(part.strip())
            for part in args.friction_range_override.split(",")
        ]
        if len(friction_range) != 2 or friction_range[0] > friction_range[1]:
            raise ValueError(
                "--friction_range_override must be formatted as min,max"
            )
        if friction_range[0] < 0.0:
            raise ValueError("--friction_range_override must be non-negative")
        env_cfg.domain_rand.randomize_friction = True
        ranges.friction_range = friction_range
    if args.fixed_friction is not None:
        if args.fixed_friction < 0.0:
            raise ValueError("--fixed_friction must be non-negative")
        env_cfg.domain_rand.randomize_friction = True
        ranges.friction_range = [args.fixed_friction, args.fixed_friction]
    if args.restitution_range_override is not None:
        restitution_range = [
            float(part.strip())
            for part in args.restitution_range_override.split(",")
        ]
        if len(restitution_range) != 2 or restitution_range[0] > restitution_range[1]:
            raise ValueError(
                "--restitution_range_override must be formatted as min,max"
            )
        env_cfg.domain_rand.randomize_restitution = True
        ranges.restitution_range = restitution_range
    if args.fixed_restitution is not None:
        env_cfg.domain_rand.randomize_restitution = True
        ranges.restitution_range = [args.fixed_restitution, args.fixed_restitution]
    if args.joint_friction_range_override is not None:
        joint_friction_range = [
            float(part.strip())
            for part in args.joint_friction_range_override.split(",")
        ]
        if len(joint_friction_range) != 2 or joint_friction_range[0] > joint_friction_range[1]:
            raise ValueError(
                "--joint_friction_range_override must be formatted as min,max"
            )
        env_cfg.domain_rand.randomize_joint_friction = True
        ranges.joint_friction_range = joint_friction_range
    if args.fixed_joint_friction is not None:
        env_cfg.domain_rand.randomize_joint_friction = True
        ranges.joint_friction_range = [
            args.fixed_joint_friction,
            args.fixed_joint_friction,
        ]
    if args.fixed_joint_damping is not None:
        env_cfg.domain_rand.randomize_joint_damping = True
        ranges.joint_damping_range = [args.fixed_joint_damping, args.fixed_joint_damping]
    if args.fixed_joint_armature is not None:
        if args.fixed_joint_armature < 0.0:
            raise ValueError("--fixed_joint_armature must be non-negative")
        env_cfg.domain_rand.randomize_joint_armature = True
        ranges.joint_armature_range = [
            args.fixed_joint_armature,
            args.fixed_joint_armature,
        ]
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
    if args.fixed_added_mass is not None:
        env_cfg.domain_rand.randomize_base_mass = True
        ranges.added_mass_range = [args.fixed_added_mass, args.fixed_added_mass]
    if args.fixed_link_mass_factor is not None:
        if args.fixed_link_mass_factor <= 0.0:
            raise ValueError("--fixed_link_mass_factor must be positive")
        env_cfg.domain_rand.randomize_link_mass = True
        ranges.added_link_mass_range = [
            args.fixed_link_mass_factor,
            args.fixed_link_mass_factor,
        ]
    if args.fixed_motor_strength is not None:
        if args.fixed_motor_strength <= 0.0:
            raise ValueError("--fixed_motor_strength must be positive")
        env_cfg.domain_rand.randomize_motor_strength = True
        ranges.motor_strength_ranges = [
            args.fixed_motor_strength,
            args.fixed_motor_strength,
        ]
    if args.fixed_p_gain_multiplier is not None:
        if args.fixed_p_gain_multiplier <= 0.0:
            raise ValueError("--fixed_p_gain_multiplier must be positive")
        env_cfg.domain_rand.randomize_PD_gains = True
        ranges.p_gains_range = [
            args.fixed_p_gain_multiplier,
            args.fixed_p_gain_multiplier,
        ]
    if args.fixed_d_gain_multiplier is not None:
        if args.fixed_d_gain_multiplier <= 0.0:
            raise ValueError("--fixed_d_gain_multiplier must be positive")
        env_cfg.domain_rand.randomize_PD_gains = True
        ranges.d_gains_range = [
            args.fixed_d_gain_multiplier,
            args.fixed_d_gain_multiplier,
        ]
    if args.fixed_motor_offset is not None:
        env_cfg.domain_rand.randomize_motor_offset = True
        ranges.motor_offset_range = [
            args.fixed_motor_offset,
            args.fixed_motor_offset,
        ]

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

    def parse_policy_vector(value, flag):
        if value is None:
            return None
        try:
            vector = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"{flag} must contain 12 comma-separated numbers") from exc
        if len(vector) != 12:
            raise ValueError(f"{flag} must contain exactly 12 values")
        return vector

    initial_dof_pos_delta_policy = parse_policy_vector(
        args.initial_dof_pos_delta_policy, "--initial_dof_pos_delta_policy"
    )
    initial_dof_vel_delta_policy = parse_policy_vector(
        args.initial_dof_vel_delta_policy, "--initial_dof_vel_delta_policy"
    )
    for name in (
        "control_stiffness",
        "control_damping",
        "action_scale_override",
        "filter_freq_override",
        "clip_actions_override",
        "max_action_delta_override",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise ValueError("--{} must be positive".format(name))
    if args.base_height_override is not None and args.base_height_override <= 0.0:
        raise ValueError("--base_height_override must be positive")
    if not 0.0 <= args.initial_contact_history_probability <= 1.0:
        raise ValueError("--initial_contact_history_probability must be in [0, 1]")
    if args.settled_contact_count is not None and not 1 <= args.settled_contact_count <= 4:
        raise ValueError("--settled_contact_count must be between 1 and 4")
    if args.initial_contact_settle_steps < 0:
        raise ValueError("--initial_contact_settle_steps must be non-negative")
    if args.fixed_action_lag_timesteps is not None and args.fixed_action_lag_timesteps < 0:
        raise ValueError("--fixed_action_lag_timesteps must be non-negative")
    if args.initial_contact_base_height_offset < 0.0:
        raise ValueError("--initial_contact_base_height_offset must be non-negative")
    if args.contact_observation_delay_max_steps < 0:
        raise ValueError(
            "--contact_observation_delay_max_steps must be non-negative"
        )
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
    _install_solo12_stance_height_randomization(
        env_cfg,
        args.initial_stance_height_range_override,
        args.initial_stance_height_probability_override,
        args.initial_stance_kinematic_margin_override,
    )
    env_cfg.env.initial_contact_history_probability = (
        args.initial_contact_history_probability
    )
    if args.settled_contact_count is not None:
        env_cfg.env.settled_contact_count = args.settled_contact_count
    env_cfg.env.initial_contact_settle_steps = args.initial_contact_settle_steps
    if args.fixed_action_lag_timesteps is not None:
        env_cfg.domain_rand.lag_timesteps = args.fixed_action_lag_timesteps
        env_cfg.domain_rand.randomize_lag_timesteps = (
            args.fixed_action_lag_timesteps > 0
        )
    env_cfg.env.initial_contact_base_height_offset = (
        args.initial_contact_base_height_offset
    )
    env_cfg.env.contact_observation_delay_max_steps = (
        args.contact_observation_delay_max_steps
    )
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
    if args.torque_limit_scale_override is not None:
        if args.torque_limit_scale_override <= 0.0:
            raise ValueError("--torque_limit_scale_override must be positive")
        env.torque_limits *= args.torque_limit_scale_override
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
    for delta_policy, state in (
        (initial_dof_pos_delta_policy, env.dof_pos),
        (initial_dof_vel_delta_policy, env.dof_vel),
    ):
        if delta_policy is None:
            continue
        delta = torch.as_tensor(
            delta_policy, dtype=state.dtype, device=env.device
        ).unsqueeze(0).repeat(env.num_envs, 1)
        state[all_env_ids] += delta[:, env.sim_dof_indices_in_policy]
    if (
        initial_dof_pos_delta_policy is not None
        or initial_dof_vel_delta_policy is not None
    ):
        env.gym.set_dof_state_tensor(
            env.sim, gymtorch.unwrap_tensor(env.dof_state)
        )
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
    replay_actions = None
    if args.action_replay_csv is not None:
        replay_array = np.loadtxt(args.action_replay_csv, delimiter=",")
        replay_array = np.atleast_2d(replay_array)
        if replay_array.shape[1] != env.num_actions:
            raise ValueError(
                "--action_replay_csv must have exactly {} columns, got {}".format(
                    env.num_actions, replay_array.shape[1]
                )
            )
        replay_actions = torch.as_tensor(
            replay_array, dtype=obs.dtype, device=env.device
        )
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
    completed_joint_velocity_rms = []
    completed_joint_velocity_abs_max = []
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
    post_landing_joint_velocity_sq_sum = torch.zeros(
        env.num_envs, device=env.device
    )
    post_landing_joint_velocity_abs_max = torch.zeros(
        env.num_envs, device=env.device
    )
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
                "takeoff_pitch_abs_deg",
                "pre_takeoff_contact_mismatch_seconds",
                "takeoff_pitch_angular_impulse_normalized",
                "takeoff_pitch_angular_impulse_abs_normalized",
            )
        }
        episode_min_height = env.root_states[:, 2].clone()
        episode_max_vz = env.root_states[:, 9].clone()
        episode_max_action_abs_mean = torch.zeros(env.num_envs, device=env.device)
        episode_max_action_abs = torch.zeros(env.num_envs, device=env.device)
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
        episode_takeoff_vx = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_dx = torch.full(
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
        episode_takeoff_pitch_abs_deg = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_roll_rate_abs = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_specific_vertical_impulse = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_front_rear_vertical_impulse_imbalance = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_front_rear_timing_gap = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_window_specific_net_vertical_impulse = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_window_specific_pitch_angular_impulse = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_window_contact_mismatch = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_vzero_specific_net_vertical_impulse = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_vzero_specific_pitch_angular_impulse = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_vzero_contact_mismatch = torch.full(
            (env.num_envs,), torch.nan, device=env.device
        )
        episode_takeoff_vzero_propulsion_valid = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        episode_pre_takeoff_contact_mismatch_seconds = torch.zeros(
            env.num_envs, device=env.device
        )
        episode_takeoff_pitch_angular_impulse_normalized = torch.zeros(
            env.num_envs, device=env.device
        )
        episode_takeoff_pitch_angular_impulse_abs_normalized = torch.zeros(
            env.num_envs, device=env.device
        )
        episode_takeoff_pitch_cancellation_normalized = torch.zeros(
            env.num_envs, device=env.device
        )
        episode_takeoff_front_rear_timing_gap = torch.zeros(
            env.num_envs, device=env.device
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
                "initial_force_contact_fraction",
                "initial_min_vertical_contact_force",
                "initial_linear_speed",
                "initial_angular_speed",
                "initial_min_foot_height",
                "initial_max_foot_height",
                "initial_dof_position_error_abs_mean",
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
    counterfactual_action_abs_sum = 0.0
    counterfactual_action_samples = 0
    counterfactual_action_abs_max = 0.0
    counterfactual_observation_abs_max = 0.0
    counterfactual_observation_changed_columns = 0
    counterfactual_observation_changed_indices = []
    recorded_frame_index = 0
    if args.record_frames_dir is not None:
        os.makedirs(args.record_frames_dir, exist_ok=True)
    if args.record_viewer_frames_dir is not None:
        os.makedirs(args.record_viewer_frames_dir, exist_ok=True)
    recorded_rigid_body_states = []
    recorded_root_states = []
    recorded_dof_positions = []
    recorded_actions = []
    if args.record_state_npz is not None:
        recorded_rigid_body_states.append(
            env.rigid_body_state[0].detach().cpu().numpy().copy()
        )
        recorded_root_states.append(
            env.root_states[0].detach().cpu().numpy().copy()
        )
        recorded_dof_positions.append(
            env.dof_pos[0].detach().cpu().numpy().copy()
        )
        recorded_actions.append(np.zeros(env.num_actions, dtype=np.float32))

    with torch.inference_mode():
        for evaluation_step in range(args.eval_max_steps):
            actions = actor_critic.act_inference(obs.detach())
            if args.counterfactual_target_x is not None:
                original_commands = env.commands.clone()
                env.commands[:, 0] = float(args.counterfactual_target_x)
                env.compute_observations()
                counterfactual_obs = env.get_observations().detach().clone()
                counterfactual_actions = actor_critic.act_inference(
                    counterfactual_obs
                )
                observation_difference = torch.abs(counterfactual_obs - obs)
                action_difference = torch.abs(counterfactual_actions - actions)
                counterfactual_action_abs_sum += float(
                    action_difference.sum().item()
                )
                counterfactual_action_samples += int(action_difference.numel())
                counterfactual_action_abs_max = max(
                    counterfactual_action_abs_max,
                    float(action_difference.max().item()),
                )
                counterfactual_observation_abs_max = max(
                    counterfactual_observation_abs_max,
                    float(observation_difference.max().item()),
                )
                if evaluation_step == 0:
                    changed_mask = torch.any(
                        observation_difference > 1e-8, dim=0
                    )
                    counterfactual_observation_changed_columns = int(
                        changed_mask.sum().item()
                    )
                    counterfactual_observation_changed_indices = (
                        torch.nonzero(changed_mask, as_tuple=False)
                        .flatten()
                        .cpu()
                        .tolist()
                    )
                env.commands.copy_(original_commands)
                env.compute_observations()
                obs = env.get_observations()
            if replay_actions is not None:
                replay_index = min(evaluation_step, replay_actions.shape[0] - 1)
                actions = replay_actions[replay_index].unsqueeze(0).repeat(
                    env.num_envs, 1
                )
            elif args.action_source == "zero":
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
                    joint_torques_policy = env._dof_to_policy(env.torques)[
                        trace_env_id
                    ]
                    foot_positions = env.feet_pos[trace_env_id]
                    foot_contact_forces = env.contact_forces[
                        trace_env_id, env.feet_indices, :
                    ]
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
                            "root_position_world": env.root_states[
                                trace_env_id, 0:3
                            ].cpu().tolist(),
                            "root_quaternion_xyzw": env.root_states[
                                trace_env_id, 3:7
                            ].cpu().tolist(),
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
                            "foot_contact_forces_world": foot_contact_forces.cpu().tolist(),
                            "calf_contact_force_norms": torch.linalg.norm(
                                env.contact_forces[
                                    trace_env_id, calf_body_indices, :
                                ],
                                dim=-1,
                            ).cpu().tolist(),
                            "dof_pos_policy": dof_pos_policy.cpu().tolist(),
                            "dof_vel_policy": dof_vel_policy.cpu().tolist(),
                            "joint_torques_policy": joint_torques_policy.cpu().tolist(),
                            "actions_policy": actions[trace_env_id].cpu().tolist(),
                            "policy_observation": obs[trace_env_id].cpu().tolist(),
                        }
                    )
            if args.diagnostics:
                first_decision = (
                    ~episode_initial_captured
                    & (env.initial_zero_action_steps_remaining == 0)
                )
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
                    initial_vertical_forces = torch.clamp(
                        env.contact_forces[first_decision][:, env.feet_indices, 2],
                        min=0.0,
                    )
                    episode_initial_values["initial_force_contact_fraction"][
                        first_decision
                    ] = (initial_vertical_forces > 1.0).float().mean(1)
                    episode_initial_values["initial_min_vertical_contact_force"][
                        first_decision
                    ] = initial_vertical_forces.min(1).values
                    episode_initial_values["initial_linear_speed"][first_decision] = (
                        torch.linalg.norm(env.root_states[first_decision, 7:10], dim=1)
                    )
                    episode_initial_values["initial_angular_speed"][first_decision] = (
                        torch.linalg.norm(env.root_states[first_decision, 10:13], dim=1)
                    )
                    episode_initial_values["initial_min_foot_height"][first_decision] = (
                        env.feet_pos[first_decision, :, 2].min(1).values
                    )
                    episode_initial_values["initial_max_foot_height"][first_decision] = (
                        env.feet_pos[first_decision, :, 2].max(1).values
                    )
                    episode_initial_values[
                        "initial_dof_position_error_abs_mean"
                    ][first_decision] = torch.abs(
                        env.dof_pos[first_decision] - env.default_dof_pos
                    ).mean(1)
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
                episode_max_action_abs = torch.maximum(
                    episode_max_action_abs, torch.abs(actions).max(1).values
                )
            action_abs_sum += torch.abs(actions).sum().item()
            action_samples += actions.numel()

            obs, _, _, dones, _ = env.step(actions.detach())
            if args.record_state_npz is not None and not bool(dones[0].item()):
                recorded_rigid_body_states.append(
                    env.rigid_body_state[0].detach().cpu().numpy().copy()
                )
                recorded_root_states.append(
                    env.root_states[0].detach().cpu().numpy().copy()
                )
                recorded_dof_positions.append(
                    env.dof_pos[0].detach().cpu().numpy().copy()
                )
                recorded_actions.append(
                    actions[0].detach().cpu().numpy().copy()
                )
            if (
                args.record_frames_dir is not None
                and evaluation_step % 2 == 1
            ):
                robot_xy = (
                    env.root_states[0, :2] - env.env_origins[0, :2]
                ).detach().cpu().numpy()
                camera_position = np.asarray(env_cfg.viewer.pos).copy()
                camera_lookat = np.asarray(env_cfg.viewer.lookat).copy()
                camera_position[:2] += robot_xy
                camera_lookat[:2] += robot_xy
                env.gym.set_camera_location(
                    env.camera_handle,
                    env.envs[0],
                    gymapi.Vec3(*camera_position),
                    gymapi.Vec3(*camera_lookat),
                )
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                env.gym.write_camera_image_to_file(
                    env.sim,
                    env.envs[0],
                    env.camera_handle,
                    gymapi.IMAGE_COLOR,
                    os.path.join(
                        args.record_frames_dir,
                        f"{recorded_frame_index:06d}.png",
                    ),
                )
                recorded_frame_index += 1
            if (
                args.record_viewer_frames_dir is not None
                and evaluation_step % 2 == 1
            ):
                # Viewer camera coordinates are in the global simulation frame.
                # Unlike a camera sensor attached to env 0, they must include
                # the environment's (possibly non-zero) terrain origin.
                robot_xy = env.root_states[0, :2].detach().cpu().numpy()
                camera_position = np.asarray(env_cfg.viewer.pos).copy()
                camera_lookat = np.asarray(env_cfg.viewer.lookat).copy()
                camera_position[:2] += robot_xy
                camera_lookat[:2] += robot_xy
                env.set_camera(camera_position, camera_lookat)
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                env.gym.write_viewer_image_to_file(
                    env.viewer,
                    os.path.join(
                        args.record_viewer_frames_dir,
                        f"{recorded_frame_index:06d}.png",
                    ),
                )
                recorded_frame_index += 1

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
                post_landing_joint_velocity_sq_sum[standing_sample] += torch.mean(
                    torch.square(env.dof_vel[standing_sample]), dim=1
                )
                post_landing_joint_velocity_abs_max[standing_sample] = torch.maximum(
                    post_landing_joint_velocity_abs_max[standing_sample],
                    torch.max(torch.abs(env.dof_vel[standing_sample]), dim=1).values,
                )
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
                episode_takeoff_pitch_angular_impulse_normalized = torch.maximum(
                    episode_takeoff_pitch_angular_impulse_normalized,
                    env.takeoff_pitch_angular_impulse_normalized,
                )
                episode_takeoff_pitch_angular_impulse_abs_normalized = torch.maximum(
                    episode_takeoff_pitch_angular_impulse_abs_normalized,
                    env.takeoff_pitch_angular_impulse_abs_normalized,
                )
                episode_takeoff_pitch_cancellation_normalized = torch.maximum(
                    episode_takeoff_pitch_cancellation_normalized,
                    env.takeoff_pitch_cancellation_normalized,
                )
                episode_takeoff_front_rear_timing_gap = torch.maximum(
                    episode_takeoff_front_rear_timing_gap,
                    env.takeoff_front_rear_timing_gap,
                )
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
                pre_takeoff_contact_window = (
                    active_mask
                    & env.settled_after_init
                    & physical_contact_observed
                    & (episode_first_air_step < 0)
                )
                front_contact = torch.any(env.contacts[:, :2], dim=1)
                rear_contact = torch.any(env.contacts[:, 2:], dim=1)
                episode_pre_takeoff_contact_mismatch_seconds[
                    pre_takeoff_contact_window
                ] += (
                    front_contact[pre_takeoff_contact_window]
                    != rear_contact[pre_takeoff_contact_window]
                ).float() * env.dt
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
                    episode_takeoff_vx[first_air] = env.root_states[first_air, 7]
                    episode_takeoff_dx[first_air] = (
                        env.root_states[first_air, 0]
                        - env.initial_root_states[first_air, 0]
                    )
                    episode_takeoff_action_abs_mean[first_air] = torch.abs(actions[first_air]).mean(1)
                    episode_takeoff_dof_vel_abs_mean[first_air] = torch.abs(
                        env.dof_vel[first_air]
                    ).mean(1)
                    episode_takeoff_torque_abs_mean[first_air] = torch.abs(
                        env.torques[first_air]
                    ).mean(1)
                    episode_takeoff_pitch_abs_deg[first_air] = torch.rad2deg(
                        torch.abs(env.euler[first_air, 1])
                    )
                    episode_takeoff_roll_rate_abs[first_air] = torch.abs(
                        env.base_ang_vel[first_air, 0]
                    )
                    episode_takeoff_specific_vertical_impulse[first_air] = (
                        env.takeoff_specific_vertical_impulse[first_air]
                    )
                    episode_takeoff_front_rear_vertical_impulse_imbalance[
                        first_air
                    ] = env.takeoff_front_rear_vertical_impulse_imbalance[
                        first_air
                    ]
                    episode_takeoff_front_rear_timing_gap[first_air] = (
                        env.takeoff_front_rear_timing_gap[first_air]
                    )
                    episode_takeoff_window_specific_net_vertical_impulse[
                        first_air
                    ] = env.takeoff_window_specific_net_vertical_impulse[first_air]
                    episode_takeoff_window_specific_pitch_angular_impulse[
                        first_air
                    ] = env.takeoff_window_specific_pitch_angular_impulse[first_air]
                    episode_takeoff_window_contact_mismatch[first_air] = (
                        env.takeoff_window_contact_mismatch[first_air]
                    )
                    episode_takeoff_vzero_specific_net_vertical_impulse[
                        first_air
                    ] = env.takeoff_vzero_specific_net_vertical_impulse[first_air]
                    episode_takeoff_vzero_specific_pitch_angular_impulse[
                        first_air
                    ] = env.takeoff_vzero_specific_pitch_angular_impulse[first_air]
                    episode_takeoff_vzero_contact_mismatch[first_air] = (
                        env.takeoff_vzero_contact_mismatch[first_air]
                    )
                    episode_takeoff_vzero_propulsion_valid[first_air] = (
                        env.takeoff_vzero_propulsion_valid[first_air]
                    )

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
                joint_velocity_rms = torch.sqrt(
                    post_landing_joint_velocity_sq_sum[done_ids]
                    / torch.clamp(sample_counts, min=1)
                )
                joint_velocity_abs_max = post_landing_joint_velocity_abs_max[done_ids]
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
                    natural_completion
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
                completed_joint_velocity_rms.extend(
                    joint_velocity_rms.cpu().tolist()
                )
                completed_joint_velocity_abs_max.extend(
                    joint_velocity_abs_max.cpu().tolist()
                )
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
                            "command_x": float(
                                env.terminal_command_xy_buf[env_id, 0].item()
                            ),
                            "command_y": float(
                                env.terminal_command_xy_buf[env_id, 1].item()
                            ),
                            "landing_displacement_x": float(
                                env.terminal_landing_displacement_buf[
                                    env_id, 0
                                ].item()
                            ),
                            "landing_displacement_y": float(
                                env.terminal_landing_displacement_buf[
                                    env_id, 1
                                ].item()
                            ),
                            "landing_yaw_error_abs_deg": float(
                                torch.rad2deg(
                                    env.terminal_landing_yaw_error_buf[env_id]
                                ).item()
                            ),
                            "landing_parallel_displacement": float(
                                torch.sum(
                                    env.terminal_landing_displacement_buf[env_id]
                                    * env.terminal_command_xy_buf[env_id]
                                ).item()
                                / max(
                                    float(
                                        torch.linalg.norm(
                                            env.terminal_command_xy_buf[env_id]
                                        ).item()
                                    ),
                                    1e-6,
                                )
                            ),
                            "completion_step": float(evaluation_step),
                            "min_height": float(episode_min_height[env_id].item()),
                            "max_vz": float(episode_max_vz[env_id].item()),
                            "max_action_abs_mean": float(
                                episode_max_action_abs_mean[env_id].item()
                            ),
                            "max_action_abs": float(
                                episode_max_action_abs[env_id].item()
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
                            "takeoff_vx": float(episode_takeoff_vx[env_id].item()),
                            "takeoff_dx": float(episode_takeoff_dx[env_id].item()),
                            "takeoff_action_abs_mean": float(
                                episode_takeoff_action_abs_mean[env_id].item()
                            ),
                            "takeoff_dof_vel_abs_mean": float(
                                episode_takeoff_dof_vel_abs_mean[env_id].item()
                            ),
                            "takeoff_torque_abs_mean": float(
                                episode_takeoff_torque_abs_mean[env_id].item()
                            ),
                            "takeoff_tail_max_speed_ratio": float(env.terminal_takeoff_tail_max_speed_ratio[env_id].item()),
                            "takeoff_tail_max_positive_power": float(env.terminal_takeoff_tail_max_positive_power[env_id].item()),
                            "takeoff_tail_near_limit_power": float(env.terminal_takeoff_tail_near_limit_power[env_id].item()),
                            "takeoff_tail_near_limit_torque_ratio": float(env.terminal_takeoff_tail_near_limit_torque_ratio[env_id].item()),
                            "takeoff_tail_event_count": float(env.terminal_takeoff_tail_event_count[env_id].item()),
                            "takeoff_tail_release_power": float(env.terminal_takeoff_tail_release_power[env_id].item()),
                            "takeoff_tail_release_speed_ratio": float(env.terminal_takeoff_tail_release_speed_ratio[env_id].item()),
                            "takeoff_tail_max_dq_step": float(env.terminal_takeoff_tail_max_dq_step[env_id].item()),
                            "takeoff_tail_max_actuator_speed_ratio": float(env.terminal_takeoff_tail_max_actuator_speed_ratio[env_id].item()),
                            "max_pre_takeoff_speed_ratio": float(env.terminal_max_pre_takeoff_speed_ratio[env_id].item()),
                            "takeoff_tail_requested_near_limit_power": float(env.terminal_takeoff_tail_requested_near_limit_power[env_id].item()),
                            "takeoff_tail_requested_near_limit_torque_ratio": float(env.terminal_takeoff_tail_requested_near_limit_torque_ratio[env_id].item()),
                            "takeoff_tail_requested_event_count": float(env.terminal_takeoff_tail_requested_event_count[env_id].item()),
                            "takeoff_tail_release_requested_power": float(env.terminal_takeoff_tail_release_requested_power[env_id].item()),
                            "physical_first_all_feet_contact_step": float(
                                physical_first_all_feet_contact_step[env_id].item()
                            ),
                            "physical_first_takeoff_step": float(
                                physical_first_takeoff_step[env_id].item()
                            ),
                            "takeoff_pitch_abs_deg": float(
                                episode_takeoff_pitch_abs_deg[env_id].item()
                            ),
                            "takeoff_roll_rate_abs": float(
                                episode_takeoff_roll_rate_abs[env_id].item()
                            ),
                            "takeoff_angvel_kick_applied": bool(
                                env.takeoff_angvel_kick_applied[env_id].item()
                            ),
                            "takeoff_angvel_kick_norm": float(
                                torch.linalg.vector_norm(
                                    env.takeoff_angvel_kick[env_id]
                                ).item()
                            ),
                            "takeoff_specific_vertical_impulse": float(
                                episode_takeoff_specific_vertical_impulse[
                                    env_id
                                ].item()
                            ),
                            "takeoff_front_rear_vertical_impulse_imbalance": float(
                                episode_takeoff_front_rear_vertical_impulse_imbalance[
                                    env_id
                                ].item()
                            ),
                            "takeoff_front_rear_timing_gap": float(
                                episode_takeoff_front_rear_timing_gap[env_id].item()
                            ),
                            "takeoff_window_specific_net_vertical_impulse": float(
                                episode_takeoff_window_specific_net_vertical_impulse[
                                    env_id
                                ].item()
                            ),
                            "takeoff_window_specific_pitch_angular_impulse": float(
                                episode_takeoff_window_specific_pitch_angular_impulse[
                                    env_id
                                ].item()
                            ),
                            "takeoff_window_contact_mismatch": float(
                                episode_takeoff_window_contact_mismatch[env_id].item()
                            ),
                            "takeoff_vzero_specific_net_vertical_impulse": float(
                                episode_takeoff_vzero_specific_net_vertical_impulse[
                                    env_id
                                ].item()
                            ),
                            "takeoff_vzero_specific_pitch_angular_impulse": float(
                                episode_takeoff_vzero_specific_pitch_angular_impulse[
                                    env_id
                                ].item()
                            ),
                            "takeoff_vzero_contact_mismatch": float(
                                episode_takeoff_vzero_contact_mismatch[env_id].item()
                            ),
                            "takeoff_vzero_propulsion_valid": float(
                                episode_takeoff_vzero_propulsion_valid[env_id].item()
                            ),
                            "pre_takeoff_contact_mismatch_seconds": float(
                                episode_pre_takeoff_contact_mismatch_seconds[
                                    env_id
                                ].item()
                            ),
                            "takeoff_pitch_angular_impulse_normalized": float(
                                episode_takeoff_pitch_angular_impulse_normalized[
                                    env_id
                                ].item()
                            ),
                            "takeoff_pitch_angular_impulse_abs_normalized": float(
                                episode_takeoff_pitch_angular_impulse_abs_normalized[
                                    env_id
                                ].item()
                            ),
                            "takeoff_pitch_cancellation_normalized": float(
                                episode_takeoff_pitch_cancellation_normalized[
                                    env_id
                                ].item()
                            ),
                            "takeoff_front_rear_timing_gap_seconds": float(
                                episode_takeoff_front_rear_timing_gap[
                                    env_id
                                ].item()
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
                            "post_landing_joint_velocity_rms": float(
                                joint_velocity_rms[local_index].item()
                            ),
                            "post_landing_joint_velocity_abs_max": float(
                                joint_velocity_abs_max[local_index].item()
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
                post_landing_joint_velocity_sq_sum[done_mask] = 0.0
                post_landing_joint_velocity_abs_max[done_mask] = 0.0
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
                    episode_max_action_abs[done_mask] = 0.0
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
                    episode_takeoff_pitch_abs_deg[done_mask] = torch.nan
                    episode_takeoff_roll_rate_abs[done_mask] = torch.nan
                    episode_takeoff_specific_vertical_impulse[done_mask] = torch.nan
                    episode_takeoff_front_rear_vertical_impulse_imbalance[
                        done_mask
                    ] = torch.nan
                    episode_takeoff_front_rear_timing_gap[done_mask] = torch.nan
                    episode_takeoff_window_specific_net_vertical_impulse[
                        done_mask
                    ] = torch.nan
                    episode_takeoff_window_specific_pitch_angular_impulse[
                        done_mask
                    ] = torch.nan
                    episode_takeoff_window_contact_mismatch[done_mask] = torch.nan
                    episode_takeoff_vzero_specific_net_vertical_impulse[
                        done_mask
                    ] = torch.nan
                    episode_takeoff_vzero_specific_pitch_angular_impulse[
                        done_mask
                    ] = torch.nan
                    episode_takeoff_vzero_contact_mismatch[done_mask] = torch.nan
                    episode_takeoff_vzero_propulsion_valid[done_mask] = False
                    episode_pre_takeoff_contact_mismatch_seconds[done_mask] = 0.0
                    episode_takeoff_pitch_angular_impulse_normalized[
                        done_mask
                    ] = 0.0
                    episode_takeoff_pitch_angular_impulse_abs_normalized[
                        done_mask
                    ] = 0.0
                    episode_takeoff_pitch_cancellation_normalized[
                        done_mask
                    ] = 0.0
                    episode_takeoff_front_rear_timing_gap[done_mask] = 0.0
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
    if args.counterfactual_target_x is not None:
        metrics.update(
            {
                "counterfactual_target_x": float(args.counterfactual_target_x),
                "counterfactual_observation_abs_max": float(
                    counterfactual_observation_abs_max
                ),
                "counterfactual_observation_changed_columns": int(
                    counterfactual_observation_changed_columns
                ),
                "counterfactual_observation_changed_indices": list(
                    counterfactual_observation_changed_indices
                ),
                "counterfactual_action_abs_difference_mean": float(
                    counterfactual_action_abs_sum
                    / max(counterfactual_action_samples, 1)
                ),
                "counterfactual_action_abs_difference_max": float(
                    counterfactual_action_abs_max
                ),
            }
        )
    if getattr(env, "jump_type", "") != "upwards" and diagnostic_records:
        forward_position_errors = np.asarray(
            [abs(record["landing_error"]) for record in diagnostic_records],
            dtype=np.float64,
        )
        forward_position_ok = (
            forward_position_errors <= float(args.forward_position_tolerance)
        )
        forward_height_ok = np.asarray(
            [
                record["height"] >= float(env_cfg.rewards.jump_success_height)
                for record in diagnostic_records
            ],
            dtype=bool,
        )
        forward_safe = np.asarray(
            [not bool(record["terminated"]) for record in diagnostic_records],
            dtype=bool,
        )
        metrics.update(
            {
                "forward_position_tolerance": float(
                    args.forward_position_tolerance
                ),
                "forward_position_success_rate": float(
                    np.mean(forward_position_ok)
                ),
                "forward_joint_success_rate": float(
                    np.mean(forward_position_ok & forward_height_ok & forward_safe)
                ),
                "forward_landing_parallel_displacement_mean": float(
                    np.mean(
                        [
                            record["landing_parallel_displacement"]
                            for record in diagnostic_records
                        ]
                    )
                ),
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
                "post_landing_joint_velocity_rms_mean": float(
                    np.nanmean(completed_joint_velocity_rms)
                ),
                "post_landing_joint_velocity_rms_median": float(
                    np.nanmedian(completed_joint_velocity_rms)
                ),
                "post_landing_joint_velocity_abs_max_mean": float(
                    np.nanmean(completed_joint_velocity_abs_max)
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
        if env_cfg.env.jump_type == "forward" and diagnostic_records:
            forward_position_ok = np.asarray(
                [
                    record["landing_error"] <= args.forward_position_tolerance
                    for record in diagnostic_records
                ],
                dtype=bool,
            )
            forward_safe = np.asarray(
                [not bool(record["terminated"]) for record in diagnostic_records],
                dtype=bool,
            )
            forward_landing_observed = np.asarray(
                [bool(record["landing_observed"]) for record in diagnostic_records],
                dtype=bool,
            )
            forward_stable = np.asarray(
                [bool(record["stable_standing"]) for record in diagnostic_records],
                dtype=bool,
            )
            metrics["forward_acceptance_success_rate"] = float(
                np.mean(
                    forward_position_ok
                    & forward_safe
                    & forward_landing_observed
                    & forward_stable
                )
            )
            if args.target_yaw_deg is not None:
                forward_yaw_errors_deg = np.asarray(
                    [
                        record["landing_yaw_error_abs_deg"]
                        for record in diagnostic_records
                    ],
                    dtype=np.float64,
                )
                forward_yaw_ok = (
                    forward_yaw_errors_deg <= args.forward_yaw_tolerance_deg
                )
                metrics["forward_yaw_tolerance_deg"] = float(
                    args.forward_yaw_tolerance_deg
                )
                metrics["forward_yaw_error_abs_mean_deg"] = float(
                    np.mean(forward_yaw_errors_deg)
                )
                metrics["forward_yaw_success_rate"] = float(
                    np.mean(forward_yaw_ok)
                )
                metrics["forward_acceptance_success_rate"] = float(
                    np.mean(
                        forward_position_ok
                        & forward_yaw_ok
                        & forward_safe
                        & forward_landing_observed
                        & forward_stable
                    )
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
        "target": {
            "x": args.target_x,
            "y": args.target_y,
            "z": 0.0,
            "yaw_deg": args.target_yaw_deg,
        },
        "success_height": env_cfg.rewards.jump_success_height,
        "deterministic_actor": args.action_noise_scale == 0.0,
        "action_source": args.action_source,
        "physical_dof_velocity_limit_override": (
            args.physical_dof_velocity_limit_override
        ),
        "physical_dof_velocity_limit_scale": args.physical_dof_velocity_limit_scale,
        "velocity_torque_limit_scale": args.velocity_torque_limit_scale,
        "velocity_torque_envelope_blend": args.velocity_torque_envelope_blend,
        "velocity_torque_envelope": args.velocity_torque_envelope,
        "initial_dof_pos_delta_policy": initial_dof_pos_delta_policy,
        "initial_dof_vel_delta_policy": initial_dof_vel_delta_policy,
        "action_replay_csv": args.action_replay_csv,
        "action_noise_scale": args.action_noise_scale,
        "checkpoint_action_std": {
            "mean": float(checkpoint_std.mean().item()),
            "min": float(checkpoint_std.min().item()),
            "max": float(checkpoint_std.max().item()),
        },
        "fixed_physics": {
            "latency_ms": args.fixed_latency_ms,
            "com_xyz": args.fixed_com_xyz,
            "friction": args.fixed_friction,
            "restitution": args.fixed_restitution,
            "joint_friction": args.fixed_joint_friction,
            "joint_damping": args.fixed_joint_damping,
            "joint_armature": args.fixed_joint_armature,
            "added_mass": args.fixed_added_mass,
            "link_mass_factor": args.fixed_link_mass_factor,
            "motor_strength": args.fixed_motor_strength,
            "p_gain_multiplier": args.fixed_p_gain_multiplier,
            "d_gain_multiplier": args.fixed_d_gain_multiplier,
            "motor_offset": args.fixed_motor_offset,
        },
        "diagnostic_randomization_overrides": {
            "friction_range": args.friction_range_override,
            "com_range_half_width": args.com_range_half_width,
            "com_range_half_width_xyz": args.com_range_half_width_xyz,
            "latency_range_max_ms": args.latency_range_max_ms,
            "restitution_range": args.restitution_range_override,
            "joint_friction_range": args.joint_friction_range_override,
            "physx_contact_offset": args.physx_contact_offset_override,
            "physx_rest_offset": args.physx_rest_offset_override,
            "physx_friction_offset_threshold": (
                args.physx_friction_offset_threshold_override
            ),
        },
        "evaluation_control": {
            "stiffness": dict(env_cfg.control.stiffness),
            "damping": dict(env_cfg.control.damping),
            "action_scale": env_cfg.control.action_scale,
            "filter_freq": env_cfg.control.filter_freq,
            "clip_actions": env_cfg.normalization.clip_actions,
            "max_action_delta": getattr(env_cfg.control, "max_action_delta", 0.0),
            "base_height": env_cfg.init_state.pos[2],
            "initial_stance_kinematic_margin": (
                args.initial_stance_kinematic_margin_override
            ),
            "initial_contact_history_probability": (
                env_cfg.env.initial_contact_history_probability
            ),
            "settled_contact_count": getattr(
                env_cfg.env, "settled_contact_count", 4
            ),
            "initial_contact_settle_steps": env_cfg.env.initial_contact_settle_steps,
            "fixed_action_lag_timesteps": args.fixed_action_lag_timesteps,
            "initial_contact_base_height_offset": (
                env_cfg.env.initial_contact_base_height_offset
            ),
            "contact_observation_delay_max_steps": (
                env_cfg.env.contact_observation_delay_max_steps
            ),
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
    if args.record_state_npz:
        state_path = os.path.abspath(args.record_state_npz)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        np.savez_compressed(
            state_path,
            dt=np.asarray(env.dt, dtype=np.float32),
            body_names=np.asarray(env.body_names),
            dof_names=np.asarray(env.dof_names),
            rigid_body_states=np.asarray(recorded_rigid_body_states),
            root_states=np.asarray(recorded_root_states),
            dof_positions=np.asarray(recorded_dof_positions),
            actions=np.asarray(recorded_actions),
        )
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(evaluate(get_args(additional_parameters=EVALUATOR_ARGS)))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print("evaluation error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
