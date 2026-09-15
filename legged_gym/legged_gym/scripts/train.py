# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import json
import math
import numpy as np
import os
import sys
from datetime import datetime

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.recovery_state import install_recovery_state_replay
import torch
import wandb


def _parse_com_half_width_xyz(value):
    try:
        half_widths = [float(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            "--com_range_half_width_xyz must be three comma-separated numbers"
        ) from exc
    if len(half_widths) != 3 or any(width < 0.0 for width in half_widths):
        raise ValueError(
            "--com_range_half_width_xyz must contain three non-negative values"
        )
    return half_widths


def _parse_range(value, option_name):
    try:
        bounds = [float(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"{option_name} must be two comma-separated numbers"
        ) from exc
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(f"{option_name} must be ordered as min,max")
    if bounds[0] < 0.0:
        raise ValueError(f"{option_name} must be non-negative")
    return bounds


def _config_to_jsonable(value, seen=None):
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _config_to_jsonable(item, seen)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_config_to_jsonable(item, seen) for item in value]
    object_id = id(value)
    if object_id in seen:
        return "<recursive-reference>"
    seen.add(object_id)
    converted = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        item = getattr(value, key)
        if callable(item):
            continue
        converted[key] = _config_to_jsonable(item, seen)
    seen.remove(object_id)
    return converted if converted else str(value)


def _write_effective_config(path, args, env_cfg):
    output_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "task": args.task,
        "command_line": sys.argv,
        "arguments": vars(args),
        "environment": _config_to_jsonable(env_cfg),
    }
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
    print(f"Effective training configuration written to {output_path}")


def _install_solo12_stance_height_randomization(
    env_cfg, height_range, probability
):
    """Correlate Solo12 reset base height with a contact-consistent leg pose."""
    if height_range is None:
        return
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "--initial_stance_height_probability_override must be in [0, 1]"
        )
    lower, upper = _parse_range(
        height_range, "--initial_stance_height_range_override"
    )
    foot_radius = 0.021
    link_length = 0.2
    max_height = foot_radius + 2.0 * link_length
    if lower <= foot_radius or upper >= max_height:
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
        ratio = (active_height - foot_radius) / (2.0 * link_length)
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
    print(
        "Solo12 correlated initial stance-height randomization: "
        f"range={[lower, upper]} m, probability={probability}"
    )


def _install_post_landing_hind_width_reward(
    env_cfg,
    scale,
    minimum_width,
    grace_seconds,
    maximum_collapse,
    window_seconds,
    contact_loss_weight,
    joint_velocity_weight,
    angular_velocity_weight,
    slip_weight,
    collapse_weight,
    collapse_free,
    displacement_weight,
    displacement_free,
):
    """Penalize only a too-narrow hind stance after a real jump and landing."""
    if scale is None:
        return
    if scale > 0.0:
        raise ValueError(
            "--post_landing_hind_width_scale_override must be non-positive"
        )
    if minimum_width <= 0.0:
        raise ValueError(
            "--post_landing_hind_width_min_override must be positive"
        )
    if grace_seconds < 0.0:
        raise ValueError(
            "--post_landing_hind_width_grace_seconds_override must be "
            "non-negative"
        )
    if maximum_collapse is not None and maximum_collapse < 0.0:
        raise ValueError(
            "--post_landing_hind_width_max_collapse_override must be "
            "non-negative"
        )
    if window_seconds <= 0.0:
        raise ValueError(
            "--post_landing_hind_width_window_seconds_override must be positive"
        )
    auxiliary_weights = {
        "contact_loss": contact_loss_weight,
        "joint_velocity": joint_velocity_weight,
        "angular_velocity": angular_velocity_weight,
        "slip": slip_weight,
        "collapse": collapse_weight,
        "displacement": displacement_weight,
    }
    if any(value < 0.0 for value in auxiliary_weights.values()):
        raise ValueError("Post-landing hind-width auxiliary weights must be non-negative")
    if collapse_free < 0.0 or displacement_free < 0.0:
        raise ValueError("Post-landing free collapse/displacement must be non-negative")
    env_cfg.rewards.scales.post_landing_hind_width = scale
    env_cfg.rewards.post_landing_hind_width_min = minimum_width

    def _reward_post_landing_hind_width(self):
        if not hasattr(self, "_hind_width_seen_real_air"):
            self._hind_width_seen_real_air = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._hind_width_landed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._hind_width_steps_since_landing = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._hind_width_in_real_air = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._hind_width_at_touchdown = torch.zeros(
                self.num_envs, device=self.device
            )
            self._hind_width_touchdown_foot_xy = torch.zeros(
                self.num_envs,
                len(self.feet_indices),
                2,
                device=self.device,
            )

        reset = self.episode_length_buf <= 1
        self._hind_width_seen_real_air[reset] = False
        self._hind_width_landed[reset] = False
        self._hind_width_steps_since_landing[reset] = 0
        self._hind_width_in_real_air[reset] = False
        self._hind_width_at_touchdown[reset] = 0.0
        self._hind_width_touchdown_foot_xy[reset] = 0.0
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        real_air = (
            (~torch.any(foot_contact, dim=1))
            & (self.root_states[:, 9] > 0.5)
            & (self.episode_length_buf > 5)
        )
        self._hind_width_in_real_air = real_air
        self._hind_width_seen_real_air |= real_air
        # A new real airborne phase closes the preceding landing window and
        # arms the detector for exactly one new landing. This is essential for
        # continuous-jump training: the reward must not stay active forever
        # after the first landing in an episode.
        self._hind_width_landed[real_air] = False
        self._hind_width_steps_since_landing[real_air] = 0
        first_landing = (
            ~self._hind_width_landed
            &
            self._hind_width_seen_real_air
            & torch.all(foot_contact, dim=1)
        )
        self._hind_width_landed |= first_landing
        self._hind_width_seen_real_air[first_landing] = False
        self._hind_width_steps_since_landing = torch.where(
            self._hind_width_landed,
            self._hind_width_steps_since_landing + 1,
            torch.zeros_like(self._hind_width_steps_since_landing),
        )

        feet_world = self.rigid_body_state[:, self.feet_indices, 0:3]
        feet_relative = feet_world - self.root_states[:, None, 0:3]
        root_quat = self.root_states[:, None, 3:7].expand(
            -1, feet_relative.shape[1], -1
        )
        feet_body = quat_rotate_inverse(
            root_quat.reshape(-1, 4), feet_relative.reshape(-1, 3)
        ).reshape_as(feet_relative)
        hind_indices = torch.topk(
            feet_body[:, :, 0], k=2, dim=1, largest=False
        ).indices
        hind_y = torch.gather(
            feet_body[:, :, 1], 1, hind_indices
        )
        hind_width = torch.abs(hind_y[:, 0] - hind_y[:, 1])
        self._hind_width_at_touchdown[first_landing] = hind_width[first_landing]
        self._hind_width_touchdown_foot_xy[first_landing] = feet_world[
            first_landing, :, 0:2
        ]
        required_width = torch.full_like(hind_width, minimum_width)
        if maximum_collapse is not None:
            required_width = torch.maximum(
                required_width,
                self._hind_width_at_touchdown - maximum_collapse,
            )
        deficit = torch.clamp(required_width - hind_width, min=0.0)
        normalized_penalty = torch.square(
            deficit / torch.clamp(required_width, min=1e-6)
        )
        grace_steps = int(math.ceil(grace_seconds / self.dt))
        window_steps = max(1, int(math.ceil(window_seconds / self.dt)))
        window_active = (
            (self._hind_width_steps_since_landing > grace_steps)
            & (
                self._hind_width_steps_since_landing
                <= grace_steps + window_steps
            )
        )

        # Do not gate the width loss on contact. Otherwise lifting a foot makes
        # the loss disappear and is an easier policy response than recovering
        # a stable four-foot stance.
        contact_fraction = torch.mean(foot_contact.float(), dim=1)
        contact_loss = 1.0 - contact_fraction
        joint_speed = torch.sqrt(torch.mean(torch.square(self.dof_vel), dim=1))
        joint_speed_excess = torch.square(
            torch.clamp((joint_speed - 0.5) / 1.5, min=0.0, max=1.0)
        )
        angular_speed = torch.linalg.vector_norm(self.root_states[:, 10:13], dim=1)
        angular_speed_excess = torch.square(
            torch.clamp((angular_speed - 0.5) / 1.5, min=0.0, max=1.0)
        )
        foot_speed_xy = torch.linalg.vector_norm(
            self.rigid_body_state[:, self.feet_indices, 7:9], dim=2
        )
        contacting_count = torch.clamp(
            torch.sum(foot_contact.float(), dim=1), min=1.0
        )
        contacting_slip_speed = torch.sum(
            foot_speed_xy * foot_contact.float(), dim=1
        ) / contacting_count
        slip_excess = torch.square(
            torch.clamp((contacting_slip_speed - 0.05) / 0.25, min=0.0, max=1.0)
        )
        collapse = torch.clamp(
            self._hind_width_at_touchdown - hind_width - collapse_free,
            min=0.0,
        )
        collapse_excess = torch.square(
            torch.clamp(collapse / 0.05, min=0.0, max=1.0)
        )
        foot_displacement = torch.linalg.vector_norm(
            feet_world[:, :, 0:2] - self._hind_width_touchdown_foot_xy,
            dim=2,
        )
        mean_foot_displacement = torch.mean(foot_displacement, dim=1)
        displacement_excess = torch.square(
            torch.clamp(
                (mean_foot_displacement - displacement_free) / 0.05,
                min=0.0,
                max=1.0,
            )
        )
        combined_penalty = (
            normalized_penalty
            + contact_loss_weight * contact_loss
            + joint_velocity_weight * joint_speed_excess
            + angular_velocity_weight * angular_speed_excess
            + slip_weight * slip_excess
            + collapse_weight * collapse_excess
            + displacement_weight * displacement_excess
        )
        return combined_penalty * window_active.float()

    LeggedRobot._reward_post_landing_hind_width = (
        _reward_post_landing_hind_width
    )
    print(
        "Post-landing hind-width penalty: "
        f"scale={scale}, minimum_width={minimum_width} m, "
        f"grace={grace_seconds} s, window={window_seconds} s, "
        f"max_collapse={maximum_collapse} m, auxiliary_weights={auxiliary_weights}"
    )


def _apply_upward_retrain_stage(env_cfg, stage):
    """Apply the frozen S1-S4 upward retraining randomization curriculum."""
    valid_stages = {
        "s1",
        "s1_init",
        "s2",
        "s3_friction",
        "s3_motor",
        "s3_com",
        "s3_restitution",
        "s4_latency20",
        "s4_latency30",
        "s4_latency40",
    }
    if stage is None:
        return
    if stage not in valid_stages:
        raise ValueError(
            "--upward_retrain_stage must be one of: "
            + ", ".join(sorted(valid_stages))
        )

    domain_rand = env_cfg.domain_rand
    ranges = domain_rand.ranges
    randomized_flags = (
        "push_robots",
        "randomize_robot_pos",
        "randomize_robot_vel",
        "randomize_robot_ori",
        "randomize_dof_pos",
        "randomize_spring_params",
        "randomize_motor_strength",
        "randomize_PD_gains",
        "randomize_has_jumped",
        "randomize_motor_offset",
        "randomize_base_mass",
        "randomize_com",
        "randomize_restitution",
        "randomize_link_mass",
        "randomize_joint_friction",
        "randomize_joint_damping",
    )
    for flag in randomized_flags:
        if hasattr(domain_rand, flag):
            setattr(domain_rand, flag, False)

    # S1 nominal physics still exercises the unchanged latency and friction
    # code paths, but both distributions collapse to their nominal endpoints.
    domain_rand.pos_vel_random_prob = 0.0
    domain_rand.has_jumped_random_prob = 0.0
    domain_rand.randomize_friction = True
    ranges.friction_range = [0.85, 0.85]
    domain_rand.friction_boundary_probability = 0.0
    domain_rand.sim_latency = True
    ranges.latency_range = [0.0, 0.0]

    if stage == "s1_init":
        # Single-variable S1 repair: retain nominal physics while restoring
        # the task's existing initial-state diversity for exploration.
        for flag in (
            "randomize_robot_pos",
            "randomize_robot_vel",
            "randomize_robot_ori",
            "randomize_dof_pos",
        ):
            if hasattr(domain_rand, flag):
                setattr(domain_rand, flag, True)
        domain_rand.pos_vel_random_prob = 0.7

    if stage not in {"s1", "s1_init"}:
        # S2 base: light initial-state, actuator, friction and latency spread.
        for flag in (
            "randomize_robot_pos",
            "randomize_robot_vel",
            "randomize_robot_ori",
            "randomize_dof_pos",
            "randomize_motor_strength",
        ):
            if hasattr(domain_rand, flag):
                setattr(domain_rand, flag, True)
        domain_rand.pos_vel_random_prob = 0.2
        ranges.motor_strength_ranges = [0.95, 1.05]
        ranges.friction_range = [0.7, 1.1]
        ranges.latency_range = [0.0, 20.0]

    s3_or_later = stage.startswith("s3_") or stage.startswith("s4_")
    if s3_or_later:
        ranges.friction_range = [0.3, 1.5]
        ranges.friction_boundary_range = [0.7, 1.0]
        domain_rand.friction_boundary_probability = 0.5
    if stage in {
        "s3_motor", "s3_com", "s3_restitution",
        "s4_latency20", "s4_latency30", "s4_latency40",
    }:
        ranges.motor_strength_ranges = [0.9, 1.1]
    if stage in {
        "s3_com", "s3_restitution",
        "s4_latency20", "s4_latency30", "s4_latency40",
    }:
        if hasattr(domain_rand, "randomize_com"):
            domain_rand.randomize_com = True
        ranges.com_displacement_range = [[-0.02, -0.02, -0.02], [0.02, 0.02, 0.02]]
    if stage in {
        "s3_restitution", "s4_latency20", "s4_latency30", "s4_latency40",
    }:
        if hasattr(domain_rand, "randomize_restitution"):
            domain_rand.randomize_restitution = True
        ranges.restitution_range = [0.0, 0.4]

    if stage.startswith("s4_"):
        # Restore every pre-existing randomizer for final robustness. Damping
        # is explicitly capped at the historical [0, 0.01] range.
        for flag in randomized_flags:
            if hasattr(domain_rand, flag):
                setattr(domain_rand, flag, True)
        domain_rand.pos_vel_random_prob = 0.7
        domain_rand.has_jumped_random_prob = 0.8
        ranges.joint_damping_range = [0.0, 0.01]
        ranges.latency_range = [0.0, float(stage[len("s4_latency"):])]

    print(f"Upward retraining stage profile: {stage}")


def train(args):
    # args.headless = True
    # args.resume = True
    # args.load_run = 'Mar02_17-00-01_'
    # args.num_envs = 2
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.save_interval_override is not None:
        if args.save_interval_override <= 0:
            raise ValueError("--save_interval_override must be positive")
        train_cfg.runner.save_interval = args.save_interval_override
    if args.upward_action_reference_csv is not None:
        reference_actions = np.atleast_2d(
            np.loadtxt(args.upward_action_reference_csv, delimiter=",")
        )
        if reference_actions.shape[1] != env_cfg.env.num_actions:
            raise ValueError(
                "--upward_action_reference_csv must contain exactly "
                f"{env_cfg.env.num_actions} columns"
            )
        env_cfg.rewards.upward_action_reference = reference_actions.tolist()
    if args.upward_action_reference_scale_override is not None:
        env_cfg.rewards.scales.upward_action_reference = (
            args.upward_action_reference_scale_override
        )
    if args.upward_action_reference_sigma_override is not None:
        if args.upward_action_reference_sigma_override <= 0.0:
            raise ValueError(
                "--upward_action_reference_sigma_override must be positive"
            )
        env_cfg.rewards.upward_action_reference_sigma = (
            args.upward_action_reference_sigma_override
        )
    ranges = env_cfg.domain_rand.ranges
    physx = env_cfg.sim.physx
    stance_height_probability = (
        1.0
        if args.initial_stance_height_probability_override is None
        else args.initial_stance_height_probability_override
    )
    _install_solo12_stance_height_randomization(
        env_cfg,
        args.initial_stance_height_range_override,
        stance_height_probability,
    )
    hind_width_minimum = (
        0.12
        if args.post_landing_hind_width_min_override is None
        else args.post_landing_hind_width_min_override
    )
    _install_post_landing_hind_width_reward(
        env_cfg,
        args.post_landing_hind_width_scale_override,
        hind_width_minimum,
        args.post_landing_hind_width_grace_seconds_override,
        args.post_landing_hind_width_max_collapse_override,
        args.post_landing_hind_width_window_seconds_override,
        args.post_landing_hind_width_contact_loss_weight_override,
        args.post_landing_hind_width_joint_velocity_weight_override,
        args.post_landing_hind_width_angular_velocity_weight_override,
        args.post_landing_hind_width_slip_weight_override,
        args.post_landing_hind_width_collapse_weight_override,
        args.post_landing_hind_width_collapse_free_override,
        args.post_landing_hind_width_displacement_weight_override,
        args.post_landing_hind_width_displacement_free_override,
    )
    install_recovery_state_replay(
        LeggedRobot,
        args.recovery_state_clone_probability,
        args.recovery_state_clone_max_post_landing_steps,
        args.recovery_state_clone_min_height,
    )
    _apply_upward_retrain_stage(env_cfg, args.upward_retrain_stage)
    if args.contact_observation_delay_max_steps_override is not None:
        delay_steps = args.contact_observation_delay_max_steps_override
        if delay_steps < 0:
            raise ValueError(
                "--contact_observation_delay_max_steps_override must be non-negative"
            )
        env_cfg.env.contact_observation_delay_max_steps = delay_steps
        print(
            "Diagnostic per-foot contact-observation delay: "
            f"uniform integer [0, {delay_steps}] policy steps"
        )
    if args.settled_contact_count_override is not None:
        settled_contact_count = args.settled_contact_count_override
        if not 1 <= settled_contact_count <= 4:
            raise ValueError(
                "--settled_contact_count_override must be between 1 and 4"
            )
        env_cfg.env.settled_contact_count = settled_contact_count
        print(
            "Settled-state foot-contact count override: "
            f"{settled_contact_count}"
        )
    if args.friction_range_override is not None:
        ranges.friction_range = _parse_range(
            args.friction_range_override, "--friction_range_override"
        )
        env_cfg.domain_rand.randomize_friction = True
    boundary_options = (
        args.friction_boundary_range,
        args.friction_boundary_probability,
    )
    if any(value is not None for value in boundary_options):
        if any(value is None for value in boundary_options):
            raise ValueError(
                "--friction_boundary_range and "
                "--friction_boundary_probability must be used together"
            )
        if not 0.0 <= args.friction_boundary_probability <= 1.0:
            raise ValueError(
                "--friction_boundary_probability must be between 0 and 1"
            )
        ranges.friction_boundary_range = _parse_range(
            args.friction_boundary_range, "--friction_boundary_range"
        )
        env_cfg.domain_rand.friction_boundary_probability = (
            args.friction_boundary_probability
        )
        env_cfg.domain_rand.randomize_friction = True
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
        setattr(physx, name, value)
    if physx.rest_offset > physx.contact_offset:
        raise ValueError("PhysX rest_offset cannot exceed contact_offset")
    if args.fixed_target_y is not None and args.fixed_target_x is None:
        raise ValueError("--fixed_target_y requires --fixed_target_x")
    commands = env_cfg.commands
    if args.initial_body_command_frame_override is not None:
        if args.initial_body_command_frame_override not in (0, 1):
            raise ValueError(
                "--initial_body_command_frame_override must be 0 or 1"
            )
        commands.initial_body_command_frame = bool(
            args.initial_body_command_frame_override
        )
        print(
            "Initial body command frame override: "
            f"{commands.initial_body_command_frame}"
        )
    if args.initial_robot_yaw_range_deg_override is not None:
        yaw_half_range_deg = args.initial_robot_yaw_range_deg_override
        if yaw_half_range_deg < 0.0 or yaw_half_range_deg > 180.0:
            raise ValueError(
                "--initial_robot_yaw_range_deg_override must be in [0, 180]"
            )
        yaw_half_range = float(np.deg2rad(yaw_half_range_deg))
        ranges.min_ori_euler[2] = -yaw_half_range
        ranges.max_ori_euler[2] = yaw_half_range
        print(
            "Initial robot yaw range override: "
            f"[-{yaw_half_range_deg}, {yaw_half_range_deg}] deg"
        )
    command_range_values = (
        args.command_pos_dx_min_override,
        args.command_pos_dx_max_override,
    )
    if any(value is not None for value in command_range_values):
        command_min = (
            commands.ranges.pos_dx_ini[0]
            if args.command_pos_dx_min_override is None
            else args.command_pos_dx_min_override
        )
        command_max = (
            commands.ranges.pos_dx_ini[1]
            if args.command_pos_dx_max_override is None
            else args.command_pos_dx_max_override
        )
        if command_min < 0.0 or command_max < command_min:
            raise ValueError(
                "command position range must satisfy 0 <= min <= max"
            )
        commands.ranges.pos_dx_ini = [command_min, command_max]
        if hasattr(commands, "endpoint_sampling_max_x"):
            commands.endpoint_sampling_max_x = command_max
        print(
            "Command x range override: [{}, {}], endpoint max={}".format(
                command_min,
                command_max,
                getattr(commands, "endpoint_sampling_max_x", "disabled"),
            )
        )

    command_y_range_values = (
        args.command_pos_dy_min_override,
        args.command_pos_dy_max_override,
    )
    if any(value is not None for value in command_y_range_values):
        command_y_min = (
            commands.ranges.pos_dy_ini[0]
            if args.command_pos_dy_min_override is None
            else args.command_pos_dy_min_override
        )
        command_y_max = (
            commands.ranges.pos_dy_ini[1]
            if args.command_pos_dy_max_override is None
            else args.command_pos_dy_max_override
        )
        if command_y_max < command_y_min:
            raise ValueError("command y range must satisfy min <= max")
        commands.ranges.pos_dy_ini = [command_y_min, command_y_max]
        print("Command y range override: [{}, {}]".format(command_y_min, command_y_max))

    command_yaw_range_values = (
        args.command_yaw_min_deg_override,
        args.command_yaw_max_deg_override,
    )
    if any(value is not None for value in command_yaw_range_values):
        yaw_min_deg = (
            -90.0
            if args.command_yaw_min_deg_override is None
            else args.command_yaw_min_deg_override
        )
        yaw_max_deg = (
            90.0
            if args.command_yaw_max_deg_override is None
            else args.command_yaw_max_deg_override
        )
        if yaw_max_deg < yaw_min_deg:
            raise ValueError("command yaw range must satisfy min <= max")
        commands.randomize_yaw = True
        commands.yaw_range = [
            float(np.deg2rad(yaw_min_deg)),
            float(np.deg2rad(yaw_max_deg)),
        ]
        print("Command yaw range override: [{}, {}] deg".format(yaw_min_deg, yaw_max_deg))

    if args.structured_command_mode_probabilities_override is not None:
        mode_probabilities = [
            float(value)
            for value in args.structured_command_mode_probabilities_override.split(",")
        ]
        if len(mode_probabilities) != 6 or any(value < 0.0 for value in mode_probabilities):
            raise ValueError(
                "structured command probabilities require six non-negative values"
            )
        probability_sum = sum(mode_probabilities)
        if probability_sum <= 0.0:
            raise ValueError("structured command probabilities must have positive sum")
        commands.structured_mode_probabilities = [
            value / probability_sum for value in mode_probabilities
        ]
        print(
            "Structured command mode probabilities: {}".format(
                commands.structured_mode_probabilities
            )
        )
    if args.structured_command_carrier_x_override is not None:
        if args.structured_command_carrier_x_override < 0.0:
            raise ValueError("structured command carrier x must be non-negative")
        commands.structured_carrier_x = args.structured_command_carrier_x_override
    for name, value in (
        ("y_endpoint_sampling_probability", args.command_y_endpoint_probability_override),
        ("yaw_endpoint_sampling_probability", args.command_yaw_endpoint_probability_override),
    ):
        if value is None:
            continue
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        setattr(commands, name, value)

    if args.fixed_target_x is not None:
        target_y = args.fixed_target_y or 0.0
        commands.curriculum = False
        commands.randomize_commands = False
        commands.randomize_yaw = False
        commands.upward_jump_probability = 0.0
        commands.distances.x = args.fixed_target_x
        commands.distances.y = target_y
        commands.distances.z = 0.0
        commands.ranges.pos_dx_ini = [args.fixed_target_x, args.fixed_target_x]
        commands.ranges.pos_dy_ini = [target_y, target_y]
        commands.ranges.pos_dz_ini = [0.0, 0.0]
    push_values = (
        args.push_towards_goal_probability_override,
        args.push_towards_goal_final_probability_override,
    )
    for value in push_values:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("Push-towards-goal probabilities must be in [0, 1]")
    if args.push_towards_goal_probability_override is not None:
        env_cfg.domain_rand.push_towards_goal_probability = (
            args.push_towards_goal_probability_override
        )
    if args.push_towards_goal_final_probability_override is not None:
        env_cfg.domain_rand.push_towards_goal_final_probability = (
            args.push_towards_goal_final_probability_override
        )
    if args.push_towards_goal_anneal_iterations_override is not None:
        if args.push_towards_goal_anneal_iterations_override < 0:
            raise ValueError(
                "--push_towards_goal_anneal_iterations_override must be non-negative"
            )
        env_cfg.domain_rand.push_towards_goal_anneal_iterations = (
            args.push_towards_goal_anneal_iterations_override
        )
    if any(value is not None for value in push_values):
        env_cfg.domain_rand.push_towards_goal = any(
            (value or 0.0) > 0.0 for value in push_values
        )
    if (
        args.com_range_half_width is not None
        and args.com_range_half_width_xyz is not None
    ):
        raise ValueError(
            "Use only one of --com_range_half_width and "
            "--com_range_half_width_xyz"
        )
    if args.com_range_half_width_xyz is not None:
        half_widths = _parse_com_half_width_xyz(args.com_range_half_width_xyz)
        env_cfg.domain_rand.ranges.com_displacement_range = [
            [-width for width in half_widths],
            half_widths,
        ]
        print(
            "Diagnostic xyz COM displacement override: "
            f"lower={env_cfg.domain_rand.ranges.com_displacement_range[0]}, "
            f"upper={env_cfg.domain_rand.ranges.com_displacement_range[1]} m"
        )
    elif args.com_range_half_width is not None:
        half_width = args.com_range_half_width
        if half_width < 0.0:
            raise ValueError("--com_range_half_width must be non-negative")
        env_cfg.domain_rand.ranges.com_displacement_range = [
            -half_width,
            half_width,
        ]
        print(
            "Diagnostic COM displacement override: "
            f"[-{half_width}, {half_width}] m on each axis"
        )
    if args.latency_range_max_ms is not None:
        latency_max_ms = args.latency_range_max_ms
        if latency_max_ms < 0.0:
            raise ValueError("--latency_range_max_ms must be non-negative")
        env_cfg.domain_rand.ranges.latency_range = [0.0, latency_max_ms]
        print(
            "Diagnostic observation-latency override: "
            f"[0.0, {latency_max_ms}] ms"
        )
    if args.latency_range_min_ms is not None:
        latency_min_ms = args.latency_range_min_ms
        latency_max_ms = env_cfg.domain_rand.ranges.latency_range[1]
        if latency_min_ms < 0.0:
            raise ValueError("--latency_range_min_ms must be non-negative")
        if latency_min_ms > latency_max_ms:
            raise ValueError(
                "--latency_range_min_ms must not exceed the latency upper bound"
            )
        env_cfg.domain_rand.ranges.latency_range[0] = latency_min_ms
        print(
            "Diagnostic observation-latency lower-bound override: "
            f"[{latency_min_ms}, {latency_max_ms}] ms"
        )
    zero_latency_probability = (
        env_cfg.domain_rand.zero_latency_probability
        if args.zero_latency_probability_override is None
        else args.zero_latency_probability_override
    )
    max_latency_probability = (
        env_cfg.domain_rand.max_latency_probability
        if args.max_latency_probability_override is None
        else args.max_latency_probability_override
    )
    if not 0.0 <= zero_latency_probability <= 1.0:
        raise ValueError("--zero_latency_probability_override must be in [0, 1]")
    if not 0.0 <= max_latency_probability <= 1.0:
        raise ValueError("--max_latency_probability_override must be in [0, 1]")
    if zero_latency_probability + max_latency_probability > 1.0:
        raise ValueError(
            "zero and max latency probabilities must sum to at most 1"
        )
    env_cfg.domain_rand.zero_latency_probability = zero_latency_probability
    env_cfg.domain_rand.max_latency_probability = max_latency_probability
    if (
        args.zero_latency_probability_override is not None
        or args.max_latency_probability_override is not None
    ):
        print(
            "Diagnostic observation-latency sampling override: "
            f"zero={zero_latency_probability}, max={max_latency_probability}, "
            f"interior={1.0 - zero_latency_probability - max_latency_probability}"
        )
    restitution_min_endpoint_probability = (
        getattr(env_cfg.domain_rand, "restitution_min_endpoint_probability", 0.0)
        if args.restitution_min_endpoint_probability_override is None
        else args.restitution_min_endpoint_probability_override
    )
    restitution_max_endpoint_probability = (
        getattr(env_cfg.domain_rand, "restitution_max_endpoint_probability", 0.0)
        if args.restitution_max_endpoint_probability_override is None
        else args.restitution_max_endpoint_probability_override
    )
    if not 0.0 <= restitution_min_endpoint_probability <= 1.0:
        raise ValueError(
            "--restitution_min_endpoint_probability_override must be in [0, 1]"
        )
    if not 0.0 <= restitution_max_endpoint_probability <= 1.0:
        raise ValueError(
            "--restitution_max_endpoint_probability_override must be in [0, 1]"
        )
    if (
        restitution_min_endpoint_probability
        + restitution_max_endpoint_probability
        > 1.0
    ):
        raise ValueError(
            "restitution endpoint probabilities must sum to at most 1"
        )
    env_cfg.domain_rand.restitution_min_endpoint_probability = (
        restitution_min_endpoint_probability
    )
    env_cfg.domain_rand.restitution_max_endpoint_probability = (
        restitution_max_endpoint_probability
    )
    if (
        args.restitution_min_endpoint_probability_override is not None
        or args.restitution_max_endpoint_probability_override is not None
    ):
        print(
            "Restitution endpoint sampling override: "
            f"min={restitution_min_endpoint_probability}, "
            f"max={restitution_max_endpoint_probability}, "
            "interior="
            f"{1.0 - restitution_min_endpoint_probability - restitution_max_endpoint_probability}"
        )
    joint_friction_min_endpoint_probability = (
        getattr(env_cfg.domain_rand, "joint_friction_min_endpoint_probability", 0.0)
        if args.joint_friction_min_endpoint_probability_override is None
        else args.joint_friction_min_endpoint_probability_override
    )
    joint_friction_max_endpoint_probability = (
        getattr(env_cfg.domain_rand, "joint_friction_max_endpoint_probability", 0.0)
        if args.joint_friction_max_endpoint_probability_override is None
        else args.joint_friction_max_endpoint_probability_override
    )
    if not 0.0 <= joint_friction_min_endpoint_probability <= 1.0:
        raise ValueError(
            "--joint_friction_min_endpoint_probability_override must be in [0, 1]"
        )
    if not 0.0 <= joint_friction_max_endpoint_probability <= 1.0:
        raise ValueError(
            "--joint_friction_max_endpoint_probability_override must be in [0, 1]"
        )
    if (
        joint_friction_min_endpoint_probability
        + joint_friction_max_endpoint_probability
        > 1.0
    ):
        raise ValueError(
            "joint friction endpoint probabilities must sum to at most 1"
        )
    env_cfg.domain_rand.joint_friction_min_endpoint_probability = (
        joint_friction_min_endpoint_probability
    )
    env_cfg.domain_rand.joint_friction_max_endpoint_probability = (
        joint_friction_max_endpoint_probability
    )
    if (
        args.joint_friction_min_endpoint_probability_override is not None
        or args.joint_friction_max_endpoint_probability_override is not None
    ):
        print(
            "Diagnostic joint-friction endpoint sampling override: "
            f"min={joint_friction_min_endpoint_probability}, "
            f"max={joint_friction_max_endpoint_probability}, "
            "interior="
            f"{1.0 - joint_friction_min_endpoint_probability - joint_friction_max_endpoint_probability}"
        )
    domain_range_overrides = (
        (
            args.motor_strength_range_override,
            "--motor_strength_range_override",
            "randomize_motor_strength",
            "motor_strength_ranges",
        ),
        (
            args.restitution_range_override,
            "--restitution_range_override",
            "randomize_restitution",
            "restitution_range",
        ),
        (
            args.joint_friction_range_override,
            "--joint_friction_range_override",
            "randomize_joint_friction",
            "joint_friction_range",
        ),
        (
            args.joint_damping_range_override,
            "--joint_damping_range_override",
            "randomize_joint_damping",
            "joint_damping_range",
        ),
        (
            args.joint_armature_range_override,
            "--joint_armature_range_override",
            "randomize_joint_armature",
            "joint_armature_range",
        ),
        (
            args.physical_dof_velocity_limit_scale_range,
            "--physical_dof_velocity_limit_scale_range",
            "randomize_physical_dof_velocity_limit",
            "physical_dof_velocity_limit_scale_range",
        ),
    )
    for raw_value, option_name, enabled_name, range_name in domain_range_overrides:
        if raw_value is None:
            continue
        try:
            values = [float(value) for value in raw_value.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must contain two numbers: min,max"
            ) from exc
        if len(values) != 2 or values[0] > values[1]:
            raise ValueError(f"{option_name} must be ordered as min,max")
        setattr(env_cfg.domain_rand, enabled_name, True)
        setattr(env_cfg.domain_rand.ranges, range_name, values)
        print(f"Domain randomization override: {range_name}={values}")
    if args.physical_dof_velocity_limit_endpoint_only is not None:
        env_cfg.domain_rand.physical_dof_velocity_limit_endpoint_only = bool(
            args.physical_dof_velocity_limit_endpoint_only
        )
    if args.takeoff_angvel_kick_max_override is not None:
        values = [
            float(value)
            for value in args.takeoff_angvel_kick_max_override.split(",")
        ]
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
    if args.filter_freq_override is not None:
        filter_freq = args.filter_freq_override
        if filter_freq <= 0.0:
            raise ValueError("--filter_freq_override must be positive")
        env_cfg.control.filter_freq = filter_freq
        print(f"Diagnostic action-filter cutoff override: {filter_freq} Hz")
    if args.pos_vel_random_prob_override is not None:
        pos_vel_random_prob = args.pos_vel_random_prob_override
        if not 0.0 <= pos_vel_random_prob <= 1.0:
            raise ValueError(
                "--pos_vel_random_prob_override must be between 0 and 1"
            )
        env_cfg.domain_rand.pos_vel_random_prob = pos_vel_random_prob
        print(
            "Diagnostic random position/velocity start probability override: "
            f"{pos_vel_random_prob}"
        )
    if args.has_jumped_random_prob_override is not None:
        has_jumped_random_prob = args.has_jumped_random_prob_override
        if not 0.0 <= has_jumped_random_prob <= 1.0:
            raise ValueError(
                "--has_jumped_random_prob_override must be between 0 and 1"
            )
        env_cfg.domain_rand.has_jumped_random_prob = has_jumped_random_prob
        env_cfg.domain_rand.randomize_has_jumped = has_jumped_random_prob > 0.0
        print(
            "Diagnostic already-jumped start probability override: "
            f"{has_jumped_random_prob}"
        )
    if args.max_contact_force_override is not None:
        max_contact_force = args.max_contact_force_override
        if max_contact_force <= 0.0:
            raise ValueError("--max_contact_force_override must be positive")
        env_cfg.rewards.max_contact_force = max_contact_force
        print(
            "Experimental maximum foot-contact force override: "
            f"{max_contact_force} N"
        )
    command_level_overrides = {
        "initial_min_level": args.command_initial_min_level,
        "initial_max_level": args.command_initial_max_level,
        "max_level": args.command_max_level,
    }
    for name, value in command_level_overrides.items():
        if value is not None:
            setattr(env_cfg.commands, name, value)
    command_initial_min_level = getattr(env_cfg.commands, "initial_min_level", 0)
    command_initial_max_level = getattr(
        env_cfg.commands,
        "initial_max_level",
        env_cfg.commands.num_levels - 1,
    )
    command_max_level = getattr(
        env_cfg.commands, "max_level", env_cfg.commands.num_levels - 1
    )
    if not (
        0
        <= command_initial_min_level
        <= command_initial_max_level
        <= command_max_level
        < env_cfg.commands.num_levels
    ):
        raise ValueError(
            "command levels must satisfy 0 <= initial min <= initial max "
            "<= max < num_levels"
        )
    if any(value is not None for value in command_level_overrides.values()):
        print(
            "Command curriculum levels: initial=[{}, {}], max={}".format(
                command_initial_min_level,
                command_initial_max_level,
                command_max_level,
            )
        )
    if args.task_pos_scale_override is not None:
        task_pos_scale = args.task_pos_scale_override
        if task_pos_scale < 0.0:
            raise ValueError("--task_pos_scale_override must be non-negative")
        env_cfg.rewards.scales.task_pos = task_pos_scale
        print(f"Experimental task position reward scale override: {task_pos_scale}")
    if args.task_event_require_success_height is not None:
        env_cfg.rewards.task_event_require_success_height = bool(
            args.task_event_require_success_height
        )
        print(
            "Task-event successful-height gate: "
            f"{env_cfg.rewards.task_event_require_success_height}"
        )
    if args.tracking_lin_vel_scale_override is not None:
        tracking_lin_vel_scale = args.tracking_lin_vel_scale_override
        if tracking_lin_vel_scale < 0.0:
            raise ValueError(
                "--tracking_lin_vel_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.tracking_lin_vel = tracking_lin_vel_scale
        print(
            "Experimental linear-velocity tracking reward scale override: "
            f"{tracking_lin_vel_scale}"
        )
    if args.jumping_scale_override is not None:
        jumping_scale = args.jumping_scale_override
        if jumping_scale < 0.0:
            raise ValueError("--jumping_scale_override must be non-negative")
        env_cfg.rewards.scales.jumping = jumping_scale
        print(f"Experimental jumping reward scale override: {jumping_scale}")
    if args.task_max_height_scale_override is not None:
        task_max_height_scale = args.task_max_height_scale_override
        if task_max_height_scale < 0.0:
            raise ValueError(
                "--task_max_height_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.task_max_height = task_max_height_scale
        print(
            "Experimental task maximum-height reward scale override: "
            f"{task_max_height_scale}"
        )
    if args.base_height_flight_scale_override is not None:
        base_height_flight_scale = args.base_height_flight_scale_override
        if base_height_flight_scale < 0.0:
            raise ValueError(
                "--base_height_flight_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.base_height_flight = base_height_flight_scale
        print(
            "Experimental in-flight height reward scale override: "
            f"{base_height_flight_scale}"
        )
    if args.joint_friction_position_reward_boost_override is not None:
        friction_position_boost = (
            args.joint_friction_position_reward_boost_override
        )
        if friction_position_boost < 0.0:
            raise ValueError(
                "--joint_friction_position_reward_boost_override must be "
                "non-negative"
            )
        env_cfg.rewards.joint_friction_position_reward_boost = (
            friction_position_boost
        )
        print(
            "Experimental joint-friction position reward boost override: "
            f"{friction_position_boost}"
        )
    if args.takeoff_roll_rate_scale_override is not None:
        takeoff_roll_rate_scale = args.takeoff_roll_rate_scale_override
        if takeoff_roll_rate_scale > 0.0:
            raise ValueError(
                "--takeoff_roll_rate_scale_override must be non-positive"
            )
        env_cfg.rewards.scales.takeoff_roll_rate = takeoff_roll_rate_scale
        print(
            "Experimental takeoff roll-rate penalty scale override: "
            f"{takeoff_roll_rate_scale}"
        )
    if args.upward_takeoff_quality_scale_override is not None:
        upward_takeoff_quality_scale = args.upward_takeoff_quality_scale_override
        if upward_takeoff_quality_scale < 0.0:
            raise ValueError(
                "--upward_takeoff_quality_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.upward_takeoff_quality = (
            upward_takeoff_quality_scale
        )
        print(
            "Experimental upward takeoff quality reward scale override: "
            f"{upward_takeoff_quality_scale}"
        )
    if args.upward_height_speed_joint_scale_override is not None:
        joint_scale = args.upward_height_speed_joint_scale_override
        if joint_scale < 0.0:
            raise ValueError(
                "--upward_height_speed_joint_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.upward_height_speed_joint = joint_scale
        print(f"Experimental upward height-speed joint reward scale: {joint_scale}")
    if args.upward_height_speed_limit_ratio_override is not None:
        limit_ratio = args.upward_height_speed_limit_ratio_override
        if limit_ratio <= 0.0:
            raise ValueError(
                "--upward_height_speed_limit_ratio_override must be positive"
            )
        env_cfg.rewards.upward_height_speed_limit_ratio = limit_ratio
        print(f"Experimental upward height-speed limit ratio: {limit_ratio}")
    if args.upward_height_speed_height_min_override is not None:
        height_min = args.upward_height_speed_height_min_override
        if height_min <= 0.0:
            raise ValueError(
                "--upward_height_speed_height_min_override must be positive"
            )
        env_cfg.rewards.upward_height_speed_height_min = height_min
    if args.forward_jump_quality_scale_override is not None:
        forward_jump_quality_scale = args.forward_jump_quality_scale_override
        if forward_jump_quality_scale < 0.0:
            raise ValueError(
                "--forward_jump_quality_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.forward_jump_quality = (
            forward_jump_quality_scale
        )
        print(
            "Experimental forward jump quality reward scale override: "
            f"{forward_jump_quality_scale}"
        )
    if args.forward_takeoff_height_floor_scale_override is not None:
        height_floor_scale = args.forward_takeoff_height_floor_scale_override
        if height_floor_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_height_floor_scale_override must be "
                "non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_height_floor = height_floor_scale
        print(
            "Experimental forward takeoff height-floor reward scale override: "
            f"{height_floor_scale}"
        )
    if args.forward_takeoff_yaw_rate_scale_override is not None:
        yaw_rate_scale = args.forward_takeoff_yaw_rate_scale_override
        if yaw_rate_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_yaw_rate_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_yaw_rate = yaw_rate_scale
        print(
            "Experimental forward takeoff yaw-rate reward scale override: "
            f"{yaw_rate_scale}"
        )
    if args.forward_takeoff_joint_scale_override is not None:
        joint_scale = args.forward_takeoff_joint_scale_override
        if joint_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_joint_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_joint = joint_scale
        print(
            "Experimental joint height-position takeoff reward scale override: "
            f"{joint_scale}"
        )
    if args.forward_takeoff_position_scale_override is not None:
        forward_position_scale = args.forward_takeoff_position_scale_override
        if forward_position_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_position_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_position = forward_position_scale
        print(
            "Experimental forward takeoff position reward scale override: "
            f"{forward_position_scale}"
        )
    if args.forward_takeoff_position_coarse_scale_override is not None:
        coarse_position_scale = (
            args.forward_takeoff_position_coarse_scale_override
        )
        if coarse_position_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_position_coarse_scale_override must be "
                "non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_position_coarse = (
            coarse_position_scale
        )
        print(
            "Experimental coarse forward takeoff position reward scale "
            f"override: {coarse_position_scale}"
        )
    if (
        args.forward_takeoff_position_retention_at_max_joint_friction_override
        is not None
    ):
        retention = (
            args.forward_takeoff_position_retention_at_max_joint_friction_override
        )
        if not 0.0 <= retention <= 1.0:
            raise ValueError(
                "--forward_takeoff_position_retention_at_max_joint_friction_override "
                "must be in [0, 1]"
            )
        env_cfg.rewards.forward_takeoff_position_retention_at_max_joint_friction = (
            retention
        )
        print(
            "Experimental takeoff-position retention at maximum joint friction "
            f"override: {retention}"
        )
    if args.task_pos_lateral_scale_override is not None:
        lateral_scale = args.task_pos_lateral_scale_override
        if lateral_scale < 0.0:
            raise ValueError(
                "--task_pos_lateral_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.task_pos_lateral = lateral_scale
        print(
            "Experimental body-frame lateral position reward scale override: "
            f"{lateral_scale}"
        )
    if args.task_pos_lateral_coarse_scale_override is not None:
        lateral_coarse_scale = args.task_pos_lateral_coarse_scale_override
        if lateral_coarse_scale < 0.0:
            raise ValueError(
                "--task_pos_lateral_coarse_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.task_pos_lateral_coarse = lateral_coarse_scale
        print(
            "Experimental coarse body-frame lateral position reward scale "
            f"override: {lateral_coarse_scale}"
        )
    if args.forward_takeoff_lateral_scale_override is not None:
        lateral_scale = args.forward_takeoff_lateral_scale_override
        if lateral_scale < 0.0:
            raise ValueError(
                "--forward_takeoff_lateral_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.forward_takeoff_lateral = lateral_scale
        print(
            "Experimental forward takeoff lateral reward scale override: "
            f"{lateral_scale}"
        )
    if args.initial_contact_history_probability_override is not None:
        probability = args.initial_contact_history_probability_override
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "--initial_contact_history_probability_override must be in [0, 1]"
            )
        env_cfg.env.initial_contact_history_probability = probability
        print(
            "Initial all-feet-contact observation history probability override: "
            f"{probability}"
        )
    if args.initial_contact_settle_steps_override is not None:
        settle_steps = args.initial_contact_settle_steps_override
        if settle_steps < 0:
            raise ValueError(
                "--initial_contact_settle_steps_override must be non-negative"
            )
        env_cfg.env.initial_contact_settle_steps = settle_steps
        print(f"Initial all-feet-contact settle steps override: {settle_steps}")
    if args.initial_contact_settle_steps_max_override is not None:
        settle_steps_max = args.initial_contact_settle_steps_max_override
        settle_steps_min = getattr(env_cfg.env, "initial_contact_settle_steps", 1)
        if settle_steps_max < settle_steps_min:
            raise ValueError(
                "--initial_contact_settle_steps_max_override must be greater "
                "than or equal to initial_contact_settle_steps"
            )
        env_cfg.env.initial_contact_settle_steps_max = settle_steps_max
        print(
            "Initial all-feet-contact settle-step maximum override: "
            f"{settle_steps_max}"
        )
    if args.initial_contact_base_height_offset_override is not None:
        base_height_offset = args.initial_contact_base_height_offset_override
        if base_height_offset < 0.0:
            raise ValueError(
                "--initial_contact_base_height_offset_override must be non-negative"
            )
        env_cfg.env.initial_contact_base_height_offset = base_height_offset
        print(
            "Initial all-feet-contact base height offset override: "
            f"{base_height_offset}"
        )
    if args.base_height_override is not None:
        base_height = args.base_height_override
        if base_height <= 0.0:
            raise ValueError("--base_height_override must be positive")
        env_cfg.init_state.pos[2] = base_height
        print(f"Base reset height override: {base_height}")
    if args.reset_landing_error_override is not None:
        reset_landing_error = args.reset_landing_error_override
        if reset_landing_error <= 0.0:
            raise ValueError("--reset_landing_error_override must be positive")
        env_cfg.env.reset_landing_error = reset_landing_error
        print(f"Landing-error termination override: {reset_landing_error}")
    if args.command_position_observation_scale_override is not None:
        command_scale = args.command_position_observation_scale_override
        if command_scale <= 0.0:
            raise ValueError(
                "--command_position_observation_scale_override must be positive"
            )
        env_cfg.env.command_position_observation_scale = command_scale
        print(f"Command position observation scale override: {command_scale}")
    if args.observe_joint_friction_in_command_y is not None:
        env_cfg.env.observe_joint_friction_in_command_y = bool(
            args.observe_joint_friction_in_command_y
        )
        print(
            "Privileged joint-friction command-y observation: "
            f"{env_cfg.env.observe_joint_friction_in_command_y}"
        )
    if args.takeoff_vz_pitch_quality_scale_override is not None:
        takeoff_vz_pitch_quality_scale = (
            args.takeoff_vz_pitch_quality_scale_override
        )
        if takeoff_vz_pitch_quality_scale < 0.0:
            raise ValueError(
                "--takeoff_vz_pitch_quality_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.takeoff_vz_pitch_quality = (
            takeoff_vz_pitch_quality_scale
        )
    if args.takeoff_predicted_height_sigma_override is not None:
        takeoff_predicted_height_sigma = (
            args.takeoff_predicted_height_sigma_override
        )
        if takeoff_predicted_height_sigma <= 0.0:
            raise ValueError(
                "--takeoff_predicted_height_sigma_override must be positive"
            )
        env_cfg.rewards.takeoff_predicted_height_sigma = (
            takeoff_predicted_height_sigma
        )
        print(
            "Takeoff predicted-height reward sigma override: "
            f"{takeoff_predicted_height_sigma}"
        )
    if args.takeoff_predicted_height_min_override is not None:
        env_cfg.rewards.takeoff_predicted_height_min = (
            args.takeoff_predicted_height_min_override
        )
    if args.takeoff_predicted_height_max_override is not None:
        env_cfg.rewards.takeoff_predicted_height_max = (
            args.takeoff_predicted_height_max_override
        )
    if (
        env_cfg.rewards.takeoff_predicted_height_min
        > env_cfg.rewards.takeoff_predicted_height_max
    ):
        raise ValueError(
            "takeoff predicted-height minimum cannot exceed maximum"
        )
    if args.termination_scale_override is not None:
        termination_scale = args.termination_scale_override
        if termination_scale > 0.0:
            raise ValueError("--termination_scale_override must be non-positive")
        env_cfg.rewards.scales.termination = termination_scale
        print(
            "Experimental termination reward scale override: "
            f"{termination_scale}"
        )
    if args.collision_scale_override is not None:
        collision_scale = args.collision_scale_override
        if collision_scale > 0.0:
            raise ValueError("--collision_scale_override must be non-positive")
        env_cfg.rewards.scales.collision = collision_scale
        print(
            "Experimental penalised-body collision reward scale override: "
            f"{collision_scale}"
        )
    if args.post_landing_ori_scale_override is not None:
        post_landing_ori_scale = args.post_landing_ori_scale_override
        if post_landing_ori_scale < 0.0:
            raise ValueError(
                "--post_landing_ori_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.post_landing_ori = post_landing_ori_scale
        print(
            "Experimental post-landing orientation reward scale override: "
            f"{post_landing_ori_scale}"
        )
    if args.post_landing_pos_scale_override is not None:
        post_landing_pos_scale = args.post_landing_pos_scale_override
        if post_landing_pos_scale < 0.0:
            raise ValueError(
                "--post_landing_pos_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.post_landing_pos = post_landing_pos_scale
        print(
            "Experimental post-landing hold-position reward scale override: "
            f"{post_landing_pos_scale}"
        )
    if args.post_landing_target_pos_scale_override is not None:
        post_landing_target_pos_scale = (
            args.post_landing_target_pos_scale_override
        )
        if post_landing_target_pos_scale < 0.0:
            raise ValueError(
                "--post_landing_target_pos_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.post_landing_target_pos = (
            post_landing_target_pos_scale
        )
        print(
            "Experimental post-landing target-position reward scale override: "
            f"{post_landing_target_pos_scale}"
        )
    if args.symmetric_joints_directional_scale_override is not None:
        directional_scale = args.symmetric_joints_directional_scale_override
        if not 0.0 <= directional_scale <= 1.0:
            raise ValueError(
                "--symmetric_joints_directional_scale_override must be in [0, 1]"
            )
        env_cfg.rewards.symmetric_joints_directional_scale = directional_scale
        print(
            "Directional pre-landing joint-symmetry scale override: "
            f"{directional_scale}"
        )
    negative_reward_overrides = {
        "takeoff_pitch": args.takeoff_pitch_scale_override,
        "takeoff_pitch_rate": args.takeoff_pitch_rate_scale_override,
        "takeoff_pitch_angular_impulse": (
            args.takeoff_pitch_angular_impulse_scale_override
        ),
        "takeoff_pitch_angular_impulse_abs": (
            args.takeoff_pitch_angular_impulse_abs_scale_override
        ),
        "takeoff_pitch_cancellation": (
            args.takeoff_pitch_cancellation_scale_override
        ),
        "takeoff_front_rear_timing": (
            args.takeoff_front_rear_timing_scale_override
        ),
        "takeoff_front_rear_vertical_impulse": (
            args.takeoff_front_rear_vertical_impulse_scale_override
        ),
        "post_landing_positive_vz": args.post_landing_positive_vz_scale_override,
        "post_landing_dof_vel": args.post_landing_dof_vel_scale_override,
        "post_landing_base_xy_vel": (
            args.post_landing_base_xy_vel_scale_override
        ),
        "front_rear_contact_mismatch": (
            args.front_rear_contact_mismatch_scale_override
        ),
        "feet_slip": args.feet_slip_scale_override,
        "action_rate": args.action_rate_scale_override,
        "symmetric_joints": args.symmetric_joints_scale_override,
        "dof_vel_limits": args.dof_vel_limits_scale_override,
        "velocity_torque_envelope_rejection": (
            args.velocity_torque_rejection_scale_override
        ),
        "pre_takeoff_overspeed_drive": (
            args.pre_takeoff_overspeed_drive_scale_override
        ),
        "pre_takeoff_dof_vel_envelope": (
            args.pre_takeoff_dof_vel_envelope_scale_override
        ),
    }
    for reward_name, value in negative_reward_overrides.items():
        if value is None:
            continue
        if value > 0.0:
            raise ValueError(
                f"--{reward_name}_scale_override must be non-positive"
            )
        setattr(env_cfg.rewards.scales, reward_name, value)
        print(f"Experimental {reward_name} reward scale override: {value}")
    if args.post_landing_dof_vel_grace_seconds_override is not None:
        grace_seconds = args.post_landing_dof_vel_grace_seconds_override
        if grace_seconds < 0.0:
            raise ValueError(
                "--post_landing_dof_vel_grace_seconds_override must be "
                "non-negative"
            )
        env_cfg.rewards.post_landing_dof_vel_grace_seconds = grace_seconds
    if args.takeoff_pitch_cancellation_free_band_override is not None:
        if not 0.0 <= args.takeoff_pitch_cancellation_free_band_override <= 1.0:
            raise ValueError(
                "--takeoff_pitch_cancellation_free_band_override must be in [0, 1]"
            )
        env_cfg.rewards.takeoff_pitch_cancellation_free_band = (
            args.takeoff_pitch_cancellation_free_band_override
        )
    if args.takeoff_front_rear_timing_free_band_ms_override is not None:
        if args.takeoff_front_rear_timing_free_band_ms_override < 0.0:
            raise ValueError(
                "--takeoff_front_rear_timing_free_band_ms_override must be non-negative"
            )
        env_cfg.rewards.takeoff_front_rear_timing_free_band = (
            args.takeoff_front_rear_timing_free_band_ms_override / 1000.0
        )
    if args.soft_dof_vel_limit_override is not None:
        if not 0.0 < args.soft_dof_vel_limit_override <= 1.0:
            raise ValueError("--soft_dof_vel_limit_override must be in (0, 1]")
        env_cfg.rewards.soft_dof_vel_limit = args.soft_dof_vel_limit_override
    if args.pre_takeoff_dof_vel_envelope_free_band_override is not None:
        free_band = args.pre_takeoff_dof_vel_envelope_free_band_override
        if free_band <= 0.0:
            raise ValueError(
                "--pre_takeoff_dof_vel_envelope_free_band_override must be positive"
            )
        env_cfg.rewards.pre_takeoff_dof_vel_envelope_free_band = free_band
    if args.pre_takeoff_dof_vel_envelope_full_cost_ratio_override is not None:
        full_cost_ratio = args.pre_takeoff_dof_vel_envelope_full_cost_ratio_override
        free_band = env_cfg.rewards.pre_takeoff_dof_vel_envelope_free_band
        if full_cost_ratio <= free_band:
            raise ValueError(
                "--pre_takeoff_dof_vel_envelope_full_cost_ratio_override must "
                "be greater than the free band"
            )
        env_cfg.rewards.pre_takeoff_dof_vel_envelope_full_cost_ratio = full_cost_ratio
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
    if args.velocity_cost_source is not None:
        env_cfg.control.velocity_cost_source = args.velocity_cost_source
    if args.velocity_cost_free_band is not None:
        env_cfg.control.velocity_cost_free_band = args.velocity_cost_free_band
    if args.velocity_cost_full_ratio is not None:
        env_cfg.control.velocity_cost_full_ratio = args.velocity_cost_full_ratio
    velocity_cost_free_band = getattr(
        env_cfg.control, "velocity_cost_free_band", 0.90
    )
    velocity_cost_full_ratio = getattr(
        env_cfg.control, "velocity_cost_full_ratio", 1.0
    )
    if velocity_cost_free_band < 0.0:
        raise ValueError("--velocity_cost_free_band must be non-negative")
    if velocity_cost_full_ratio <= velocity_cost_free_band:
        raise ValueError(
            "--velocity_cost_full_ratio must exceed the free band"
        )
    if args.velocity_cost_enabled is not None:
        train_cfg.algorithm.velocity_cost_enabled = bool(
            args.velocity_cost_enabled
        )
    if args.velocity_cost_limit is not None:
        if args.velocity_cost_limit < 0.0:
            raise ValueError("--velocity_cost_limit must be non-negative")
        train_cfg.algorithm.velocity_cost_limit = args.velocity_cost_limit
    if args.velocity_cost_dual_lr is not None:
        if args.velocity_cost_dual_lr < 0.0:
            raise ValueError("--velocity_cost_dual_lr must be non-negative")
        train_cfg.algorithm.velocity_cost_dual_lr = args.velocity_cost_dual_lr
    if args.velocity_cost_lambda_init is not None:
        if args.velocity_cost_lambda_init < 0.0:
            raise ValueError("--velocity_cost_lambda_init must be non-negative")
        train_cfg.algorithm.velocity_cost_lambda_init = (
            args.velocity_cost_lambda_init
        )
    if args.velocity_cost_cvar_fraction is not None:
        if not 0.0 < args.velocity_cost_cvar_fraction <= 1.0:
            raise ValueError(
                "--velocity_cost_cvar_fraction must be in (0, 1]"
            )
        train_cfg.algorithm.velocity_cost_cvar_fraction = (
            args.velocity_cost_cvar_fraction
        )
    if args.clip_actions_override is not None:
        if args.clip_actions_override <= 0.0:
            raise ValueError("--clip_actions_override must be positive")
        env_cfg.normalization.clip_actions = args.clip_actions_override
    if args.max_action_delta_override is not None:
        if args.max_action_delta_override <= 0.0:
            raise ValueError("--max_action_delta_override must be positive")
        env_cfg.control.max_action_delta = args.max_action_delta_override
    if args.feet_tuck_activation_height_override is not None:
        activation_height = args.feet_tuck_activation_height_override
        if activation_height <= 0.0:
            raise ValueError(
                "--feet_tuck_activation_height_override must be positive"
            )
        env_cfg.rewards.feet_tuck_activation_height = activation_height
        print(
            "Experimental feet-tuck activation-height override: "
            f"{activation_height} m"
        )
    if args.feet_landing_pose_scale_override is not None:
        landing_pose_scale = args.feet_landing_pose_scale_override
        if landing_pose_scale < 0.0:
            raise ValueError(
                "--feet_landing_pose_scale_override must be non-negative"
            )
        env_cfg.rewards.scales.feet_landing_pose = landing_pose_scale
        print(
            "Experimental descending landing-pose reward scale override: "
            f"{landing_pose_scale}"
        )
    if args.feet_landing_pose_activation_height_override is not None:
        landing_pose_height = args.feet_landing_pose_activation_height_override
        if landing_pose_height <= 0.0:
            raise ValueError(
                "--feet_landing_pose_activation_height_override must be positive"
            )
        env_cfg.rewards.feet_landing_pose_activation_height = landing_pose_height
        print(
            "Experimental descending landing-pose activation-height override: "
            f"{landing_pose_height} m"
        )
    if args.feet_landing_pose_sigma_override is not None:
        landing_pose_sigma = args.feet_landing_pose_sigma_override
        if landing_pose_sigma <= 0.0:
            raise ValueError(
                "--feet_landing_pose_sigma_override must be positive"
            )
        env_cfg.rewards.feet_landing_pose_sigma = landing_pose_sigma
        print(
            "Experimental descending landing-pose reward sigma override: "
            f"{landing_pose_sigma}"
        )
    if args.forward_flight_height_target_override is not None:
        flight_height_target = args.forward_flight_height_target_override
        if flight_height_target <= 0.0:
            raise ValueError(
                "--forward_flight_height_target_override must be positive"
            )
        env_cfg.rewards.forward_flight_height_target = flight_height_target
        print(
            "Experimental forward flight-height target override: "
            f"{flight_height_target} m"
        )
    if args.upward_flight_height_target_override is not None:
        upward_flight_height_target = args.upward_flight_height_target_override
        if upward_flight_height_target <= 0.0:
            raise ValueError(
                "--upward_flight_height_target_override must be positive"
            )
        env_cfg.rewards.upward_flight_height_target = upward_flight_height_target
        print(
            "Experimental upward flight-height target override: "
            f"{upward_flight_height_target} m"
        )
    if args.max_height_target_override is not None:
        max_height_target = args.max_height_target_override
        if max_height_target <= 0.0:
            raise ValueError(
                "--max_height_target_override must be positive"
            )
        env_cfg.rewards.max_height_target = max_height_target
        print(
            "Experimental episode maximum-height target override: "
            f"{max_height_target} m"
        )
    if args.effective_config_output:
        _write_effective_config(args.effective_config_output, args, env_cfg)
    if args.config_only:
        return
    env, env_cfg = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    if args.pure_x_reference_action_coef > 0.0:
        reference_actor = __import__("copy").deepcopy(
            ppo_runner.alg.actor_critic.actor
        ).eval()
        for parameter in reference_actor.parameters():
            parameter.requires_grad_(False)
        ppo_runner.alg.reference_actor = reference_actor
        ppo_runner.alg.reference_action_loss_coef = (
            args.pure_x_reference_action_coef
        )
        if args.task == "solo12_v3_1_upwards":
            if env_cfg.env.num_observations != 1014:
                raise ValueError(
                    "Solo12 upward pre-landing reference preservation expects "
                    "1014 observations"
                )
            # Solo12 upward layout with 20 history frames: historical state
            # [0:920], command[920:933], has_jumped[933], contacts[934:1014].
            ppo_runner.alg.reference_action_pre_landing_only = True
            ppo_runner.alg.reference_action_has_jumped_observation_index = 933
            ppo_runner.alg.reference_action_ignore_command_mask = True
            ppo_runner.alg.reference_action_require_current_contact = True
            ppo_runner.alg.reference_action_current_contact_start_index = 934
            if not 0.0 <= args.pure_x_reference_action_flight_weight <= 1.0:
                raise ValueError(
                    "--pure_x_reference_action_flight_weight must be in [0, 1]"
                )
            ppo_runner.alg.reference_action_flight_weight = (
                args.pure_x_reference_action_flight_weight
            )
        if args.pure_x_reference_max_abs_command_x is not None:
            if args.pure_x_reference_max_abs_command_x < 0.0:
                raise ValueError(
                    "--pure_x_reference_max_abs_command_x must be non-negative"
                )
            ppo_runner.alg.reference_action_max_abs_command_x_observation = (
                args.pure_x_reference_max_abs_command_x
                * env_cfg.env.command_position_observation_scale
            )
        print(
            "Pure-x reference action preservation enabled with coefficient "
            f"{args.pure_x_reference_action_coef}, max |x|="
            f"{args.pure_x_reference_max_abs_command_x} m"
        )
        if args.task == "solo12_v3_1_upwards":
            print(
                "Solo12 upward reference preservation is active only before "
                "the first landing; flight weight="
                f"{args.pure_x_reference_action_flight_weight}"
            )
    if args.reset_optimizer_on_resume:
        ppo_runner.alg.optimizer.state.clear()
        print("Reset resumed optimizer moments; policy/value weights preserved")
    if args.directional_input_adapter_only:
        actor = ppo_runner.alg.actor_critic.actor
        first_linear = next(
            module for module in actor.modules()
            if isinstance(module, torch.nn.Linear)
        )
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        first_linear.weight.requires_grad_(True)
        directional_gradient_mask = torch.zeros_like(first_linear.weight)
        directional_gradient_mask[:, 921] = 1.0  # body-frame y command
        directional_gradient_mask[:, 925] = 1.0  # relative-yaw quaternion z
        first_linear.weight.register_hook(
            lambda gradient: gradient * directional_gradient_mask
        )
        print(
            "Directional input adapter-only training enabled: actor first-layer "
            "columns 921 and 925 are trainable; the original actor is frozen"
        )
    if args.ppo_finetune_learning_rate_override is not None:
        fine_tune_learning_rate = args.ppo_finetune_learning_rate_override
        if fine_tune_learning_rate <= 0.0:
            raise ValueError(
                "--ppo_finetune_learning_rate_override must be positive"
            )
        ppo_runner.alg.learning_rate = fine_tune_learning_rate
        ppo_runner.alg.schedule = "fixed"
        for param_group in ppo_runner.alg.optimizer.param_groups:
            param_group["lr"] = fine_tune_learning_rate
        print(
            "PPO fine-tune learning rate override: "
            f"{fine_tune_learning_rate} (fixed schedule)"
        )
    if args.ppo_finetune_entropy_coef_override is not None:
        fine_tune_entropy_coef = args.ppo_finetune_entropy_coef_override
        if fine_tune_entropy_coef < 0.0:
            raise ValueError(
                "--ppo_finetune_entropy_coef_override must be non-negative"
            )
        ppo_runner.alg.entropy_coef = fine_tune_entropy_coef
        print(
            "PPO fine-tune entropy coefficient override: "
            f"{fine_tune_entropy_coef}"
        )
    if args.ppo_finetune_action_std_override is not None:
        fine_tune_action_std = args.ppo_finetune_action_std_override
        if fine_tune_action_std <= 0.0:
            raise ValueError(
                "--ppo_finetune_action_std_override must be positive"
            )
        with torch.no_grad():
            ppo_runner.alg.actor_critic.std.fill_(fine_tune_action_std)
        print(
            "PPO fine-tune action std override: "
            f"{fine_tune_action_std}"
        )
    if args.ppo_finetune_freeze_action_std:
        ppo_runner.alg.actor_critic.std.requires_grad_(False)
        print("PPO fine-tune action std frozen")
    

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    run_name = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')

    mode = "online"

    wandb.init(project="", name=run_name,  group=args.group_name, mode=mode, dir="../../logs",tags=["task_" + args.task,env.cfg.task_name])


    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args(
        [
            {
                "name": "--upward_retrain_stage",
                "type": str,
                "help": "Frozen upward retraining profile from s1 through s4_latency40",
            },
            {
                "name": "--com_range_half_width",
                "type": float,
                "help": (
                    "Diagnostic-only symmetric COM displacement half-width "
                    "in metres; leaves the registered task config unchanged"
                ),
            },
            {
                "name": "--com_range_half_width_xyz",
                "type": str,
                "help": (
                    "Diagnostic-only symmetric xyz COM displacement half-widths "
                    "in metres, formatted as x,y,z"
                ),
            },
            {
                "name": "--latency_range_max_ms",
                "type": float,
                "help": (
                    "Diagnostic-only observation-latency upper bound in "
                    "milliseconds; leaves the registered task config unchanged"
                ),
            },
            {
                "name": "--latency_range_min_ms",
                "type": float,
                "help": (
                    "Diagnostic-only observation-latency lower bound in "
                    "milliseconds; leaves the registered task config unchanged"
                ),
            },
            {
                "name": "--zero_latency_probability_override",
                "type": float,
                "help": (
                    "Diagnostic-only probability of sampling the exact zero "
                    "observation-latency endpoint"
                ),
            },
            {
                "name": "--max_latency_probability_override",
                "type": float,
                "help": (
                    "Diagnostic-only probability of sampling the exact maximum "
                    "observation-latency endpoint"
                ),
            },
            {
                "name": "--motor_strength_range_override",
                "type": str,
                "help": "Motor-strength multiplier range formatted as min,max",
            },
            {
                "name": "--restitution_range_override",
                "type": str,
                "help": "Rigid-contact restitution range formatted as min,max",
            },
            {
                "name": "--restitution_min_endpoint_probability_override",
                "type": float,
                "help": "Probability of sampling the exact minimum restitution",
            },
            {
                "name": "--restitution_max_endpoint_probability_override",
                "type": float,
                "help": "Probability of sampling the exact maximum restitution",
            },
            {
                "name": "--joint_friction_range_override",
                "type": str,
                "help": "Passive joint-friction range formatted as min,max",
            },
            {
                "name": "--joint_friction_min_endpoint_probability_override",
                "type": float,
                "help": (
                    "Probability of sampling the exact minimum passive "
                    "joint-friction endpoint"
                ),
            },
            {
                "name": "--joint_friction_max_endpoint_probability_override",
                "type": float,
                "help": (
                    "Probability of sampling the exact maximum passive "
                    "joint-friction endpoint"
                ),
            },
            {
                "name": "--joint_damping_range_override",
                "type": str,
                "help": "Passive joint-damping range formatted as min,max",
            },
            {
                "name": "--joint_armature_range_override",
                "type": str,
                "help": "Joint armature range in kg*m^2 formatted as min,max",
            },
            {
                "name": "--physical_dof_velocity_limit_scale_range",
                "type": str,
                "help": "Per-environment physical DOF velocity-limit scale range",
            },
            {
                "name": "--physical_dof_velocity_limit_endpoint_only",
                "type": int,
                "choices": [0, 1],
            },
            {
                "name": "--takeoff_angvel_kick_max_override",
                "type": str,
                "help": "Diagnostic takeoff angular-velocity kick maxima: roll,pitch,yaw",
            },
            {
                "name": "--takeoff_angvel_kick_probability_override",
                "type": float,
            },
            {
                "name": "--takeoff_contact_force_perturb_max_override",
                "type": str,
            },
            {
                "name": "--takeoff_contact_force_perturb_probability_override",
                "type": float,
            },
            {
                "name": "--filter_freq_override",
                "type": float,
                "help": (
                    "Diagnostic-only action-filter cutoff in Hz; leaves the "
                    "registered task config unchanged"
                ),
            },
            {
                "name": "--pos_vel_random_prob_override",
                "type": float,
                "help": "Diagnostic random position/velocity start probability",
            },
            {
                "name": "--has_jumped_random_prob_override",
                "type": float,
                "help": "Diagnostic already-jumped start probability",
            },
            {
                "name": "--max_contact_force_override",
                "type": float,
                "help": "Experimental maximum foot-contact force in newtons",
            },
            {"name": "--command_initial_min_level", "type": int},
            {"name": "--command_initial_max_level", "type": int},
            {"name": "--command_max_level", "type": int},
            {"name": "--task_pos_scale_override", "type": float},
            {
                "name": "--task_event_require_success_height",
                "type": int,
                "choices": [0, 1],
                "help": (
                    "Require real jumps to exceed jump_success_height before "
                    "terminal task and post-landing rewards are credited"
                ),
            },
            {"name": "--tracking_lin_vel_scale_override", "type": float},
            {
                "name": "--ppo_finetune_learning_rate_override",
                "type": float,
                "help": (
                    "Override resumed PPO learning rate and use a fixed "
                    "schedule for fine-tuning"
                ),
            },
            {
                "name": "--reset_optimizer_on_resume",
                "type": int,
                "default": 0,
                "help": (
                    "Clear loaded optimizer moments while preserving resumed "
                    "policy and value weights"
                ),
            },
            {
                "name": "--directional_input_adapter_only",
                "type": int,
                "default": 0,
                "help": (
                    "Train only actor first-layer body-y and relative-yaw-z "
                    "input columns; keep the original actor mapping frozen"
                ),
            },
            {
                "name": "--pure_x_reference_action_coef",
                "type": float,
                "default": 0.0,
                "help": (
                    "MSE coefficient that preserves the resumed actor on "
                    "pure-x observation samples during directional fine-tuning"
                ),
            },
            {
                "name": "--pure_x_reference_action_flight_weight",
                "type": float,
                "default": 0.0,
                "help": (
                    "Relative reference-action loss weight during flight for "
                    "Solo12 upward; stance/takeoff weight is 1"
                ),
            },
            {
                "name": "--pure_x_reference_max_abs_command_x",
                "type": float,
                "help": (
                    "Apply pure-x reference preservation only when the absolute "
                    "body-frame x command is within this trained range in metres"
                ),
            },
            {
                "name": "--ppo_finetune_entropy_coef_override",
                "type": float,
                "help": "Override resumed PPO entropy coefficient",
            },
            {
                "name": "--ppo_finetune_action_std_override",
                "type": float,
                "help": "Override resumed PPO action standard deviation",
            },
            {
                "name": "--ppo_finetune_freeze_action_std",
                "action": "store_true",
                "help": "Freeze PPO action standard deviation during fine-tuning",
            },
            {
                "name": "--joint_friction_position_reward_boost_override",
                "type": float,
                "help": (
                    "Linearly increase the true landing-position reward "
                    "toward the maximum passive joint-friction endpoint"
                ),
            },
            {
                "name": "--takeoff_roll_rate_scale_override",
                "type": float,
                "help": (
                    "One-shot first-takeoff penalty for body roll rate above "
                    "the configured free band"
                ),
            },
            {
                "name": "--jumping_scale_override",
                "type": float,
                "help": "Experimental override for the existing binary jumping reward scale",
            },
            {
                "name": "--task_max_height_scale_override",
                "type": float,
                "help": "Experimental task maximum-height reward scale",
            },
            {
                "name": "--base_height_flight_scale_override",
                "type": float,
                "help": "Experimental in-flight height reward scale",
            },
            {
                "name": "--upward_takeoff_quality_scale_override",
                "type": float,
                "help": (
                    "Experimental terminal reward for jointly satisfying "
                    "upward height and takeoff-pitch targets"
                ),
            },
            {
                "name": "--forward_jump_quality_scale_override",
                "type": float,
                "help": (
                    "Experimental terminal reward for jointly satisfying "
                    "a real forward jump, landing position, and height band"
                ),
            },
            {
                "name": "--forward_takeoff_height_floor_scale_override",
                "type": float,
                "help": "Immediate forward takeoff reward for reaching a minimum predicted apex",
            },
            {
                "name": "--forward_takeoff_yaw_rate_scale_override",
                "type": float,
                "help": "One-shot commanded yaw-rate reward at first takeoff",
            },
            {
                "name": "--forward_takeoff_joint_scale_override",
                "type": float,
                "help": "Multiplicative takeoff reward for height and coarse position",
            },
            {
                "name": "--forward_takeoff_position_scale_override",
                "type": float,
                "help": "Immediate command-position reward from the takeoff state",
            },
            {
                "name": "--forward_takeoff_position_coarse_scale_override",
                "type": float,
                "help": (
                    "Broad linear command-position reward from the takeoff "
                    "state"
                ),
            },
            {
                "name": "--forward_takeoff_position_retention_at_max_joint_friction_override",
                "type": float,
                "help": (
                    "Override the commanded-distance fraction retained at the "
                    "maximum passive joint-friction endpoint"
                ),
            },
            {
                "name": "--task_pos_lateral_scale_override",
                "type": float,
                "help": "Dense body-frame lateral position reward scale",
            },
            {
                "name": "--task_pos_lateral_coarse_scale_override",
                "type": float,
                "help": "Broad dense body-frame lateral position reward scale",
            },
            {
                "name": "--forward_takeoff_lateral_scale_override",
                "type": float,
                "help": "One-shot body-frame lateral landing reward at takeoff",
            },
            {
                "name": "--initial_contact_history_probability_override",
                "type": float,
                "help": (
                    "Probability of initializing reset contact observation "
                    "history as all-feet contact"
                ),
            },
            {
                "name": "--contact_observation_delay_max_steps_override",
                "type": int,
                "help": (
                    "Diagnostic per-foot episodic contact-observation delay; "
                    "sampled uniformly from 0..N policy steps"
                ),
            },
            {
                "name": "--initial_contact_settle_steps_override",
                "type": int,
                "help": (
                    "Zero-action control steps used to physically settle "
                    "all-feet-contact reset environments"
                ),
            },
            {
                "name": "--initial_contact_settle_steps_max_override",
                "type": int,
                "help": (
                    "Inclusive maximum zero-action settle steps; values are "
                    "sampled per reset from minimum..maximum"
                ),
            },
            {
                "name": "--initial_contact_base_height_offset_override",
                "type": float,
                "help": (
                    "Downward base-height offset used to establish physical "
                    "foot contact for all-feet-contact reset environments"
                ),
            },
            {
                "name": "--base_height_override",
                "type": float,
                "help": "Override init_state.pos[2] in metres",
            },
            {
                "name": "--reset_landing_error_override",
                "type": float,
                "help": "Override forward landing-error termination threshold",
            },
            {
                "name": "--command_position_observation_scale_override",
                "type": float,
                "help": "Override position-command scale in actor observations",
            },
            {
                "name": "--observe_joint_friction_in_command_y",
                "type": int,
                "help": "Validation-only privileged friction condition in the unused forward command-y slot",
            },
            {
                "name": "--command_pos_dx_min_override",
                "type": float,
                "help": "Override the minimum sampled forward position command",
            },
            {
                "name": "--initial_body_command_frame_override",
                "type": int,
                "help": (
                    "Diagnostic command-frame override: 1 for initial body "
                    "frame, 0 for the legacy world frame"
                ),
            },
            {
                "name": "--initial_robot_yaw_range_deg_override",
                "type": float,
                "help": "Override the symmetric reset yaw range in degrees",
            },
            {
                "name": "--command_pos_dx_max_override",
                "type": float,
                "help": "Override the maximum sampled forward position command",
            },
            {
                "name": "--command_pos_dy_min_override",
                "type": float,
                "help": "Override the minimum sampled body-frame lateral command",
            },
            {
                "name": "--command_pos_dy_max_override",
                "type": float,
                "help": "Override the maximum sampled body-frame lateral command",
            },
            {
                "name": "--command_yaw_min_deg_override",
                "type": float,
                "help": "Override the minimum sampled relative-yaw command in degrees",
            },
            {
                "name": "--command_yaw_max_deg_override",
                "type": float,
                "help": "Override the maximum sampled relative-yaw command in degrees",
            },
            {
                "name": "--structured_command_mode_probabilities_override",
                "type": str,
                "help": (
                    "Six comma-separated probabilities for pure_x, carrier_y, "
                    "pure_y, carrier_yaw, pure_yaw, joint_xyyaw"
                ),
            },
            {
                "name": "--structured_command_carrier_x_override",
                "type": float,
                "help": "Body-frame x command used by carrier_y and carrier_yaw modes",
            },
            {
                "name": "--command_y_endpoint_probability_override",
                "type": float,
                "help": "Probability of replacing sampled y by either range endpoint",
            },
            {
                "name": "--command_yaw_endpoint_probability_override",
                "type": float,
                "help": "Probability of replacing sampled yaw by either range endpoint",
            },
            {
                "name": "--initial_stance_height_range_override",
                "type": str,
                "help": (
                    "Solo12 correlated deployment-stance reset-height range "
                    "formatted as min,max"
                ),
            },
            {
                "name": "--initial_stance_height_probability_override",
                "type": float,
                "help": "Probability of applying a correlated stance reset",
            },
            {
                "name": "--post_landing_hind_width_scale_override",
                "type": float,
                "help": (
                    "Non-positive scale for the post-landing narrow-hind-stance "
                    "penalty"
                ),
            },
            {
                "name": "--post_landing_hind_width_min_override",
                "type": float,
                "help": "Free lower bound for post-landing hind-foot width in metres",
            },
            {
                "name": "--post_landing_hind_width_grace_seconds_override",
                "type": float,
                "default": 0.2,
                "help": "Delay after first real landing before applying hind-width penalty",
            },
            {
                "name": "--post_landing_hind_width_max_collapse_override",
                "type": float,
                "help": "Maximum free hind-width reduction after first real touchdown",
            },
            {
                "name": "--post_landing_hind_width_window_seconds_override",
                "type": float,
                "default": 0.4,
                "help": "Finite evaluation window after the landing grace period",
            },
            {
                "name": "--post_landing_hind_width_contact_loss_weight_override",
                "type": float,
                "default": 0.5,
            },
            {
                "name": "--post_landing_hind_width_joint_velocity_weight_override",
                "type": float,
                "default": 0.1,
            },
            {
                "name": "--post_landing_hind_width_angular_velocity_weight_override",
                "type": float,
                "default": 0.1,
            },
            {
                "name": "--post_landing_hind_width_slip_weight_override",
                "type": float,
                "default": 0.25,
            },
            {
                "name": "--post_landing_hind_width_collapse_weight_override",
                "type": float,
                "default": 0.0,
            },
            {
                "name": "--post_landing_hind_width_collapse_free_override",
                "type": float,
                "default": 0.01,
            },
            {
                "name": "--post_landing_hind_width_displacement_weight_override",
                "type": float,
                "default": 0.0,
            },
            {
                "name": "--post_landing_hind_width_displacement_free_override",
                "type": float,
                "default": 0.015,
            },
            {
                "name": "--takeoff_vz_pitch_quality_scale_override",
                "type": float,
            },
            {
                "name": "--takeoff_predicted_height_sigma_override",
                "type": float,
            },
            {
                "name": "--takeoff_predicted_height_min_override",
                "type": float,
            },
            {
                "name": "--takeoff_predicted_height_max_override",
                "type": float,
            },
            {
                "name": "--termination_scale_override",
                "type": float,
                "help": "Experimental non-positive termination reward scale",
            },
            {
                "name": "--collision_scale_override",
                "type": float,
                "help": (
                    "Experimental non-positive thigh/calf collision reward "
                    "scale"
                ),
            },
            {"name": "--post_landing_ori_scale_override", "type": float},
            {"name": "--post_landing_pos_scale_override", "type": float},
            {
                "name": "--post_landing_target_pos_scale_override",
                "type": float,
                "help": (
                    "Dense post-landing reward coupling first-touchdown "
                    "accuracy to surviving recovery"
                ),
            },
            {"name": "--takeoff_pitch_scale_override", "type": float},
            {"name": "--takeoff_pitch_rate_scale_override", "type": float},
            {
                "name": "--takeoff_pitch_angular_impulse_scale_override",
                "type": float,
            },
            {
                "name": "--takeoff_pitch_angular_impulse_abs_scale_override",
                "type": float,
            },
            {"name": "--takeoff_pitch_cancellation_scale_override", "type": float},
            {"name": "--takeoff_front_rear_timing_scale_override", "type": float},
            {
                "name": "--takeoff_front_rear_vertical_impulse_scale_override",
                "type": float,
            },
            {
                "name": "--takeoff_pitch_cancellation_free_band_override",
                "type": float,
            },
            {
                "name": "--takeoff_front_rear_timing_free_band_ms_override",
                "type": float,
            },
            {"name": "--post_landing_positive_vz_scale_override", "type": float},
            {"name": "--post_landing_dof_vel_scale_override", "type": float},
            {
                "name": "--post_landing_base_xy_vel_scale_override",
                "type": float,
            },
            {
                "name": "--post_landing_dof_vel_grace_seconds_override",
                "type": float,
            },
            {
                "name": "--front_rear_contact_mismatch_scale_override",
                "type": float,
            },
            {"name": "--feet_slip_scale_override", "type": float},
            {"name": "--action_rate_scale_override", "type": float},
            {
                "name": "--symmetric_joints_scale_override",
                "type": float,
                "help": (
                    "Diagnostic non-positive override for the existing "
                    "left/right joint-symmetry penalty"
                ),
            },
            {
                "name": "--symmetric_joints_directional_scale_override",
                "type": float,
                "help": (
                    "Scale the existing joint-symmetry penalty before first "
                    "landing for nonzero lateral commands"
                ),
            },
            {"name": "--dof_vel_limits_scale_override", "type": float},
            {"name": "--velocity_torque_rejection_scale_override", "type": float},
            {
                "name": "--pre_takeoff_overspeed_drive_scale_override",
                "type": float,
            },
            {
                "name": "--pre_takeoff_dof_vel_envelope_scale_override",
                "type": float,
            },
            {
                "name": "--pre_takeoff_dof_vel_envelope_free_band_override",
                "type": float,
            },
            {
                "name": "--pre_takeoff_dof_vel_envelope_full_cost_ratio_override",
                "type": float,
            },
            {
                "name": "--upward_height_speed_joint_scale_override",
                "type": float,
            },
            {
                "name": "--upward_height_speed_limit_ratio_override",
                "type": float,
            },
            {
                "name": "--upward_height_speed_height_min_override",
                "type": float,
            },
            {"name": "--soft_dof_vel_limit_override", "type": float},
            {"name": "--physical_dof_velocity_limit_override", "type": float},
            {"name": "--physical_dof_velocity_limit_scale", "type": float},
            {"name": "--velocity_torque_limit_scale", "type": float},
            {"name": "--velocity_torque_envelope_blend", "type": float},
            {
                "name": "--velocity_cost_source",
                "type": str,
                "choices": [
                    "envelope_rejection",
                    "pre_takeoff_overspeed_drive",
                    "pre_takeoff_true_overspeed_drive",
                    "takeoff_tail_overspeed",
                ],
            },
            {"name": "--velocity_cost_free_band", "type": float},
            {"name": "--velocity_cost_full_ratio", "type": float},
            {
                "name": "--velocity_torque_envelope",
                "type": int,
                "choices": [0, 1],
            },
            {
                "name": "--velocity_cost_enabled",
                "type": int,
                "choices": [0, 1],
            },
            {"name": "--velocity_cost_limit", "type": float},
            {"name": "--velocity_cost_dual_lr", "type": float},
            {"name": "--velocity_cost_lambda_init", "type": float},
            {"name": "--velocity_cost_cvar_fraction", "type": float},
            {"name": "--save_interval_override", "type": int},
            {"name": "--upward_action_reference_csv", "type": str},
            {
                "name": "--upward_action_reference_scale_override",
                "type": float,
            },
            {
                "name": "--upward_action_reference_sigma_override",
                "type": float,
            },
            {"name": "--clip_actions_override", "type": float},
            {"name": "--max_action_delta_override", "type": float},
            {
                "name": "--recovery_state_clone_probability",
                "type": float,
                "default": 0.0,
                "help": (
                    "Probability that a reset starts from a same-environment "
                    "real post-landing state with closed-loop histories"
                ),
            },
            {
                "name": "--recovery_state_clone_max_post_landing_steps",
                "type": int,
                "default": 5,
                "help": "Latest post-landing policy step retained in replay",
            },
            {
                "name": "--recovery_state_clone_min_height",
                "type": float,
                "default": 0.52,
                "help": "Minimum source-episode apex height for replay",
            },
            {
                "name": "--settled_contact_count_override",
                "type": int,
                "help": "Required contacting feet for the settled state",
            },
            {
                "name": "--feet_tuck_activation_height_override",
                "type": float,
                "help": (
                    "Experimental base-height threshold above which the "
                    "in-flight tucked-feet penalty is active"
                ),
            },
            {
                "name": "--feet_landing_pose_scale_override",
                "type": float,
                "help": "Experimental descending landing-pose reward scale",
            },
            {
                "name": "--feet_landing_pose_activation_height_override",
                "type": float,
                "help": (
                    "Experimental base-height threshold below which the "
                    "descending landing-pose reward is active"
                ),
            },
            {
                "name": "--feet_landing_pose_sigma_override",
                "type": float,
                "help": "Experimental descending landing-pose reward sigma",
            },
            {
                "name": "--forward_flight_height_target_override",
                "type": float,
                "help": "Experimental forward flight-height reward target in metres",
            },
            {
                "name": "--upward_flight_height_target_override",
                "type": float,
                "help": "Experimental upward flight-height reward target in metres",
            },
            {
                "name": "--max_height_target_override",
                "type": float,
                "help": "Experimental episode maximum-height reward target in metres",
            },
            {
                "name": "--friction_range_override",
                "type": str,
                "help": "Broad rigid-contact friction range formatted as min,max",
            },
            {
                "name": "--friction_boundary_range",
                "type": str,
                "help": "High-value friction boundary range formatted as min,max",
            },
            {
                "name": "--friction_boundary_probability",
                "type": float,
                "help": "Probability of sampling from the friction boundary range",
            },
            {"name": "--physx_contact_offset_override", "type": float},
            {"name": "--physx_rest_offset_override", "type": float},
            {
                "name": "--physx_friction_offset_threshold_override",
                "type": float,
            },
            {"name": "--fixed_target_x", "type": float},
            {"name": "--fixed_target_y", "type": float},
            {
                "name": "--push_towards_goal_probability_override",
                "type": float,
            },
            {
                "name": "--push_towards_goal_final_probability_override",
                "type": float,
            },
            {
                "name": "--push_towards_goal_anneal_iterations_override",
                "type": int,
            },
            {"name": "--effective_config_output", "type": str},
            {
                "name": "--config_only",
                "action": "store_true",
                "default": False,
            },
        ]
    )
    train(args)
