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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
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


def train(args):
    # args.headless = True
    # args.resume = True
    # args.load_run = 'Mar02_17-00-01_'
    # args.num_envs = 2
    env_cfg, _ = task_registry.get_cfgs(args.task)
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
    env, env_cfg = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    run_name = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_')

    mode = "online"

    wandb.init(project="", name=run_name,  group=args.group_name, mode=mode, dir="../../logs",tags=["task_" + args.task,env.cfg.task_name])


    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args(
        [
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
        ]
    )
    train(args)
