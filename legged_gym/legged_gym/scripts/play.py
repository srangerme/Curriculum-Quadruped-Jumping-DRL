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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.post_landing_view_seconds < 0.0:
        raise ValueError("--post_landing_view_seconds must be non-negative")
    if args.landing_stability_seconds < 0.0:
        raise ValueError("--landing_stability_seconds must be non-negative")
    if args.landing_contact_grace_seconds < 0.0:
        raise ValueError("--landing_contact_grace_seconds must be non-negative")
    if not 0.0 <= args.landing_min_all_feet_contact_ratio <= 1.0:
        raise ValueError("--landing_min_all_feet_contact_ratio must be in [0, 1]")
    if args.landing_max_leg_torque_cv < 0.0:
        raise ValueError("--landing_max_leg_torque_cv must be non-negative")
    if not 0.0 <= args.landing_min_success_rate <= 1.0:
        raise ValueError("--landing_min_success_rate must be in [0, 1]")
    if args.enforce_landing_stability and args.landing_stability_seconds <= 0.0:
        raise ValueError(
            "--enforce_landing_stability requires --landing_stability_seconds"
        )
    # Evaluation-only landing hold. Training keeps the default value of zero,
    # so its termination distribution is unchanged.
    stability_view_seconds = (
        args.landing_contact_grace_seconds + args.landing_stability_seconds
    )
    env_cfg.env.post_landing_view_seconds = max(
        args.post_landing_view_seconds, stability_view_seconds
    )
    env_cfg.env.landing_stability_seconds = args.landing_stability_seconds
    env_cfg.env.landing_contact_grace_seconds = args.landing_contact_grace_seconds
    env_cfg.env.landing_min_all_feet_contact_ratio = (
        args.landing_min_all_feet_contact_ratio
    )
    env_cfg.env.landing_max_leg_torque_cv = args.landing_max_leg_torque_cv
    if args.landing_stability_seconds > 0.0:
        # The normal four-second episode can expire before a full three-second
        # post-landing window. Extend evaluation only; training is unchanged.
        env_cfg.env.episode_length_s += stability_view_seconds
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    if args.deterministic_eval and args.eval_with_randomization:
        raise ValueError("--deterministic_eval and --eval_with_randomization are mutually exclusive")
    fixed_jump_command = (
        args.jump_distance is not None
        or args.jump_distance_y is not None
        or args.jump_yaw_deg is not None
    )
    if not args.eval_with_randomization:
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.randomize_friction = False
        if not args.eval_pushes:
            env_cfg.domain_rand.push_robots = False
    # Recovery-state resets are a training curriculum, never part of jump
    # checkpoint acceptance (including randomized sim2sim-style evaluation).
    if hasattr(env_cfg.domain_rand, "landing_recovery_state_probability"):
        env_cfg.domain_rand.landing_recovery_state_probability = 0.0
    deterministic_eval = args.deterministic_eval or (
        fixed_jump_command and not args.eval_with_randomization
    )
    if deterministic_eval:
        # Deterministic fixed-distance diagnostics make checkpoint-to-checkpoint
        # comparisons meaningful. These overrides affect evaluation only.
        env_cfg.env.continuous_jumping = False
        env_cfg.env.continuous_jumping_reset_probability = 0.0
        env_cfg.domain_rand.randomize_robot_pos = False
        env_cfg.domain_rand.randomize_robot_vel = False
        env_cfg.domain_rand.randomize_robot_ori = False
        env_cfg.domain_rand.randomize_dof_pos = False
        env_cfg.domain_rand.randomize_spring_params = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_PD_gains = False
        env_cfg.domain_rand.randomize_has_jumped = False
        env_cfg.domain_rand.reset_has_jumped = False
        env_cfg.domain_rand.randomize_motor_offset = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_joint_damping = False
        env_cfg.domain_rand.randomize_joint_armature = False
        env_cfg.domain_rand.sim_latency = False
        env_cfg.domain_rand.sim_pd_latency = False
    # Ablation switches for attributing failures under full randomised
    # evaluation.  They never affect training and can be combined.
    if args.eval_disable_latency:
        env_cfg.domain_rand.sim_latency = False
        env_cfg.domain_rand.sim_pd_latency = False
    if args.eval_disable_observation_noise:
        env_cfg.noise.add_noise = False
    if args.eval_disable_initial_state_rand:
        env_cfg.domain_rand.randomize_robot_pos = False
        env_cfg.domain_rand.randomize_robot_vel = False
        env_cfg.domain_rand.randomize_robot_ori = False
        env_cfg.domain_rand.randomize_dof_pos = False
        env_cfg.domain_rand.randomize_spring_params = False
    if args.eval_disable_actuator_rand:
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_PD_gains = False
        env_cfg.domain_rand.randomize_motor_offset = False
    if args.eval_disable_dynamics_rand:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_link_mass = False
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_joint_damping = False
        env_cfg.domain_rand.randomize_joint_armature = False
    if args.eval_disable_inertial_rand:
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com = False
        env_cfg.domain_rand.randomize_link_mass = False
    if args.eval_disable_contact_rand:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_joint_damping = False
        env_cfg.domain_rand.randomize_joint_armature = False
    if args.eval_disable_surface_rand:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
    if args.eval_disable_joint_resistance_rand:
        env_cfg.domain_rand.randomize_joint_friction = False
        env_cfg.domain_rand.randomize_joint_damping = False
        env_cfg.domain_rand.randomize_joint_armature = False
    # Evaluation-only controlled interventions. Ranges are collapsed instead
    # of disabling the sampler so paired runs retain the same random-number
    # consumption as closely as possible.
    has_jumped_mode = args.eval_has_jumped_mode.lower()
    if has_jumped_mode not in ("configured", "false", "true"):
        raise ValueError("--eval_has_jumped_mode must be configured, false, or true")
    if args.eval_has_jumped_reset_step < 1:
        raise ValueError("--eval_has_jumped_reset_step must be at least 1")
    if has_jumped_mode == "false":
        # Keep the Bernoulli draw in the reset path for paired comparisons,
        # but force every draw to False.
        env_cfg.domain_rand.randomize_has_jumped = True
        env_cfg.domain_rand.has_jumped_random_prob = 0.0
        env_cfg.domain_rand.reset_has_jumped = True
    elif has_jumped_mode == "true":
        env_cfg.domain_rand.randomize_has_jumped = True
        env_cfg.domain_rand.has_jumped_random_prob = 1.0
        env_cfg.domain_rand.reset_has_jumped = True
        env_cfg.domain_rand.manual_has_jumped_reset_time = args.eval_has_jumped_reset_step
    if args.eval_latency_ms is not None:
        if args.eval_latency_ms < 0.0:
            raise ValueError("--eval_latency_ms must be non-negative")
        env_cfg.domain_rand.sim_latency = args.eval_latency_ms > 0.0
        env_cfg.domain_rand.ranges.latency_range = [args.eval_latency_ms, args.eval_latency_ms]
        env_cfg.domain_rand.ranges.additional_latency_range = [0.0, 0.0]
    if args.eval_added_mass is not None:
        env_cfg.domain_rand.randomize_base_mass = True
        env_cfg.domain_rand.ranges.added_mass_range = [args.eval_added_mass, args.eval_added_mass]
    if args.eval_link_mass_scale is not None:
        if args.eval_link_mass_scale <= 0.0:
            raise ValueError("--eval_link_mass_scale must be positive")
        env_cfg.domain_rand.randomize_link_mass = True
        env_cfg.domain_rand.ranges.added_link_mass_range = [
            args.eval_link_mass_scale,
            args.eval_link_mass_scale,
        ]
    if args.eval_joint_friction is not None:
        if args.eval_joint_friction < 0.0:
            raise ValueError("--eval_joint_friction must be non-negative")
        env_cfg.domain_rand.randomize_joint_friction = True
        env_cfg.domain_rand.ranges.joint_friction_range = [
            args.eval_joint_friction,
            args.eval_joint_friction,
        ]
    if args.eval_joint_damping is not None:
        if args.eval_joint_damping < 0.0:
            raise ValueError("--eval_joint_damping must be non-negative")
        env_cfg.domain_rand.randomize_joint_damping = True
        env_cfg.domain_rand.ranges.joint_damping_range = [
            args.eval_joint_damping,
            args.eval_joint_damping,
        ]
    # Use a close evaluation view while leaving the camera under manual control.
    env_cfg.viewer.pos = [5.0, -1.4, 0.65]
    env_cfg.viewer.lookat = [5.0, 0.0, 0.35]
    env_cfg.commands.ranges.lin_vel_x = [0.1,0.1]
    env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.2,0.2]
    env_cfg.commands.ranges.heading = [0.0,0.0]
            

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.headless:
        robot_position = env.root_states[0, :3].detach().cpu().numpy()
        camera_position = robot_position + np.array([0.5, -0.8, 0.25])
        camera_target = robot_position + np.array([0.0, 0.0, 0.05])
        env.set_camera(camera_position.tolist(), camera_target.tolist())
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = -1 if args.headless else 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # action_multipliers = [env.cfg.control.hip_scale_multip,env.cfg.control.thigh_scale_multip,env.cfg.control.calf_scale_multip]
    # action_multiplier = action_multipliers[joint_index%3]
    

    rollout_episodes = 2 if fixed_jump_command else 10
    for i in range(rollout_episodes*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    # 'dof_pos_target': actions[robot_index, joint_index].item() * 0.5 * action_multiplier + env.default_dof_pos.cpu().numpy()[0,joint_index],
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

    if args.enforce_landing_stability:
        success_rate = logger.get_mean("jump_stable_standing_success")
        contact_ratio = logger.get_mean("jump_post_landing_all_feet_contact_ratio")
        final_contact_rate = logger.get_mean(
            "jump_post_landing_final_all_feet_contact"
        )
        torque_cv = logger.get_mean("jump_post_landing_leg_torque_cv")
        leg_torques = {
            leg: logger.get_mean(f"jump_post_landing_leg_torque_{leg}")
            for leg in ("fl", "fr", "rl", "rr")
        }
        natural_failure_rate = logger.get_mean(
            "jump_post_landing_natural_failure"
        )
        required = {
            "success_rate": success_rate,
            "contact_ratio": contact_ratio,
            "final_contact_rate": final_contact_rate,
            "torque_cv": torque_cv,
            "natural_failure_rate": natural_failure_rate,
        }
        if any(value is None for value in required.values()):
            raise RuntimeError(
                "Stable-landing diagnostics were not produced; no completed "
                "landing evaluation was observed"
            )
        passed = (
            success_rate >= args.landing_min_success_rate
            and contact_ratio >= args.landing_min_all_feet_contact_ratio
            and final_contact_rate >= args.landing_min_success_rate
            and torque_cv <= args.landing_max_leg_torque_cv
            and natural_failure_rate == 0.0
        )
        print(
            "Landing stability acceptance: "
            f"success={success_rate:.3f}, "
            f"four_feet_ratio={contact_ratio:.3f}, "
            f"final_four_feet={final_contact_rate:.3f}, "
            f"leg_torque_cv={torque_cv:.3f}, "
            "leg_torque_mean=["
            + ", ".join(
                f"{leg.upper()}={value:.3f}" if value is not None else f"{leg.upper()}=n/a"
                for leg, value in leg_torques.items()
            )
            + "], "
            f"natural_failure={natural_failure_rate:.3f} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            raise SystemExit(2)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
