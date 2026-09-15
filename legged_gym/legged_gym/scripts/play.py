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
import csv
import os
import time

import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import get_euler_xyz, quat_mul
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.model_interface import export_model_interface_config

import numpy as np
import torch


def play(args):
    if args.play_episodes <= 0:
        raise ValueError("--play_episodes must be positive")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = (
        args.num_envs
        if args.num_envs is not None
        else min(env_cfg.env.num_envs, 50)
    )
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.domain_rand.push_upwards = False
    if args.nominal_physics:
        for flag in (
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
            "randomize_joint_armature",
        ):
            if hasattr(env_cfg.domain_rand, flag):
                setattr(env_cfg.domain_rand, flag, False)
        env_cfg.domain_rand.pos_vel_random_prob = 0.0
        env_cfg.domain_rand.has_jumped_random_prob = 0.0
        env_cfg.domain_rand.sim_latency = False
        env_cfg.domain_rand.sim_pd_latency = False
    if args.fixed_joint_armature is not None:
        if args.fixed_joint_armature < 0.0:
            raise ValueError("--fixed_joint_armature must be non-negative")
        env_cfg.domain_rand.randomize_joint_armature = True
        env_cfg.domain_rand.ranges.joint_armature_range = [
            args.fixed_joint_armature,
            args.fixed_joint_armature,
        ]
    if args.physical_dof_velocity_limit_override is not None:
        if args.physical_dof_velocity_limit_override <= 0.0:
            raise ValueError(
                "--physical_dof_velocity_limit_override must be positive"
            )
        env_cfg.asset.physical_dof_velocity_limit_override = (
            args.physical_dof_velocity_limit_override
        )
    if args.filter_freq_override is not None:
        if args.filter_freq_override <= 0.0:
            raise ValueError("--filter_freq_override must be positive")
        env_cfg.control.filter_freq = args.filter_freq_override
    if args.velocity_torque_envelope is not None:
        env_cfg.control.velocity_torque_envelope = bool(
            args.velocity_torque_envelope
        )
    if args.fixed_friction is not None:
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.ranges.friction_range = [
            args.fixed_friction,
            args.fixed_friction,
        ]
    if args.episode_length is not None:
        env_cfg.env.episode_length_s = args.episode_length
    if args.base_height_override is not None:
        if args.base_height_override <= 0.0:
            raise ValueError("--base_height_override must be positive")
        env_cfg.init_state.pos[2] = args.base_height_override
    if args.initial_contact_history_probability is not None:
        probability = args.initial_contact_history_probability
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "--initial_contact_history_probability must be in [0, 1]"
            )
        env_cfg.env.initial_contact_history_probability = probability
    if args.initial_contact_settle_steps is not None:
        if args.initial_contact_settle_steps < 0:
            raise ValueError("--initial_contact_settle_steps must be non-negative")
        env_cfg.env.initial_contact_settle_steps = args.initial_contact_settle_steps
    if args.initial_contact_base_height_offset is not None:
        env_cfg.env.initial_contact_base_height_offset = (
            args.initial_contact_base_height_offset
        )
    for arg_name, config_name in (
        ("camera_pos_override", "pos"),
        ("camera_lookat_override", "lookat"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            components = [float(component) for component in value.split(",")]
            if len(components) != 3:
                raise ValueError(f"--{arg_name} must contain three comma-separated values")
            setattr(env_cfg.viewer, config_name, components)
    env_cfg.commands.ranges.lin_vel_x = [0.1,0.1]
    env_cfg.commands.ranges.lin_vel_y = [0.0,0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.2,0.2]
    env_cfg.commands.ranges.heading = [0.0,0.0]

    if args.jump_distance is not None:
        # A fixed forward-jump target is a position command, not lin_vel_x.
        env_cfg.commands.curriculum = False
        # Use the same command recomputation path as training, with a
        # degenerate range to make the sampled target deterministic.
        env_cfg.commands.randomize_commands = True
        env_cfg.commands.upward_jump_probability = 0.0
        env_cfg.commands.randomize_yaw = False
        # distances.x is an additive offset applied after sampling.
        env_cfg.commands.distances.x = 0.0
        env_cfg.commands.distances.y = 0.0
        env_cfg.commands.distances.z = 0.0
        env_cfg.commands.distances.des_yaw = 0.0
        env_cfg.commands.ranges.pos_dx_ini = [args.jump_distance, args.jump_distance]
        env_cfg.commands.ranges.pos_dy_ini = [0.0, 0.0]
        env_cfg.commands.ranges.pos_dz_ini = [0.0, 0.0]

    if args.track_robot:
        # The tracking camera uses the close offset defined in LeggedRobot.step().
        env_cfg.viewer.camera_track_robot = True
    if args.offscreen_camera:
        env_cfg.viewer.simulate_camera = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if args.offscreen_camera:
        robot_origin = env.root_states[0, :3].detach().cpu().numpy()
        camera_position = robot_origin + np.asarray(env_cfg.viewer.pos)
        camera_lookat = robot_origin + np.asarray(env_cfg.viewer.lookat)
        env.gym.set_camera_location(
            env.camera_handle,
            env.envs[0],
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_lookat),
        )
    if args.jump_distance is not None:
        # The environment computes its initial command during construction.
        # Synchronize both the persistent offset and the already-created
        # command tensor so media playback uses the requested fixed target.
        env.cfg.commands.randomize_commands = False
        env.command_distances["x"] = args.jump_distance
        env.commands[:, 0] = args.jump_distance
        print(f"Fixed forward-jump command: {env.commands[0, 0].item():.3f} m")
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
        if env_cfg.asset.policy_dof_names and env_cfg.asset.policy_foot_names:
            interface_path = export_model_interface_config(env_cfg, path)
            print('Exported model interface to: ', interface_path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    frames_dir = None
    if args.record_frames:
        frames_dir = args.frames_dir
        if frames_dir is None:
            frames_dir = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name,
                'exported',
                'frames',
            )
        elif not os.path.isabs(frames_dir):
            frames_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        print(f'Recording viewer frames: {frames_dir}')

    csv_file = None
    csv_writer = None
    if args.csv_log is not None:
        csv_path = args.csv_log
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                'logs',
                train_cfg.runner.experiment_name,
                'exported',
                'telemetry',
                csv_path,
            )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if os.path.exists(csv_path):
            stem, suffix = os.path.splitext(csv_path)
            duplicate_index = 1
            while os.path.exists(f"{stem}_{duplicate_index}{suffix}"):
                duplicate_index += 1
            csv_path = f"{stem}_{duplicate_index}{suffix}"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        joint_names = list(env.policy_dof_names)
        csv_header = [
            'step',
            'sim_time_s',
            'unix_time_ns',
            'episode_step',
            'done',
            'has_jumped',
            'takeoff_event',
            'takeoff_sim_time_s',
            'takeoff_unix_time_ns',
        ]
        for quantity, unit in (
            ('joint_position', 'rad'),
            ('joint_velocity', 'rad_s'),
            ('applied_torque', 'nm'),
            ('model_output', 'unitless'),
        ):
            csv_header.extend(
                f'{quantity}_{joint_name}_{unit}' for joint_name in joint_names
            )
        csv_header.extend(
            f'policy_observation_{index:03d}'
            for index in range(obs.shape[-1])
        )
        csv_header.extend(
            f'foot_contact_force_{foot_name}_world_{axis}_n'
            for foot_name in env.policy_foot_names
            for axis in ('x', 'y', 'z')
        )
        csv_header.extend(
            f'foot_contact_state_{foot_name}'
            for foot_name in env.policy_foot_names
        )
        csv_header.extend([
            'body_linear_velocity_x_m_s',
            'body_linear_velocity_y_m_s',
            'world_linear_velocity_x_m_s',
            'world_linear_velocity_y_m_s',
            'imu_accel_x_m_s2',
            'imu_accel_y_m_s2',
            'imu_accel_z_m_s2',
            'imu_roll_deg',
            'imu_pitch_deg',
            'imu_yaw_deg',
            'body_roll_deg',
            'body_pitch_deg',
            'body_yaw_deg',
            'body_angular_velocity_x_rad_s',
            'body_angular_velocity_y_rad_s',
            'body_angular_velocity_z_rad_s',
        ])
        csv_writer.writerow(csv_header)
        csv_file.flush()
        print(f'Telemetry CSV: {csv_path}')

    # action_multipliers = [env.cfg.control.hip_scale_multip,env.cfg.control.thigh_scale_multip,env.cfg.control.calf_scale_multip]
    # action_multiplier = action_multipliers[joint_index%3]
    

    completed_episodes = 0
    takeoff_recorded = False
    takeoff_sim_time_s = None
    takeoff_unix_time_ns = None
    max_play_steps = args.play_episodes * (int(env.max_episode_length) + 1)
    for i in range(max_play_steps):
        policy_observation = obs.detach()
        actions = policy(policy_observation)
        obs, _, rews, dones, infos = env.step(actions.detach())
        row_unix_time_ns = time.time_ns()
        takeoff_event = bool(
            env.settled_after_init[robot_index].item()
            and env.was_in_flight[robot_index].item()
            and env.mid_air[robot_index].item()
            and not takeoff_recorded
        )
        if takeoff_event:
            takeoff_recorded = True
            takeoff_sim_time_s = (i + 1) * env.dt
            takeoff_unix_time_ns = row_unix_time_ns
        if csv_writer is not None:
            joint_position = env._dof_to_policy(
                env.dof_pos[robot_index]
            ).detach().cpu().tolist()
            joint_velocity = env._dof_to_policy(
                env.dof_vel[robot_index]
            ).detach().cpu().tolist()
            applied_torque = env._dof_to_policy(
                env.torques_to_apply[robot_index]
            ).detach().cpu().tolist()
            model_output = actions[robot_index].detach().cpu().tolist()
            policy_observation_values = policy_observation[
                robot_index
            ].cpu().tolist()
            foot_contact_forces = env.contact_forces[
                robot_index, env.feet_indices, :
            ].detach().cpu().reshape(-1).tolist()
            foot_contact_states = env.contacts[
                robot_index
            ].detach().cpu().to(torch.int).tolist()
            body_linear_velocity = env.base_lin_vel[
                robot_index, :2
            ].detach().cpu().tolist()
            world_linear_velocity = env.root_states[
                robot_index, 7:9
            ].detach().cpu().tolist()
            imu_accel = env.imu_state[robot_index].detach().cpu().tolist()

            imu_quat = quat_mul(
                env.base_quat[robot_index:robot_index + 1],
                env.imu_quaternion_to_base.unsqueeze(0),
            )
            imu_euler = torch.stack(get_euler_xyz(imu_quat), dim=-1)[0]
            imu_euler = torch.remainder(imu_euler + np.pi, 2 * np.pi) - np.pi
            imu_euler_deg = torch.rad2deg(imu_euler).detach().cpu().tolist()
            body_euler_deg = torch.rad2deg(
                env.euler[robot_index]
            ).detach().cpu().tolist()
            body_angular_velocity = env.base_ang_vel[
                robot_index
            ].detach().cpu().tolist()

            csv_writer.writerow([
                i,
                i * env.dt,
                row_unix_time_ns,
                int(env.episode_length_buf[robot_index].item()),
                int(dones[robot_index].item()),
                int(env.has_jumped[robot_index].item()),
                int(takeoff_event),
                '' if takeoff_sim_time_s is None else takeoff_sim_time_s,
                '' if takeoff_unix_time_ns is None else takeoff_unix_time_ns,
                *joint_position,
                *joint_velocity,
                *applied_torque,
                *model_output,
                *policy_observation_values,
                *foot_contact_forces,
                *foot_contact_states,
                *body_linear_velocity,
                *world_linear_velocity,
                *imu_accel,
                *imu_euler_deg,
                *body_euler_deg,
                *body_angular_velocity,
            ])
            if i % 100 == 0:
                csv_file.flush()
        if args.record_frames:
            if i % 2:
                filename = os.path.join(frames_dir, f"{img_idx:06d}.png")
                if args.offscreen_camera:
                    robot_xy = (
                        env.root_states[0, :2] - env.env_origins[0, :2]
                    ).detach().cpu().numpy()
                    camera_position = np.asarray(env_cfg.viewer.pos).copy()
                    camera_lookat = np.asarray(env_cfg.viewer.lookat).copy()
                    camera_position[:2] += robot_xy
                    camera_lookat[:2] += robot_xy
                    if img_idx == 0:
                        print(
                            "Offscreen camera robot/camera/lookat: "
                            f"{env.root_states[0, :3].detach().cpu().tolist()} / "
                            f"{camera_position.tolist()} / "
                            f"{camera_lookat.tolist()}"
                        )
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
                        filename,
                    )
                else:
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

        if dones[robot_index].item():
            completed_episodes += 1
            if completed_episodes >= args.play_episodes:
                break
            takeoff_recorded = False
            takeoff_sim_time_s = None
            takeoff_unix_time_ns = None

    if csv_file is not None:
        csv_file.close()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args(additional_parameters=[
        {
            "name": "--jump_distance",
            "type": float,
            "default": None,
            "help": "Fix the forward-jump target distance in metres.",
        },
        {
            "name": "--track_robot",
            "action": "store_true",
            "default": False,
            "help": "Use a close camera that follows environment 0.",
        },
        {
            "name": "--episode_length",
            "type": float,
            "default": None,
            "help": "Override the episode length in seconds.",
        },
        {
            "name": "--base_height_override",
            "type": float,
            "default": None,
            "help": "Override init_state.pos[2] in metres.",
        },
        {
            "name": "--initial_contact_history_probability",
            "type": float,
            "default": None,
            "help": "Probability of initializing history as all-feet contact.",
        },
        {
            "name": "--initial_contact_settle_steps",
            "type": int,
            "default": None,
            "help": "Override initial all-feet-contact physics settle steps.",
        },
        {
            "name": "--initial_contact_base_height_offset",
            "type": float,
            "default": None,
            "help": "Override initial-contact base-height offset in metres.",
        },
        {
            "name": "--camera_pos_override",
            "type": str,
            "default": None,
            "help": "Viewer camera position as comma-separated x,y,z.",
        },
        {
            "name": "--camera_lookat_override",
            "type": str,
            "default": None,
            "help": "Viewer camera look-at point as comma-separated x,y,z.",
        },
        {
            "name": "--csv_log",
            "type": str,
            "default": None,
            "help": "Write environment-0 telemetry to this CSV filename.",
        },
        {
            "name": "--play_episodes",
            "type": int,
            "default": 10,
            "help": "Stop after environment 0 completes this many episodes.",
        },
        {
            "name": "--nominal_physics",
            "action": "store_true",
            "default": False,
            "help": "Disable all physics and initial-state randomization.",
        },
        {
            "name": "--fixed_friction",
            "type": float,
            "default": None,
            "help": "Use one fixed terrain friction coefficient.",
        },
        {"name": "--fixed_joint_armature", "type": float},
        {"name": "--physical_dof_velocity_limit_override", "type": float},
        {"name": "--filter_freq_override", "type": float},
        {
            "name": "--velocity_torque_envelope",
            "type": int,
            "choices": [0, 1],
        },
        {
            "name": "--record_frames",
            "action": "store_true",
            "default": False,
            "help": "Write every second viewer frame as PNG.",
        },
        {
            "name": "--offscreen_camera",
            "action": "store_true",
            "default": False,
            "help": "Render frames with a headless camera sensor.",
        },
        {
            "name": "--frames_dir",
            "type": str,
            "default": None,
            "help": "Absolute or legged_gym/logs-relative frame directory.",
        },
    ])
    play(args)
