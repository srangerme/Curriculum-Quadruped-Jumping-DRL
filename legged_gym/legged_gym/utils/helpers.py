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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from datetime import datetime

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    runs = []
    for run in os.listdir(root):
        if run == 'exported' or not os.path.isdir(os.path.join(root, run)):
            continue
        try:
            timestamp = datetime.strptime(run[:16], "%b%d_%H-%M-%S_")
        except ValueError:
            continue
        runs.append((timestamp, run))
    if not runs:
        raise ValueError("No runs in this directory: " + root)
    runs.sort(key=lambda item: item[0])
    last_run = os.path.join(root, runs[-1][1])
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        jump_distance = getattr(args, "jump_distance", None)
        jump_distance_y = getattr(args, "jump_distance_y", None)
        jump_yaw_deg = getattr(args, "jump_yaw_deg", None)
        if jump_distance is not None or jump_distance_y is not None or jump_yaw_deg is not None:
            fixed_x = 0.0 if jump_distance is None else jump_distance
            fixed_y = 0.0 if jump_distance_y is None else jump_distance_y
            if fixed_x < 0.0:
                raise ValueError("--jump_distance must be non-negative")
            if env_cfg.env.jump_type != "forward":
                raise ValueError("Fixed jump commands are only valid for forward-jump tasks")
            env_cfg.commands.curriculum = False
            env_cfg.commands.randomize_commands = True
            env_cfg.commands.randomize_yaw = False
            if hasattr(env_cfg.commands, "mixed_short_command_sampling"):
                env_cfg.commands.mixed_short_command_sampling = False
            if hasattr(env_cfg.commands, "balanced_command_sampling"):
                # Fixed-command evaluation must bypass every higher-priority
                # command sampler. Otherwise the balanced sampler still
                # assigns most environments to zero/lateral/yaw buckets even
                # though the requested x/y ranges have been collapsed.
                env_cfg.commands.balanced_command_sampling = False
            env_cfg.commands.upward_jump_probability = 0.0
            env_cfg.commands.jump_over_box = False
            env_cfg.commands.ranges.pos_dx_ini = [fixed_x, fixed_x]
            env_cfg.commands.ranges.pos_dy_ini = [fixed_y, fixed_y]
            env_cfg.commands.ranges.pos_dz_ini = [0.0, 0.0]
            env_cfg.commands.distances.des_yaw = (
                None if jump_yaw_deg is None else np.deg2rad(jump_yaw_deg)
            )
        goal_push_probability = getattr(args, "goal_push_probability", None)
        if goal_push_probability is not None:
            probability = goal_push_probability
            if not 0.0 <= probability <= 1.0:
                raise ValueError("--goal_push_probability must be between 0 and 1")
            env_cfg.domain_rand.push_towards_goal = probability > 0.0
            env_cfg.domain_rand.push_towards_goal_probability = probability
            env_cfg.domain_rand.push_towards_goal_final_probability = probability
            env_cfg.domain_rand.push_towards_goal_anneal_start_iteration = 0
            env_cfg.domain_rand.push_towards_goal_anneal_iterations = 0
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
        if args.reset_optimizer:
            cfg_train.runner.load_optimizer = False

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go1", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        {"name": "--reset_optimizer", "action": "store_true", "default": False, "help": "Load policy/value weights without restoring optimizer state."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--group_name", "type": str, "default": "standard", "help": "Name of the wandb group"},
        {"name": "--jump_distance", "type": float, "help": "Evaluate/train a fixed forward jump distance in metres."},
        {"name": "--jump_distance_y", "type": float, "help": "Evaluate a fixed lateral jump displacement in metres."},
        {"name": "--jump_yaw_deg", "type": float, "help": "Evaluate a fixed absolute landing yaw in degrees."},
        {"name": "--goal_push_probability", "type": float, "help": "Override goal-directed take-off velocity injection probability in [0, 1]."},
        {"name": "--post_landing_view_seconds", "type": float, "default": 0.0, "help": "In play, keep the robot visible for this many seconds after first landing before reset."},
        {"name": "--landing_stability_seconds", "type": float, "default": 0.0, "help": "In play, evaluate stable standing for this many seconds after first landing."},
        {"name": "--landing_contact_grace_seconds", "type": float, "default": 0.25, "help": "Grace period after first contact before enforcing four-foot contact."},
        {"name": "--landing_min_all_feet_contact_ratio", "type": float, "default": 0.95, "help": "Minimum four-foot-contact duty ratio during stable-standing evaluation."},
        {"name": "--landing_max_leg_torque_cv", "type": float, "default": 0.25, "help": "Maximum coefficient of variation of mean absolute torque load across the four legs."},
        {"name": "--landing_min_success_rate", "type": float, "default": 0.95, "help": "Minimum stable-landing episode success rate."},
        {"name": "--enforce_landing_stability", "action": "store_true", "default": False, "help": "Exit play with failure when the stable-landing acceptance thresholds are not met."},
        {"name": "--deterministic_eval", "action": "store_true", "default": False, "help": "Disable evaluation-time randomization without overriding the jump command."},
        {"name": "--eval_with_randomization", "action": "store_true", "default": False, "help": "Keep the task's configured randomization while evaluating fixed jump commands."},
        {"name": "--eval_disable_latency", "action": "store_true", "default": False, "help": "In play, disable latency while retaining other configured randomization."},
        {"name": "--eval_disable_observation_noise", "action": "store_true", "default": False, "help": "In play, disable observation noise while retaining other configured randomization."},
        {"name": "--eval_disable_initial_state_rand", "action": "store_true", "default": False, "help": "In play, disable initial-state randomization while retaining other configured randomization."},
        {"name": "--eval_disable_actuator_rand", "action": "store_true", "default": False, "help": "In play, disable actuator randomization while retaining other configured randomization."},
        {"name": "--eval_disable_dynamics_rand", "action": "store_true", "default": False, "help": "In play, disable dynamics/contact randomization while retaining other configured randomization."},
        {"name": "--eval_disable_inertial_rand", "action": "store_true", "default": False, "help": "In play, disable mass/COM/link-mass randomization while retaining other configured randomization."},
        {"name": "--eval_disable_contact_rand", "action": "store_true", "default": False, "help": "In play, disable contact/joint resistance randomization while retaining other configured randomization."},
        {"name": "--eval_disable_surface_rand", "action": "store_true", "default": False, "help": "In play, disable ground friction/restitution randomization while retaining other configured randomization."},
        {"name": "--eval_disable_joint_resistance_rand", "action": "store_true", "default": False, "help": "In play, disable joint friction/damping/armature randomization while retaining other configured randomization."},
        {"name": "--eval_has_jumped_mode", "type": str, "default": "configured", "help": "In play, use configured, false, or true initial has_jumped state."},
        {"name": "--eval_has_jumped_reset_step", "type": int, "default": 30, "help": "Policy step at which a forced true has_jumped state resets to false."},
        {"name": "--eval_latency_ms", "type": float, "help": "In play, fix observation latency in milliseconds and disable latency jitter."},
        {"name": "--eval_added_mass", "type": float, "help": "In play, fix added base mass in kilograms."},
        {"name": "--eval_link_mass_scale", "type": float, "help": "In play, fix every non-base link mass scale."},
        {"name": "--eval_joint_friction", "type": float, "help": "In play, fix joint friction."},
        {"name": "--eval_joint_damping", "type": float, "help": "In play, fix joint damping."},
        {"name": "--eval_pushes", "action": "store_true", "default": False, "help": "Keep the task's external robot pushes enabled during play."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path,policy_name='1'):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_' + policy_name + '.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
