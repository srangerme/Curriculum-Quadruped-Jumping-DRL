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

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 velocity_cost_enabled=False,
                 velocity_cost_limit=0.0,
                 velocity_cost_dual_lr=0.01,
                 velocity_cost_lambda_init=0.0,
                 velocity_cost_value_loss_coef=1.0,
                 velocity_cost_cvar_fraction=1.0,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.velocity_cost_enabled = velocity_cost_enabled
        self.velocity_cost_limit = velocity_cost_limit
        self.velocity_cost_dual_lr = velocity_cost_dual_lr
        self.velocity_cost_lambda = max(0.0, velocity_cost_lambda_init)
        self.velocity_cost_value_loss_coef = velocity_cost_value_loss_coef
        self.velocity_cost_cvar_fraction = velocity_cost_cvar_fraction
        self.cost_critic = None
        optimizer_parameters = list(self.actor_critic.parameters())
        if self.velocity_cost_enabled:
            self.cost_critic = copy.deepcopy(self.actor_critic.critic).to(self.device)
            optimizer_parameters += list(self.cost_critic.parameters())
        self.optimizer = optim.Adam(optimizer_parameters, lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        if self.velocity_cost_enabled:
            self.transition.cost_values = self.cost_critic(critic_obs).detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if self.velocity_cost_enabled:
            velocity_cost = infos.get("velocity_limit_cost")
            if velocity_cost is None:
                raise RuntimeError(
                    "velocity cost PPO requires infos['velocity_limit_cost']"
                )
            self.transition.costs = velocity_cost.detach().clone()
        # Bootstrapping on time outs
        # if 'time_outs' in infos:
        #     self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        if self.velocity_cost_enabled:
            last_cost_values = self.cost_critic(last_critic_obs).detach()
            self.storage.compute_cost_returns(
                last_cost_values, self.gamma, self.lam
            )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_reference_action_loss = 0
        mean_reference_action_mask_fraction = 0
        mean_cost_surrogate_loss = 0
        mean_cost_value_loss = 0
        velocity_cost_tail_return_threshold = None
        if self.velocity_cost_enabled:
            flat_velocity_costs = self.storage.costs.flatten()
            if self.velocity_cost_cvar_fraction < 1.0:
                tail_count = max(
                    1,
                    int(
                        math.ceil(
                            flat_velocity_costs.numel()
                            * self.velocity_cost_cvar_fraction
                        )
                    ),
                )
                velocity_cost_mean = torch.topk(
                    flat_velocity_costs, tail_count
                ).values.mean().item()
                flat_cost_returns = self.storage.cost_returns.flatten()
                return_tail_count = max(
                    1,
                    int(
                        math.ceil(
                            flat_cost_returns.numel()
                            * self.velocity_cost_cvar_fraction
                        )
                    ),
                )
                velocity_cost_tail_return_threshold = torch.topk(
                    flat_cost_returns, return_tail_count
                ).values.min()
            else:
                velocity_cost_mean = flat_velocity_costs.mean().item()
            self.velocity_cost_lambda = max(
                0.0,
                self.velocity_cost_lambda
                + self.velocity_cost_dual_lr
                * (velocity_cost_mean - self.velocity_cost_limit),
            )
        else:
            velocity_cost_mean = 0.0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, costs_batch, cost_values_batch, cost_returns_batch, cost_advantages_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                cost_surrogate_loss = torch.zeros((), device=self.device)
                cost_value_loss = torch.zeros((), device=self.device)
                if self.velocity_cost_enabled:
                    cost_surrogate = torch.squeeze(cost_advantages_batch) * ratio
                    cost_surrogate_clipped = torch.squeeze(
                        cost_advantages_batch
                    ) * torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    cost_surrogate_terms = torch.max(
                        cost_surrogate, cost_surrogate_clipped
                    )
                    if velocity_cost_tail_return_threshold is not None:
                        tail_mask = (
                            torch.squeeze(cost_returns_batch).detach()
                            >= velocity_cost_tail_return_threshold
                        ).to(cost_surrogate_terms.dtype)
                        cost_surrogate_loss = torch.sum(
                            cost_surrogate_terms * tail_mask
                        ) / torch.clamp(torch.sum(tail_mask), min=1.0)
                    else:
                        cost_surrogate_loss = cost_surrogate_terms.mean()
                    predicted_cost_values = self.cost_critic(critic_obs_batch)
                    cost_value_loss = torch.mean(
                        torch.square(predicted_cost_values - cost_returns_batch)
                    )
                reference_action_loss = torch.zeros((), device=self.device)
                reference_action_mask_fraction = torch.zeros(
                    (), device=self.device
                )
                reference_actor = getattr(self, "reference_actor", None)
                reference_coef = getattr(
                    self, "reference_action_loss_coef", 0.0
                )
                if reference_actor is not None and reference_coef > 0.0:
                    # Anchor only the previously learned pure-x behaviour.
                    # Observation 921 is the body-frame lateral command and
                    # 925 is the z component of the relative-yaw quaternion.
                    # Checking only y also anchored yaw commands and prevented
                    # the policy from learning non-zero target yaw.
                    if getattr(
                        self, "reference_action_ignore_command_mask", False
                    ):
                        pure_x_mask = torch.ones(
                            obs_batch.shape[0],
                            dtype=torch.bool,
                            device=obs_batch.device,
                        )
                    else:
                        pure_x_mask = (
                            (torch.abs(obs_batch[:, 921]) < 1e-6)
                            & (torch.abs(obs_batch[:, 925]) < 1e-6)
                        )
                    reference_max_abs_command_x = getattr(
                        self,
                        "reference_action_max_abs_command_x_observation",
                        None,
                    )
                    if reference_max_abs_command_x is not None:
                        pure_x_mask &= (
                            torch.abs(obs_batch[:, 920])
                            <= reference_max_abs_command_x + 1e-6
                        )
                    reference_pre_landing_only = getattr(
                        self, "reference_action_pre_landing_only", False
                    )
                    if reference_pre_landing_only:
                        has_jumped_index = getattr(
                            self,
                            "reference_action_has_jumped_observation_index",
                            None,
                        )
                        if has_jumped_index is None:
                            raise RuntimeError(
                                "Pre-landing reference preservation requires "
                                "a has-jumped observation index"
                            )
                        pure_x_mask &= obs_batch[:, has_jumped_index] < 0.5
                    reference_action_weights = pure_x_mask.float()
                    if getattr(
                        self, "reference_action_require_current_contact", False
                    ):
                        contact_start = getattr(
                            self,
                            "reference_action_current_contact_start_index",
                            None,
                        )
                        if contact_start is None:
                            raise RuntimeError(
                                "Contact-gated reference preservation requires "
                                "a contact observation index"
                            )
                        current_contact_mask = torch.any(
                            obs_batch[:, contact_start:contact_start + 4] > 0.5,
                            dim=1,
                        )
                        flight_weight = getattr(
                            self, "reference_action_flight_weight", 0.0
                        )
                        reference_action_weights *= torch.where(
                            current_contact_mask,
                            torch.ones_like(reference_action_weights),
                            torch.full_like(reference_action_weights, flight_weight),
                        )
                        pure_x_mask = reference_action_weights > 0.0
                    reference_action_mask_fraction = (
                        reference_action_weights.mean()
                    )
                    if torch.any(pure_x_mask):
                        with torch.no_grad():
                            reference_mu = reference_actor(obs_batch)
                        reference_action_error = torch.mean(
                            torch.square(mu_batch - reference_mu), dim=1
                        )
                        reference_action_loss = torch.sum(
                            reference_action_error * reference_action_weights
                        ) / torch.clamp(
                            torch.sum(reference_action_weights), min=1e-6
                        )
                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + reference_coef * reference_action_loss
                    + self.velocity_cost_lambda * cost_surrogate_loss
                    + self.velocity_cost_value_loss_coef * cost_value_loss
                )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy_loss += entropy_batch.mean().item()
                mean_reference_action_loss += reference_action_loss.item()
                mean_reference_action_mask_fraction += (
                    reference_action_mask_fraction.item()
                )
                mean_cost_surrogate_loss += cost_surrogate_loss.item()
                mean_cost_value_loss += cost_value_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_reference_action_loss /= num_updates
        mean_reference_action_mask_fraction /= num_updates
        self.last_reference_action_loss = mean_reference_action_loss
        self.last_reference_action_mask_fraction = (
            mean_reference_action_mask_fraction
        )
        self.last_velocity_cost_mean = velocity_cost_mean
        self.last_velocity_cost_lambda = self.velocity_cost_lambda
        self.last_cost_surrogate_loss = mean_cost_surrogate_loss / num_updates
        self.last_cost_value_loss = mean_cost_value_loss / num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss
