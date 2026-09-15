"""On-policy post-landing state replay for jump recovery training.

The replay bank is deliberately per environment.  A snapshot is restored only
to the environment that produced it, so fixed-at-creation physics properties
(mass, COM, shape material and physical velocity limit) remain consistent.
"""

from __future__ import annotations

from typing import Dict

import torch
from isaacgym import gymtorch


# State needed to continue the closed loop from a real landing.  Observation
# history tensors are added by suffix in _snapshot_tensor_names().
_TENSOR_STATE_NAMES = (
    "root_states",
    "dof_pos",
    "dof_vel",
    "base_quat",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "base_lin_vel_imu",
    "feet_pos",
    "feet_vel",
    "feet_acc",
    "contacts",
    "last_contacts",
    "contact_filt",
    "contact_filt_prev",
    "ori_error",
    "euler",
    "dof_acc",
    "dof_jerk",
    "base_acc",
    "base_acc_prev",
    "imu_state",
    "actions",
    "actions_scaled",
    "actions_filtered",
    "last_actions",
    "last_last_actions",
    "last_dof_vel",
    "last_dof_acc",
    "last_feet_vel",
    "last_root_vel",
    "torques",
    "torques_to_apply",
    "requested_torques_pre_envelope",
    "commands",
    "command_vels",
    "initial_root_states",
    "initial_root_states_nonrandomised",
    "initial_foot_poses",
    "landing_poses",
    "landing_foot_poses",
    "max_height",
    "min_height",
    "was_in_flight",
    "mid_air",
    "has_jumped",
    "settled_after_init",
    "settled_after_init_timer",
    "_has_jumped_switched_time",
    "_has_jumped_rand_envs",
    "not_pushed",
    "post_landing_positive_vz_seen",
    "post_landing_positive_vz_steps",
    "episodic_latency",
    "episodic_pd_latency",
    "sampled_observation_latency",
    "sampled_pd_latency",
    "prev_sampled_latency",
    "contact_observation_delay_steps",
    "contact_observation_delay_buffer",
    "motor_strengths",
    "motor_offsets",
    "p_gains",
    "d_gains",
    "joint_friction_coeffs",
    "joint_damping_coeffs",
    "joint_armatures",
    "spring_stiffness",
    "spring_damping",
    "spring_rest_pos",
)

_HISTORY_SUFFIXES = ("_history", "_stored", "_delayed")


def _snapshot_tensor_names(env) -> tuple[str, ...]:
    """Return existing per-environment tensors required by policy feedback."""
    names = set(_TENSOR_STATE_NAMES)
    for name, value in vars(env).items():
        if name.endswith(_HISTORY_SUFFIXES) and isinstance(value, torch.Tensor):
            names.add(name)
    return tuple(sorted(names))


class RecoveryStateReplay:
    """Capture real landings and replay them as episode initial states."""

    def __init__(
        self,
        probability: float,
        max_post_landing_steps: int,
        minimum_jump_height: float,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("recovery replay probability must be in [0, 1]")
        if max_post_landing_steps < 0:
            raise ValueError("maximum post-landing capture steps must be non-negative")
        if minimum_jump_height <= 0.0:
            raise ValueError("minimum recovery snapshot jump height must be positive")
        self.probability = probability
        self.max_post_landing_steps = max_post_landing_steps
        self.minimum_jump_height = minimum_jump_height
        self._initialized = False
        self._bank: Dict[str, torch.Tensor] = {}
        self._filter_bank: Dict[str, torch.Tensor] = {}
        self._lag_bank: list[torch.Tensor] = []
        self.total_capture_count = 0
        self.total_restore_count = 0

    def _initialize(self, env) -> None:
        device = env.device
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.recovery_episode = torch.zeros(
            env.num_envs, dtype=torch.bool, device=device
        )
        self.steps_since_landing = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=device
        )
        self.capture_target = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
        self.snapshot_landing_age = torch.zeros(
            env.num_envs, dtype=torch.long, device=device
        )
        for name in _snapshot_tensor_names(env):
            value = getattr(env, name, None)
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == env.num_envs
            ):
                self._bank[name] = torch.zeros_like(value)
        action_filter = getattr(env, "action_filter", None)
        for name in ("filtered_values", "xhist", "yhist"):
            value = getattr(action_filter, name, None)
            if isinstance(value, torch.Tensor):
                self._filter_bank[name] = torch.zeros_like(value)
        self._lag_bank = [torch.zeros_like(value) for value in env.lag_buffer]
        self._initialized = True

    def _capture(self, env, mask: torch.Tensor) -> None:
        env_ids = torch.nonzero(mask, as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        for name, bank in self._bank.items():
            value = getattr(env, name)
            # A few legacy delayed-state buffers are initialized with history
            # width and later replaced by a current-state tensor.  Allocate
            # against the live shape at first useful capture, not the bootstrap
            # shape observed on the first simulation step.
            if bank.shape != value.shape:
                bank = torch.zeros_like(value)
                self._bank[name] = bank
            bank[env_ids] = value[env_ids]
        action_filter = getattr(env, "action_filter", None)
        for name, bank in self._filter_bank.items():
            bank[env_ids] = getattr(action_filter, name)[env_ids]
        for bank, value in zip(self._lag_bank, env.lag_buffer):
            bank[env_ids] = value[env_ids]
        self.snapshot_landing_age[env_ids] = self.steps_since_landing[env_ids]
        self.valid[env_ids] = True
        self.total_capture_count += int(env_ids.numel())

    def capture_before_termination(self, env) -> None:
        """Capture first contact and a randomized early-recovery successor."""
        if not self._initialized:
            self._initialize(env)

        natural = ~self.recovery_episode
        qualified_landing = (
            env.first_landing_event.bool()
            & natural
            & (env.max_height >= self.minimum_jump_height)
        )
        if torch.any(qualified_landing):
            env_ids = torch.nonzero(qualified_landing, as_tuple=False).flatten()
            self.steps_since_landing[env_ids] = 0
            if self.max_post_landing_steps > 0:
                self.capture_target[env_ids] = torch.randint(
                    0,
                    self.max_post_landing_steps + 1,
                    (env_ids.numel(),),
                    device=env.device,
                )
            else:
                self.capture_target[env_ids] = 0
            # Always retain a valid first-contact state.  It may be replaced
            # below by the randomized later recovery state.
            self._capture(env, qualified_landing)

        active = (self.steps_since_landing >= 0) & natural
        capture_now = active & (
            self.steps_since_landing == self.capture_target
        )
        self._capture(env, capture_now)
        self.steps_since_landing[active] += 1

    def choose_restore_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._initialized or env_ids.numel() == 0:
            return env_ids[:0]
        eligible = self.valid[env_ids]
        draw = torch.rand(env_ids.numel(), device=env_ids.device)
        return env_ids[eligible & (draw < self.probability)]

    def restore_after_reset(self, env, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for name, bank in self._bank.items():
            getattr(env, name)[env_ids] = bank[env_ids]
        action_filter = getattr(env, "action_filter", None)
        for name, bank in self._filter_bank.items():
            getattr(action_filter, name)[env_ids] = bank[env_ids]
        for value, bank in zip(env.lag_buffer, self._lag_bank):
            value[env_ids] = bank[env_ids]

        # Recovery is a new PPO episode, but begins after a real landing.  It
        # therefore gets a full horizon and must not repeat one-shot landing
        # event rewards.
        env.episode_length_buf[env_ids] = 0
        env.first_landing_event[env_ids] = False
        env._has_jumped_rand_envs[env_ids] = False
        env._has_jumped_switched_time[env_ids] = -self.snapshot_landing_age[env_ids]
        env.settled_after_init_timer[env_ids] = -env.max_episode_length
        env.initial_zero_action_steps_remaining[env_ids] = 0
        if hasattr(env, "remaining_time"):
            env.remaining_time[env_ids] = env.max_episode_length
        self.recovery_episode[env_ids] = True
        self.steps_since_landing[env_ids] = -1
        first_restore = self.total_restore_count == 0
        self.total_restore_count += int(env_ids.numel())

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        # reset_idx() samples episodic DOF properties before replay restoration.
        # Put the actor back in the property state that produced the snapshot.
        env._refresh_actor_dof_props(env_ids)
        if first_restore:
            print(
                "Recovery-state replay first restore: "
                f"restored={int(env_ids.numel())}, "
                f"valid_bank={int(self.valid.sum().item())}/{env.num_envs}, "
                f"captures={self.total_capture_count}"
            )

    def note_reset(self, env_ids: torch.Tensor) -> None:
        if not self._initialized or env_ids.numel() == 0:
            return
        self.recovery_episode[env_ids] = False
        self.steps_since_landing[env_ids] = -1


def install_recovery_state_replay(
    robot_class,
    probability: float,
    max_post_landing_steps: int,
    minimum_jump_height: float,
) -> None:
    """Install opt-in replay hooks on ``robot_class`` before env creation."""
    if probability <= 0.0:
        return
    replay = RecoveryStateReplay(
        probability, max_post_landing_steps, minimum_jump_height
    )
    original_check_termination = robot_class.check_termination
    original_reset_idx = robot_class.reset_idx

    def check_termination_with_recovery_capture(self):
        replay.capture_before_termination(self)
        return original_check_termination(self)

    def reset_idx_with_recovery_replay(self, env_ids):
        restore_ids = replay.choose_restore_ids(env_ids)
        replay.note_reset(env_ids)
        result = original_reset_idx(self, env_ids)
        replay.restore_after_reset(self, restore_ids)
        return result

    robot_class.check_termination = check_termination_with_recovery_capture
    robot_class.reset_idx = reset_idx_with_recovery_replay
    print(
        "On-policy recovery-state replay: "
        f"probability={probability}, "
        f"capture_window=0..{max_post_landing_steps} steps, "
        f"minimum_jump_height={minimum_jump_height} m"
    )
