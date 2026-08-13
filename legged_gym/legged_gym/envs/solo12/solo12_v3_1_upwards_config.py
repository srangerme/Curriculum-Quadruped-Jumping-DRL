"""First-stage upward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go1.go1_upwards_config import Go1UpwardsCfg, Go1UpwardsCfgPPO
from legged_gym.envs.solo12.solo12_v3_1_common import (
    Solo12V31Asset,
    Solo12V31Control,
    Solo12V31InitState,
    Solo12V31Morphology,
    Solo12V31Viewer,
)


class Solo12V31UpwardsCfg(Go1UpwardsCfg):
    task_name = "solo12_v3_1_upwards"
    init_state = Solo12V31InitState
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    asset = Solo12V31Asset
    viewer = Solo12V31Viewer

    class env(Go1UpwardsCfg.env):
        reset_height = 0.12

    class domain_rand(Go1UpwardsCfg.domain_rand):
        # First learn a clean take-off. Robustness randomisation can be widened
        # again after the jumping skill has formed.
        push_robots = False
        pos_vel_random_prob = 0.4
        has_jumped_random_prob = 0.4

        class ranges(Go1UpwardsCfg.domain_rand.ranges):
            # Scaled to the 4.398 kg Solo body rather than the Go1 body.
            motor_strength_ranges = [0.95, 1.05]
            p_gains_range = [0.95, 1.05]
            d_gains_range = [0.95, 1.05]
            latency_range = [0.0, 20.0]
            added_mass_range = [-0.25, 0.5]
            com_displacement_range = [-0.02, 0.02]
            added_link_mass_range = [0.9, 1.1]

    class rewards(Go1UpwardsCfg.rewards):
        max_contact_force = 120.0

        class scales(Go1UpwardsCfg.rewards.scales):
            jumping = 100.0
            task_max_height = 3000.0
            action_rate = -0.1
            energy_usage_actuators = -0.005


class Solo12V31UpwardsCfgPPO(Go1UpwardsCfgPPO):
    class runner(Go1UpwardsCfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
