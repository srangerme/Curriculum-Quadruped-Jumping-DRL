"""First-stage upward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go2.go2_upwards_config import Go2UpwardsCfg, Go2UpwardsCfgPPO
from legged_gym.envs.solo12.solo12_v3_1_common import (
    Solo12V31Asset,
    Solo12V31Control,
    Solo12V31InitState,
    Solo12V31Morphology,
    Solo12V31Viewer,
)


class Solo12V31UpwardsCfg(Go2UpwardsCfg):
    task_name = "solo12_v3_1_upwards"
    init_state = Solo12V31InitState
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    asset = Solo12V31Asset
    viewer = Solo12V31Viewer

    class env(Go2UpwardsCfg.env):
        reset_height = 0.12

    class domain_rand(Go2UpwardsCfg.domain_rand):
        class ranges(Go2UpwardsCfg.domain_rand.ranges):
            # Scaled to the lighter Solo12 model rather than the Go2 model.
            motor_strength_ranges = [0.95, 1.05]
            p_gains_range = [0.95, 1.05]
            d_gains_range = [0.95, 1.05]
            latency_range = [0.0, 20.0]
            added_mass_range = [-0.25, 0.5]
            com_displacement_range = [-0.02, 0.02]
            added_link_mass_range = [0.9, 1.1]

    class rewards(Go2UpwardsCfg.rewards):
        max_contact_force = 120.0


class Solo12V31UpwardsCfgPPO(Go2UpwardsCfgPPO):
    class runner(Go2UpwardsCfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
