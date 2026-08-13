"""Second-stage forward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go2.go2_config import Go2Cfg, Go2CfgPPO
from legged_gym.envs.solo12.solo12_v3_1_common import (
    Solo12V31Asset,
    Solo12V31Control,
    Solo12V31InitState,
    Solo12V31Morphology,
    Solo12V31Viewer,
)


class Solo12V31Cfg(Go2Cfg):
    task_name = "solo12_v3_1_forward"
    init_state = Solo12V31InitState
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    asset = Solo12V31Asset
    viewer = Solo12V31Viewer

    class env(Go2Cfg.env):
        reset_height = 0.12

    class domain_rand(Go2Cfg.domain_rand):
        class ranges(Go2Cfg.domain_rand.ranges):
            added_mass_range = [-0.5, 1.0]
            com_displacement_range = [-0.04, 0.04]
            added_link_mass_range = [0.8, 1.2]

    class rewards(Go2Cfg.rewards):
        max_contact_force = 160.0


class Solo12V31CfgPPO(Go2CfgPPO):
    class runner(Go2CfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
