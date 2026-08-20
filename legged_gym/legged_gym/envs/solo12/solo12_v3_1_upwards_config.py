"""First-stage upward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go2.go2_upwards_config import Go2UpwardsCfg, Go2UpwardsCfgPPO
from legged_gym.envs.solo12.solo12_v3_1_common import (
    Solo12V31Asset,
    Solo12V31Control,
    Solo12V31Imu,
    Solo12V31InitState,
    Solo12V31Morphology,
    Solo12V31PhysicalRandomizationRanges,
    Solo12V31Sim,
    Solo12V31Viewer,
)


class Solo12V31UpwardsCfg(Go2UpwardsCfg):
    task_name = "solo12_v3_1_upwards"
    init_state = Solo12V31InitState
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    imu = Solo12V31Imu
    asset = Solo12V31Asset
    sim = Solo12V31Sim
    viewer = Solo12V31Viewer

    class env(Go2UpwardsCfg.env):
        reset_height = 0.12
        settled_height_threshold = 0.38

    class domain_rand(Go2UpwardsCfg.domain_rand):
        class ranges(
            Solo12V31PhysicalRandomizationRanges,
            Go2UpwardsCfg.domain_rand.ranges,
        ):
            pass

    class rewards(Go2UpwardsCfg.rewards):
        stance_height_target = 0.30
        squat_height_target = 0.188
        feet_tuck_height_target = -0.141
        feet_tuck_activation_height = 0.42
        max_contact_force = 120.0


class Solo12V31UpwardsCfgPPO(Go2UpwardsCfgPPO):
    class runner(Go2UpwardsCfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
