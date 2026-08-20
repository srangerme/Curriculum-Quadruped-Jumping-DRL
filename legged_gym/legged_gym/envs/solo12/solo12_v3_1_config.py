"""Second-stage forward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go2.go2_config import Go2Cfg, Go2CfgPPO
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


class Solo12V31Cfg(Go2Cfg):
    task_name = "solo12_v3_1_forward"
    init_state = Solo12V31InitState
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    imu = Solo12V31Imu
    asset = Solo12V31Asset
    sim = Solo12V31Sim
    viewer = Solo12V31Viewer

    class env(Go2Cfg.env):
        reset_height = 0.12
        settled_height_threshold = 0.38
        # Preserve the reset/action phase used by the accepted upward policy.
        initial_zero_action_steps = 1

    class domain_rand(Go2Cfg.domain_rand):
        # Preserve the latency coverage that made the 20/30/40 ms endpoints
        # learnable in the upward stage.  The command line selects the current
        # maximum; these probabilities select zero/max/interior regimes.
        zero_latency_probability = 0.5
        max_latency_probability = 0.25

        class ranges(Solo12V31PhysicalRandomizationRanges, Go2Cfg.domain_rand.ranges):
            pass

    class rewards(Go2Cfg.rewards):
        stance_height_target = 0.30
        squat_height_target = 0.188
        feet_tuck_height_target = -0.141
        feet_tuck_activation_height = 0.42
        # Solo's shorter legs need more time to leave the tucked pose before
        # the calf links can reach the ground.  This only matters when the
        # optional feet_landing_pose reward scale is enabled.
        feet_landing_pose_activation_height = 0.50
        max_contact_force = 160.0


class Solo12V31CfgPPO(Go2CfgPPO):
    class runner(Go2CfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
