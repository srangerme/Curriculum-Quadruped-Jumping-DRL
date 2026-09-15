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
        settled_contact_count = 3

    class domain_rand(Go2UpwardsCfg.domain_rand):
        class ranges(
            Solo12V31PhysicalRandomizationRanges,
            Go2UpwardsCfg.domain_rand.ranges,
        ):
            # Upward-only override, validated across seed22 and seed23.
            com_displacement_range = [
                [-0.02, -0.02, -0.02],
                [0.02, 0.02, 0.02],
            ]

    class rewards(Go2UpwardsCfg.rewards):
        stance_height_target = 0.30
        squat_height_target = 0.188
        feet_tuck_height_target = -0.141
        feet_tuck_activation_height = 0.42
        max_contact_force = 120.0
        upward_height_speed_height_min = 0.50
        upward_height_speed_height_sigma = 0.02
        upward_height_speed_limit_ratio = 1.25
        upward_height_speed_speed_sigma = 0.05
        upward_height_speed_max_penalty = 3.0
        upward_action_reference = []
        upward_action_reference_sigma = 0.25

        class scales(Go2UpwardsCfg.rewards.scales):
            velocity_torque_envelope_rejection = 0.0
            upward_height_speed_joint = 0.0
            upward_action_reference = 0.0

class Solo12V31UpwardsCfgPPO(Go2UpwardsCfgPPO):
    class runner(Go2UpwardsCfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
