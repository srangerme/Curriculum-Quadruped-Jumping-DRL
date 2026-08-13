"""First Solo12 transition for lateral jumps while holding yaw at zero."""

from legged_gym.envs.solo12.solo12_v3_1_forward_transition_config import (
    Solo12V31ForwardTransitionCfg,
    Solo12V31ForwardTransitionCfgPPO,
)


class Solo12V31YTransitionCfg(Solo12V31ForwardTransitionCfg):
    task_name = "solo12_v3_1_y_transition"

    class commands(Solo12V31ForwardTransitionCfg.commands):
        randomize_yaw = False
        mixed_short_zero_probability = 0.15
        mixed_short_max_probability = 0.15
        mixed_lateral_endpoint_probability = 0.20

        class ranges(Solo12V31ForwardTransitionCfg.commands.ranges):
            pos_dx_ini = [0.0, 1.0]
            pos_dy_ini = [-0.25, 0.25]
            pos_dz_ini = [0.0, 0.0]

        class distances(Solo12V31ForwardTransitionCfg.commands.distances):
            des_yaw = 0.0

    class rewards(Solo12V31ForwardTransitionCfg.rewards):
        class scales(Solo12V31ForwardTransitionCfg.rewards.scales):
            # Lateral impulse requires intentional left/right asymmetry.
            symmetric_joints = 0.0
            takeoff_horizontal_velocity = 300.0


class Solo12V31YTransitionCfgPPO(Solo12V31ForwardTransitionCfgPPO):
    pass
