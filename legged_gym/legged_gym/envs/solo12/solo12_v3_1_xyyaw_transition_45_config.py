"""Solo12 transition adding independent landing-yaw commands up to +/-45 deg."""

import math

from legged_gym.envs.solo12.solo12_v3_1_y_transition_2_config import (
    Solo12V31YTransition2Cfg,
    Solo12V31YTransition2CfgPPO,
)


class Solo12V31XYYawTransition45Cfg(Solo12V31YTransition2Cfg):
    task_name = "solo12_v3_1_xyyaw_transition_45"

    class commands(Solo12V31YTransition2Cfg.commands):
        randomize_yaw = True
        yaw_relative_to_goal_heading = False
        # Four exact (x=0.5, y endpoint, yaw endpoint) sign combinations.
        # Each receives 2.5% of samples; the remaining continuous bucket is 20%.
        mixed_xyyaw_corner_probability = 0.025
        mixed_xyyaw_corner_x = 0.5

        class ranges(Solo12V31YTransition2Cfg.commands.ranges):
            yaw_ini = [-math.pi / 4, math.pi / 4]

        class distances(Solo12V31YTransition2Cfg.commands.distances):
            des_yaw = None

    class rewards(Solo12V31YTransition2Cfg.rewards):
        class scales(Solo12V31YTransition2Cfg.rewards.scales):
            task_ori = 1500.0
            post_landing_ori = 6.0


class Solo12V31XYYawTransition45CfgPPO(Solo12V31YTransition2CfgPPO):
    pass
