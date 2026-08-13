"""Solo12 transition from forward-only jumps to lateral and yaw commands."""

import math

from legged_gym.envs.solo12.solo12_v3_1_xyyaw_transition_45_config import (
    Solo12V31XYYawTransition45Cfg,
    Solo12V31XYYawTransition45CfgPPO,
)


class Solo12V31XYYawTransitionCfg(Solo12V31XYYawTransition45Cfg):
    task_name = "solo12_v3_1_xyyaw_transition"

    class commands(Solo12V31XYYawTransition45Cfg.commands):
        class ranges(Solo12V31XYYawTransition45Cfg.commands.ranges):
            yaw_ini = [-math.pi / 2, math.pi / 2]


class Solo12V31XYYawTransitionCfgPPO(Solo12V31XYYawTransition45CfgPPO):
    pass
