"""Second Solo12 transition extending lateral jumps to +/-0.5 m."""

from legged_gym.envs.solo12.solo12_v3_1_y_transition_config import (
    Solo12V31YTransitionCfg,
    Solo12V31YTransitionCfgPPO,
)


class Solo12V31YTransition2Cfg(Solo12V31YTransitionCfg):
    task_name = "solo12_v3_1_y_transition_2"

    class commands(Solo12V31YTransitionCfg.commands):
        class ranges(Solo12V31YTransitionCfg.commands.ranges):
            pos_dy_ini = [-0.5, 0.5]


class Solo12V31YTransition2CfgPPO(Solo12V31YTransitionCfgPPO):
    pass
