"""Second Solo12 v3_1 transition: continuous, medium-range jumping."""

from legged_gym.envs.solo12.solo12_v3_1_forward_transition_config import (
    Solo12V31ForwardTransitionCfg,
    Solo12V31ForwardTransitionCfgPPO,
)


class Solo12V31ForwardTransition2Cfg(Solo12V31ForwardTransitionCfg):
    task_name = "solo12_v3_1_forward_transition_2"

    class domain_rand(Solo12V31ForwardTransitionCfg.domain_rand):
        # Do not expand randomisation while learning distance. Only reduce the
        # fraction of simulator-assisted take-offs.
        push_towards_goal_probability = 0.3
        push_towards_goal_final_probability = 0.3

    class commands(Solo12V31ForwardTransitionCfg.commands):
        # Slightly lower than transition one because continuous resets and
        # medium-range commands make a completed jump more difficult.
        min_jump_height_for_curriculum = 0.43
        upward_jump_probability = 0.2

        class ranges(Solo12V31ForwardTransitionCfg.commands.ranges):
            pos_dx_ini = [0.0, 0.6]
            pos_dy_ini = [-0.15, 0.15]
            pos_dz_ini = [0.0, 0.0]

class Solo12V31ForwardTransition2CfgPPO(Solo12V31ForwardTransitionCfgPPO):
    pass
