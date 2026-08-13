"""Final Solo12 v3_1 transition before the full forward-jump task."""

from legged_gym.envs.solo12.solo12_v3_1_forward_transition_2_config import (
    Solo12V31ForwardTransition2Cfg,
    Solo12V31ForwardTransition2CfgPPO,
)


class Solo12V31ForwardTransition3Cfg(Solo12V31ForwardTransition2Cfg):
    task_name = "solo12_v3_1_forward_transition_3"

    class domain_rand(Solo12V31ForwardTransition2Cfg.domain_rand):
        # Final distance-learning stage: phase out assistance over these 500
        # global PPO iterations while keeping randomisation unchanged.
        push_towards_goal_probability = 0.15
        push_towards_goal_final_probability = 0.0
        push_towards_goal_anneal_start_iteration = 2500
        push_towards_goal_anneal_iterations = 500

    class commands(Solo12V31ForwardTransition2Cfg.commands):
        min_jump_height_for_curriculum = 0.43
        upward_jump_probability = 0.2

        class ranges(Solo12V31ForwardTransition2Cfg.commands.ranges):
            pos_dx_ini = [0.0, 1.0]
            pos_dy_ini = [-0.3, 0.3]
            pos_dz_ini = [0.0, 0.0]

class Solo12V31ForwardTransition3CfgPPO(Solo12V31ForwardTransition2CfgPPO):
    pass
