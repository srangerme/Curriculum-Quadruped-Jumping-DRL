"""Full-range protected Solo12 v3_1 forward-jump training."""

from legged_gym.envs.solo12.solo12_v3_1_forward_transition_3_config import (
    Solo12V31ForwardTransition3Cfg,
    Solo12V31ForwardTransition3CfgPPO,
)


class Solo12V31ForwardTransition4Cfg(Solo12V31ForwardTransition3Cfg):
    task_name = "solo12_v3_1_forward_transition_4"

    class domain_rand(Solo12V31ForwardTransition3Cfg.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.5
        pos_vel_random_prob = 0.7

        # Introduce randomized post-jump initial states, but do not reset the
        # phase mid-episode yet. This is the last robustness step before the
        # unprotected full task.
        randomize_has_jumped = True
        has_jumped_random_prob = 0.3
        reset_has_jumped = False

        class ranges(Solo12V31ForwardTransition3Cfg.domain_rand.ranges):
            min_ori_euler = [-0.05, -0.05, -0.5]
            max_ori_euler = [0.05, 0.05, 0.5]

    class commands(Solo12V31ForwardTransition3Cfg.commands):
        min_jump_height_for_curriculum = 0.43

        class ranges(Solo12V31ForwardTransition3Cfg.commands.ranges):
            # Match the complete forward task's command envelope.
            pos_dx_ini = [0.0, 1.0]
            pos_dy_ini = [-0.3, 0.3]
            pos_dz_ini = [0.0, 0.0]

    class rewards(Solo12V31ForwardTransition3Cfg.rewards):
        class scales(Solo12V31ForwardTransition3Cfg.rewards.scales):
            # Precision is now close to the full objective while take-off is
            # still protected against the low-hop landing shortcut.
            task_pos = 1000.0
            task_ori = 1000.0
            task_max_height = 6000.0
            jumping = 500.0


class Solo12V31ForwardTransition4CfgPPO(Solo12V31ForwardTransition3CfgPPO):
    pass
