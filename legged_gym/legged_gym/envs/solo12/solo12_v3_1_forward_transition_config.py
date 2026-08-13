"""Short-range transition from Solo12 v3_1 upward to full forward jumping."""

from legged_gym.envs.solo12.solo12_v3_1_config import (
    Solo12V31Cfg,
    Solo12V31CfgPPO,
)


class Solo12V31ForwardTransitionCfg(Solo12V31Cfg):
    task_name = "solo12_v3_1_forward_transition"

    class env(Solo12V31Cfg.env):
        # Match the full forward task and Go2 so completed jumps can contribute
        # full-episode height and landing rewards.
        continuous_jumping = True
        continuous_jumping_reset_probability = 0.9

    class domain_rand(Solo12V31Cfg.domain_rand):
        push_robots = False
        push_towards_goal = True
        push_towards_goal_probability = 0.25
        push_towards_goal_final_probability = 0.25
        push_towards_goal_anneal_start_iteration = 0
        push_towards_goal_anneal_iterations = 0
        pos_vel_random_prob = 0.4
        randomize_has_jumped = False
        has_jumped_random_prob = 0.0
        reset_has_jumped = False

        class ranges(Solo12V31Cfg.domain_rand.ranges):
            min_ori_euler = [0.0, 0.0, -0.1]
            max_ori_euler = [0.0, 0.0, 0.1]
            motor_strength_ranges = [0.95, 1.05]
            p_gains_range = [0.95, 1.05]
            d_gains_range = [0.95, 1.05]
            latency_range = [0.0, 20.0]
            added_mass_range = [-0.25, 0.5]
            com_displacement_range = [-0.02, 0.02]
            added_link_mass_range = [0.9, 1.1]

    class commands(Solo12V31Cfg.commands):
        jump_over_box = False
        randomize_yaw = False
        # A correct landing alone is insufficient: retain the upward-stage
        # take-off capability before advancing to longer commands.
        min_jump_height_for_curriculum = 0.45
        upward_jump_probability = 0.3
        mixed_short_command_sampling = True
        mixed_short_zero_probability = 0.2
        mixed_short_max_probability = 0.2
        # The retained vertical policy produces a stable ~0.44 s flight. At
        # take-off, horizontal targets use only the remaining displacement.
        expected_flight_time = 0.44

        class ranges(Solo12V31Cfg.commands.ranges):
            # Third distance stage: retain the same endpoint/uniform mixture
            # while expanding the forward command envelope to 1.0 m.
            pos_dx_ini = [0.0, 1.0]
            pos_dy_ini = [-0.05, 0.05]
            pos_dz_ini = [0.0, 0.0]

    class rewards(Solo12V31Cfg.rewards):
        class scales(Solo12V31Cfg.rewards.scales):
            # Reduce easy stance/landing shortcuts and strengthen the explicit
            # jump incentive during policy transfer.
            base_height_stance = 5.0
            default_pose = 6.0
            post_landing_ori = 3.0
            task_pos = 1500.0
            task_ori = 300.0
            task_max_height = 5000.0
            jumping = 400.0
            # Preserve the upward policy's take-off impulse while adapting the
            # horizontal component. This is paid once at the first flight step.
            takeoff_vertical_velocity = 300.0

        # Tighten landing accuracy now that task_pos is paid at first contact.
        command_pos_tracking_sigma = 0.015


class Solo12V31ForwardTransitionCfgPPO(Solo12V31CfgPPO):
    class algorithm(Solo12V31CfgPPO.algorithm):
        learning_rate = 3.e-4
        schedule = "fixed"

    class runner(Solo12V31CfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
        load_optimizer = True
