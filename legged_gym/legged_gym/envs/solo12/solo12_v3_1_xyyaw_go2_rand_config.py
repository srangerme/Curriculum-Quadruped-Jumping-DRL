"""Solo12 x/y/yaw continuation with Go2-level physical randomisation."""

from legged_gym.envs.solo12.solo12_v3_1_xyyaw_transition_45_config import (
    Solo12V31XYYawTransition45Cfg,
    Solo12V31XYYawTransition45CfgPPO,
)


class Solo12V31XYYawGo2RandCfg(Solo12V31XYYawTransition45Cfg):
    task_name = "solo12_v3_1_xyyaw_go2_rand"

    class domain_rand(Solo12V31XYYawTransition45Cfg.domain_rand):
        # Match Go2's physical/domain randomisation. Keep the Solo curriculum's
        # 25% goal-directed take-off assistance unchanged so this stage tests
        # robustness rather than adding more simulator help.
        push_robots = True
        push_interval_s = 1.0
        max_push_vel_xy = 1.0
        pos_vel_random_prob = 0.7

        randomize_has_jumped = True
        has_jumped_random_prob = 0.6
        reset_has_jumped = True
        manual_has_jumped_reset_time = 0

        push_towards_goal = True
        push_towards_goal_probability = 0.25
        push_towards_goal_final_probability = 0.25
        push_towards_goal_anneal_start_iteration = 0
        push_towards_goal_anneal_iterations = 0

        sim_latency = True
        base_latency = 0
        sim_pd_latency = False
        lag_timesteps = 6
        randomize_lag_timesteps = False

        class ranges(Solo12V31XYYawTransition45Cfg.domain_rand.ranges):
            min_ori_euler = [-0.05, -0.05, -0.5]
            max_ori_euler = [0.05, 0.05, 0.5]
            motor_strength_ranges = [0.9, 1.1]
            p_gains_range = [0.9, 1.1]
            d_gains_range = [0.9, 1.1]
            added_mass_range = [-1.0, 3.0]
            latency_range = [0.0, 40.0]
            pd_latency_range = [0.0, 0.0]
            additional_latency_range = [-5.0, 5.0]
            motor_offset_range = [-0.02, 0.02]
            restitution_range = [0.0, 0.4]
            friction_range = [0.01, 3.0]
            com_displacement_range = [-0.1, 0.1]
            joint_friction_range = [0.0, 0.04]
            joint_damping_range = [0.0, 0.01]
            added_link_mass_range = [0.7, 1.3]


class Solo12V31XYYawGo2RandCfgPPO(Solo12V31XYYawTransition45CfgPPO):
    pass
