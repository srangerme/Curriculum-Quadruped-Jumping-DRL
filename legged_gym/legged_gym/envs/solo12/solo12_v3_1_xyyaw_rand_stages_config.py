"""Progressive Solo12 x/y/yaw domain-randomisation continuation stages."""

from legged_gym.envs.solo12.solo12_v3_1_xyyaw_transition_45_config import (
    Solo12V31XYYawTransition45Cfg,
    Solo12V31XYYawTransition45CfgPPO,
)


class Solo12V31XYYawRandStage1Cfg(Solo12V31XYYawTransition45Cfg):
    """Intermediate actuator and observation-latency randomisation."""

    task_name = "solo12_v3_1_xyyaw_rand_stage1"

    class domain_rand(Solo12V31XYYawTransition45Cfg.domain_rand):
        class ranges(Solo12V31XYYawTransition45Cfg.domain_rand.ranges):
            motor_strength_ranges = [0.925, 1.075]
            p_gains_range = [0.925, 1.075]
            d_gains_range = [0.925, 1.075]
            latency_range = [0.0, 30.0]


class Solo12V31XYYawRandStage1CfgPPO(Solo12V31XYYawTransition45CfgPPO):
    pass


class Solo12V31XYYawRandStage2ActuatorCfg(Solo12V31XYYawRandStage1Cfg):
    """Reach full actuator variation while retaining the 30 ms latency cap."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_actuator"

    class domain_rand(Solo12V31XYYawRandStage1Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage1Cfg.domain_rand.ranges):
            motor_strength_ranges = [0.9, 1.1]
            p_gains_range = [0.9, 1.1]
            d_gains_range = [0.9, 1.1]


class Solo12V31XYYawRandStage2ActuatorCfgPPO(Solo12V31XYYawRandStage1CfgPPO):
    class algorithm(Solo12V31XYYawRandStage1CfgPPO.algorithm):
        learning_rate = 1.e-4


class Solo12V31XYYawRandStage2Latency30Jitter5Cfg(Solo12V31XYYawRandStage2ActuatorCfg):
    """Full actuator variation with 0--30 ms latency and +/-5 ms jitter."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_latency30_jitter5"

    class domain_rand(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand.ranges):
            latency_range = [0.0, 30.0]
            additional_latency_range = [-5.0, 5.0]


class Solo12V31XYYawRandStage2Latency30Jitter5CfgPPO(Solo12V31XYYawRandStage2ActuatorCfgPPO):
    pass


class Solo12V31XYYawRandStage2BridgeCfg(Solo12V31XYYawRandStage2ActuatorCfg):
    """Bridge to full actuator variation while limiting latency to 35 ms."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_bridge"

    class domain_rand(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand.ranges):
            latency_range = [0.0, 35.0]
            additional_latency_range = [-5.0, 5.0]


class Solo12V31XYYawRandStage2BridgeCfgPPO(Solo12V31XYYawRandStage2ActuatorCfgPPO):
    pass


class Solo12V31XYYawRandStage2BridgeNoAssistCfg(Solo12V31XYYawRandStage2BridgeCfg):
    """35 ms bridge that requires all take-off momentum from the policy."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_bridge_noassist"

    class domain_rand(Solo12V31XYYawRandStage2BridgeCfg.domain_rand):
        push_towards_goal = False
        push_towards_goal_probability = 0.0
        push_towards_goal_final_probability = 0.0
        push_towards_goal_anneal_start_iteration = 0
        push_towards_goal_anneal_iterations = 0


class Solo12V31XYYawRandStage2BridgeNoAssistCfgPPO(Solo12V31XYYawRandStage2BridgeCfgPPO):
    pass


class Solo12V31XYYawRandStage2Latency35StableCfg(Solo12V31XYYawRandStage2ActuatorCfg):
    """Raise episodic latency to 35 ms without per-step latency jitter."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_latency35_stable"

    class domain_rand(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2ActuatorCfg.domain_rand.ranges):
            latency_range = [0.0, 35.0]
            additional_latency_range = [0.0, 0.0]


class Solo12V31XYYawRandStage2Latency35StableCfgPPO(Solo12V31XYYawRandStage2ActuatorCfgPPO):
    pass


class Solo12V31XYYawRandStage2Latency40StableCfg(Solo12V31XYYawRandStage2Latency35StableCfg):
    """Raise episodic latency to 40 ms while retaining zero jitter."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_latency40_stable"

    class domain_rand(Solo12V31XYYawRandStage2Latency35StableCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2Latency35StableCfg.domain_rand.ranges):
            latency_range = [0.0, 40.0]


class Solo12V31XYYawRandStage2Latency40StableCfgPPO(Solo12V31XYYawRandStage2Latency35StableCfgPPO):
    pass


class Solo12V31XYYawRandStage2Latency40Jitter25Cfg(Solo12V31XYYawRandStage2Latency40StableCfg):
    """Introduce half-strength per-step latency jitter at the 40 ms cap."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2_latency40_jitter25"

    class domain_rand(Solo12V31XYYawRandStage2Latency40StableCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2Latency40StableCfg.domain_rand.ranges):
            additional_latency_range = [-2.5, 2.5]


class Solo12V31XYYawRandStage2Latency40Jitter25CfgPPO(Solo12V31XYYawRandStage2Latency40StableCfgPPO):
    pass


class Solo12V31XYYawRandStage2Cfg(Solo12V31XYYawRandStage2BridgeCfg):
    """Go2-level actuator and observation-latency randomisation."""

    task_name = "solo12_v3_1_xyyaw_rand_stage2"

    class domain_rand(Solo12V31XYYawRandStage2BridgeCfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage2BridgeCfg.domain_rand.ranges):
            latency_range = [0.0, 40.0]


class Solo12V31XYYawRandStage2CfgPPO(Solo12V31XYYawRandStage2BridgeCfgPPO):
    pass


class Solo12V31XYYawRandStage3HasJump30Cfg(Solo12V31XYYawRandStage2Latency30Jitter5Cfg):
    """Introduce Go2-style episode-level has_jumped states at 30%."""

    task_name = "solo12_v3_1_xyyaw_rand_stage3_hasjump30"

    class domain_rand(Solo12V31XYYawRandStage2Latency30Jitter5Cfg.domain_rand):
        randomize_has_jumped = True
        has_jumped_random_prob = 0.3
        reset_has_jumped = True
        manual_has_jumped_reset_time = 0

class Solo12V31XYYawRandStage3HasJump30CfgPPO(Solo12V31XYYawRandStage2Latency30Jitter5CfgPPO):
    class algorithm(Solo12V31XYYawRandStage2Latency30Jitter5CfgPPO.algorithm):
        # The phase randomisation changes the episode-state distribution.  Use
        # a smaller fixed step so that adapting to post-jump states does not
        # erase the already learned long-distance and yaw tracking behaviour.
        learning_rate = 5.e-5


class Solo12V31XYYawRandStage3HasJump60Cfg(Solo12V31XYYawRandStage3HasJump30Cfg):
    """Reach Go2's 60% conditional episode-level has_jumped probability."""

    task_name = "solo12_v3_1_xyyaw_rand_stage3_hasjump60"

    class domain_rand(Solo12V31XYYawRandStage3HasJump30Cfg.domain_rand):
        has_jumped_random_prob = 0.6


class Solo12V31XYYawRandStage3HasJump60CfgPPO(Solo12V31XYYawRandStage3HasJump30CfgPPO):
    pass


class Solo12V31XYYawRandStage4Cfg(Solo12V31XYYawRandStage3HasJump60Cfg):
    """Solo-scaled inertial randomisation."""

    task_name = "solo12_v3_1_xyyaw_rand_stage4"

    class domain_rand(Solo12V31XYYawRandStage3HasJump60Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandStage3HasJump60Cfg.domain_rand.ranges):
            added_mass_range = [-0.5, 1.0]
            com_displacement_range = [-0.04, 0.04]
            added_link_mass_range = [0.8, 1.2]


class Solo12V31XYYawRandStage4CfgPPO(Solo12V31XYYawRandStage3HasJump60CfgPPO):
    pass


class Solo12V31XYYawRandStage5Push02Cfg(Solo12V31XYYawRandStage4Cfg):
    """Introduce mild external pushes."""

    task_name = "solo12_v3_1_xyyaw_rand_stage5_push02"

    class domain_rand(Solo12V31XYYawRandStage4Cfg.domain_rand):
        push_robots = True
        push_interval_s = 1.0
        max_push_vel_xy = 0.2


class Solo12V31XYYawRandStage5Push02CfgPPO(Solo12V31XYYawRandStage4CfgPPO):
    pass


class Solo12V31XYYawRandStage5Push04Cfg(Solo12V31XYYawRandStage5Push02Cfg):
    """Increase external pushes to the planned production maximum."""

    task_name = "solo12_v3_1_xyyaw_rand_stage5_push04"

    class domain_rand(Solo12V31XYYawRandStage5Push02Cfg.domain_rand):
        max_push_vel_xy = 0.4


class Solo12V31XYYawRandStage5Push04CfgPPO(Solo12V31XYYawRandStage5Push02CfgPPO):
    pass
