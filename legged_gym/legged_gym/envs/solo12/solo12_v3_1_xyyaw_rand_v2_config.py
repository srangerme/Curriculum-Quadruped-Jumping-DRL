"""Attributable Solo12 x/y/yaw domain-randomisation curriculum.

Every registered task changes only one related randomisation family.  The
command and reward definitions always come from the +/-90 degree policy task.
"""

from legged_gym.envs.solo12.solo12_v3_1_xyyaw_transition_config import (
    Solo12V31XYYawTransitionCfg,
    Solo12V31XYYawTransitionCfgPPO,
)


class Solo12V31XYYawRandV2S0CleanCfg(Solo12V31XYYawTransitionCfg):
    """Exact randomisation baseline used while training model 8100."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline"

    class commands(Solo12V31XYYawTransitionCfg.commands):
        # Make the V2 upward/forward phase setting effective even though the
        # inherited policy uses the endpoint-aware mixed command sampler.
        mixed_use_upward_jump_probability = True
        mixed_reference_zero_probability = 0.15

    class domain_rand(Solo12V31XYYawTransitionCfg.domain_rand):
        # Preserve every randomisation inherited by the original 8100 task and
        # only pin the effective training assistance used by the accepted run.
        push_towards_goal = True
        push_towards_goal_probability = 0.25
        push_towards_goal_final_probability = 0.25
        push_towards_goal_anneal_start_iteration = 0
        push_towards_goal_anneal_iterations = 0


class Solo12V31XYYawRandV2S0CleanCfgPPO(Solo12V31XYYawTransitionCfgPPO):
    class algorithm(Solo12V31XYYawTransitionCfgPPO.algorithm):
        learning_rate = 1.e-4
        schedule = "fixed"


class Solo12V31XYYawRandV2S0BaselineUpCfg(Solo12V31XYYawRandV2S0CleanCfg):
    """Adapt model 8100 to its existing randomisation, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_up"

    class commands(Solo12V31XYYawRandV2S0CleanCfg.commands):
        upward_jump_probability = 0.5


class Solo12V31XYYawRandV2S0BaselineUpCfgPPO(Solo12V31XYYawRandV2S0CleanCfgPPO):
    pass


class Solo12V31XYYawRandV2S0BaselineForwardCfg(Solo12V31XYYawRandV2S0BaselineUpCfg):
    """Transfer baseline robustness back to the full x/y/yaw distribution."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward"

    class commands(Solo12V31XYYawRandV2S0BaselineUpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S0BaselineForwardCfgPPO(Solo12V31XYYawRandV2S0BaselineUpCfgPPO):
    pass


class Solo12V31XYYawRandV2S0BaselineForwardLowLrCfg(Solo12V31XYYawRandV2S0BaselineForwardCfg):
    """Low-learning-rate consolidation of the accepted baseline policy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_lr5e5"


class Solo12V31XYYawRandV2S0BaselineForwardLowLrCfgPPO(Solo12V31XYYawRandV2S0BaselineForwardCfgPPO):
    class algorithm(Solo12V31XYYawRandV2S0BaselineForwardCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S0BaselineForwardFarCfg(Solo12V31XYYawRandV2S0BaselineForwardCfg):
    """Consolidate the 1 m boundary while retaining all other command buckets."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far"

    class commands(Solo12V31XYYawRandV2S0BaselineForwardCfg.commands):
        # After the 20% upward override these become 18.8% for x=1 m and
        # 16.5% for each lateral endpoint.  Zero, corner, and continuous
        # buckets remain unchanged.
        mixed_short_max_probability = 0.20
        mixed_lateral_endpoint_probability = 0.175


class Solo12V31XYYawRandV2S0BaselineForwardFarCfgPPO(Solo12V31XYYawRandV2S0BaselineForwardCfgPPO):
    pass


class Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfg(Solo12V31XYYawRandV2S0BaselineForwardFarCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far_lr5e5"


class Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfgPPO(Solo12V31XYYawRandV2S0BaselineForwardFarCfgPPO):
    class algorithm(Solo12V31XYYawRandV2S0BaselineForwardFarCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfg(Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfg):
    """Final small reweighting to give the 1 m gate measurable margin."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far2_lr5e5"

    class commands(Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfg.commands):
        mixed_short_max_probability = 0.25
        mixed_lateral_endpoint_probability = 0.15


class Solo12V31XYYawRandV2S0BaselineForwardFar2LowLrCfgPPO(Solo12V31XYYawRandV2S0BaselineForwardFarLowLrCfgPPO):
    pass


class Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfg(Solo12V31XYYawRandV2S0BaselineForwardCfg):
    """Increase the 1 m anchor by borrowing only from continuous commands."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_baseline_forward_far_cont_lr5e5"

    class commands(Solo12V31XYYawRandV2S0BaselineForwardCfg.commands):
        mixed_short_max_probability = 0.20


class Solo12V31XYYawRandV2S0BaselineForwardFarContinuousLowLrCfgPPO(Solo12V31XYYawRandV2S0BaselineForwardCfgPPO):
    class algorithm(Solo12V31XYYawRandV2S0BaselineForwardCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S0JointEndpointsUpCfg(Solo12V31XYYawRandV2S0BaselineUpCfg):
    """Stratify the already-configured joint resistance range at its endpoints."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_joint_endpoints_up"

    class domain_rand(Solo12V31XYYawRandV2S0BaselineUpCfg.domain_rand):
        joint_resistance_endpoint_probability = 0.5


class Solo12V31XYYawRandV2S0JointEndpointsUpCfgPPO(Solo12V31XYYawRandV2S0BaselineUpCfgPPO):
    class algorithm(Solo12V31XYYawRandV2S0BaselineUpCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S0JointEndpointsForwardCfg(Solo12V31XYYawRandV2S0JointEndpointsUpCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s0_joint_endpoints_forward"

    class commands(Solo12V31XYYawRandV2S0JointEndpointsUpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S0JointEndpointsForwardCfgPPO(Solo12V31XYYawRandV2S0JointEndpointsUpCfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator5Cfg(Solo12V31XYYawRandV2S0JointEndpointsForwardCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator5_up"

    class commands(Solo12V31XYYawRandV2S0JointEndpointsForwardCfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S0JointEndpointsForwardCfg.domain_rand):
        randomize_motor_strength = True
        randomize_PD_gains = True

        class ranges(Solo12V31XYYawRandV2S0JointEndpointsForwardCfg.domain_rand.ranges):
            motor_strength_ranges = [0.95, 1.05]
            p_gains_range = [0.95, 1.05]
            d_gains_range = [0.95, 1.05]


class Solo12V31XYYawRandV2S1Actuator5CfgPPO(Solo12V31XYYawRandV2S0JointEndpointsForwardCfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator5ForwardCfg(Solo12V31XYYawRandV2S1Actuator5Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator5_forward"

    class commands(Solo12V31XYYawRandV2S1Actuator5Cfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S1Actuator5ForwardCfgPPO(Solo12V31XYYawRandV2S1Actuator5CfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator75Cfg(Solo12V31XYYawRandV2S1Actuator5ForwardCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator75_up"

    class commands(Solo12V31XYYawRandV2S1Actuator5ForwardCfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S1Actuator5ForwardCfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S1Actuator5ForwardCfg.domain_rand.ranges):
            motor_strength_ranges = [0.925, 1.075]
            p_gains_range = [0.925, 1.075]
            d_gains_range = [0.925, 1.075]


class Solo12V31XYYawRandV2S1Actuator75CfgPPO(Solo12V31XYYawRandV2S1Actuator5ForwardCfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator75ForwardCfg(Solo12V31XYYawRandV2S1Actuator75Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator75_forward"

    class commands(Solo12V31XYYawRandV2S1Actuator75Cfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S1Actuator75ForwardCfgPPO(Solo12V31XYYawRandV2S1Actuator75CfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator10Cfg(Solo12V31XYYawRandV2S1Actuator75ForwardCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_up"

    class commands(Solo12V31XYYawRandV2S1Actuator75ForwardCfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S1Actuator75ForwardCfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S1Actuator75ForwardCfg.domain_rand.ranges):
            motor_strength_ranges = [0.9, 1.1]
            p_gains_range = [0.9, 1.1]
            d_gains_range = [0.9, 1.1]


class Solo12V31XYYawRandV2S1Actuator10CfgPPO(Solo12V31XYYawRandV2S1Actuator75CfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator10ForwardCfg(Solo12V31XYYawRandV2S1Actuator10Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_forward"

    class commands(Solo12V31XYYawRandV2S1Actuator10Cfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S1Actuator10ForwardCfgPPO(Solo12V31XYYawRandV2S1Actuator10CfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator10EndpointsCfg(Solo12V31XYYawRandV2S1Actuator10Cfg):
    """Reinforce the +/-10% actuator boundaries while retaining uniform samples."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_up"

    class domain_rand(Solo12V31XYYawRandV2S1Actuator10Cfg.domain_rand):
        actuator_endpoint_probability = 0.5


class Solo12V31XYYawRandV2S1Actuator10EndpointsCfgPPO(Solo12V31XYYawRandV2S1Actuator10CfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfg(Solo12V31XYYawRandV2S1Actuator10EndpointsCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward"

    class commands(Solo12V31XYYawRandV2S1Actuator10EndpointsCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfgPPO(Solo12V31XYYawRandV2S1Actuator10EndpointsCfgPPO):
    pass


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfg(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfg
):
    """Low-rate Forward retry used to avoid drifting past robust checkpoints."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_stable"


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfgPPO(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfgPPO
):
    class algorithm(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardCfgPPO.algorithm):
        learning_rate = 2.e-5


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfg(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfg
):
    """Retain the 1 m boundary by borrowing samples from the uniform bucket."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_far_stable"

    class commands(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfg.commands):
        # Effective probability is 18.8% after the Forward 20% upward override.
        # Lateral/yaw/corner anchor probabilities are unchanged.
        mixed_short_max_probability = 0.20


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfgPPO(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardStableCfgPPO
):
    pass


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfg
):
    """Final Forward retry with additional 1 m boundary coverage."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s1_actuator10_endpoints_forward_far2_stable"

    class commands(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfg.commands):
        # Effective probability is 23.5% after the Forward 20% upward override.
        mixed_short_max_probability = 0.25


class Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfgPPO(
    Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFarStableCfgPPO
):
    pass


class Solo12V31XYYawRandV2S2Latency20Cfg(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency20"

    class domain_rand(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg.domain_rand):
        sim_latency = True
        base_latency = 0
        sim_pd_latency = False

        class ranges(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfg.domain_rand.ranges):
            latency_range = [0.0, 20.0]
            additional_latency_range = [0.0, 0.0]


class Solo12V31XYYawRandV2S2Latency20CfgPPO(Solo12V31XYYawRandV2S1Actuator10EndpointsForwardFar2StableCfgPPO):
    pass


class Solo12V31XYYawRandV2S2Latency30Cfg(Solo12V31XYYawRandV2S2Latency20Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency30"

    class domain_rand(Solo12V31XYYawRandV2S2Latency20Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S2Latency20Cfg.domain_rand.ranges):
            latency_range = [0.0, 30.0]


class Solo12V31XYYawRandV2S2Latency30CfgPPO(Solo12V31XYYawRandV2S2Latency20CfgPPO):
    pass


class Solo12V31XYYawRandV2S2Latency30Jitter5Cfg(Solo12V31XYYawRandV2S2Latency30Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5"

    class domain_rand(Solo12V31XYYawRandV2S2Latency30Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S2Latency30Cfg.domain_rand.ranges):
            additional_latency_range = [-5.0, 5.0]


class Solo12V31XYYawRandV2S2Latency30Jitter5CfgPPO(Solo12V31XYYawRandV2S2Latency30CfgPPO):
    pass


class Solo12V31XYYawRandV2S2Latency30Jitter5UpCfg(Solo12V31XYYawRandV2S2Latency30Jitter5Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_up"

    class commands(Solo12V31XYYawRandV2S2Latency30Jitter5Cfg.commands):
        upward_jump_probability = 0.5


class Solo12V31XYYawRandV2S2Latency30Jitter5UpCfgPPO(Solo12V31XYYawRandV2S2Latency30Jitter5CfgPPO):
    pass


class Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg(Solo12V31XYYawRandV2S2Latency30Jitter5UpCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_forward"

    class commands(Solo12V31XYYawRandV2S2Latency30Jitter5UpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfgPPO(Solo12V31XYYawRandV2S2Latency30Jitter5UpCfgPPO):
    pass


class Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg(
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg
):
    """Increase 1 m retention while adapting the full command mix to latency."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s2_latency30_jitter5_forward_far3"

    class commands(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg.commands):
        # Effective probability is 28.2% after the Forward 20% upward override.
        mixed_short_max_probability = 0.30


class Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3CfgPPO(
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriMildUpCfg(
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg
):
    """Widen only the reset orientation distribution, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_mild_up"

    class commands(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3Cfg.domain_rand.ranges):
            min_ori_euler = [-0.025, -0.025, -0.3]
            max_ori_euler = [0.025, 0.025, 0.3]


class Solo12V31XYYawRandV2S3InitialOriMildUpCfgPPO(
    Solo12V31XYYawRandV2S2Latency30Jitter5ForwardFar3CfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriMildForwardCfg(
    Solo12V31XYYawRandV2S3InitialOriMildUpCfg
):
    """Transfer the mild reset-orientation robustness to the command mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_mild_forward"

    class commands(Solo12V31XYYawRandV2S3InitialOriMildUpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S3InitialOriMildForwardCfgPPO(
    Solo12V31XYYawRandV2S3InitialOriMildUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriFullUpCfg(
    Solo12V31XYYawRandV2S3InitialOriMildForwardCfg
):
    """Reach the Go2 reset-orientation range, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_up"

    class commands(Solo12V31XYYawRandV2S3InitialOriMildForwardCfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S3InitialOriMildForwardCfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S3InitialOriMildForwardCfg.domain_rand.ranges):
            min_ori_euler = [-0.05, -0.05, -0.5]
            max_ori_euler = [0.05, 0.05, 0.5]


class Solo12V31XYYawRandV2S3InitialOriFullUpCfgPPO(
    Solo12V31XYYawRandV2S3InitialOriMildForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriFullForwardCfg(
    Solo12V31XYYawRandV2S3InitialOriFullUpCfg
):
    """Transfer the full Go2 reset-orientation range to the command mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward"

    class commands(Solo12V31XYYawRandV2S3InitialOriFullUpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S3InitialOriFullForwardCfgPPO(
    Solo12V31XYYawRandV2S3InitialOriFullUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfg(
    Solo12V31XYYawRandV2S3InitialOriFullForwardCfg
):
    """Small 1 m reweighting after full-orientation Forward adaptation."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward_far"

    class commands(Solo12V31XYYawRandV2S3InitialOriFullForwardCfg.commands):
        # Effective 1 m share is ~29.4% with the 20% upward override, while
        # retaining ~3.5% for continuous commands.
        mixed_short_max_probability = 0.3125


class Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfgPPO(
    Solo12V31XYYawRandV2S3InitialOriFullForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg(
    Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfg
):
    """Third full-orientation Forward attempt with a 30.6% effective 1 m share."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_ori_full_forward_far2"

    class commands(Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfg.commands):
        mixed_short_max_probability = 0.325


class Solo12V31XYYawRandV2S3InitialOriFullForwardFar2CfgPPO(
    Solo12V31XYYawRandV2S3InitialOriFullForwardFarCfgPPO
):
    pass


class Solo12V31XYYawRandV2S4InertialFullUpCfg(
    Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg
):
    """Expand the already-enabled inertial family to the Solo full range."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_up"

    class commands(Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg.commands):
        upward_jump_probability = 0.5

    class domain_rand(Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S3InitialOriFullForwardFar2Cfg.domain_rand.ranges):
            added_mass_range = [-0.5, 1.0]
            com_displacement_range = [-0.04, 0.04]
            added_link_mass_range = [0.8, 1.2]


class Solo12V31XYYawRandV2S4InertialFullUpCfgPPO(
    Solo12V31XYYawRandV2S3InitialOriFullForwardFar2CfgPPO
):
    pass


class Solo12V31XYYawRandV2S4InertialFullForwardCfg(
    Solo12V31XYYawRandV2S4InertialFullUpCfg
):
    """Transfer full inertial robustness back to the Forward command mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward"

    class commands(Solo12V31XYYawRandV2S4InertialFullUpCfg.commands):
        upward_jump_probability = 0.2


class Solo12V31XYYawRandV2S4InertialFullForwardCfgPPO(
    Solo12V31XYYawRandV2S4InertialFullUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S4InertialFullForwardFarCfg(
    Solo12V31XYYawRandV2S4InertialFullForwardCfg
):
    """Reweight Forward commands toward 1 m without removing continuous samples."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward_far"

    class commands(Solo12V31XYYawRandV2S4InertialFullForwardCfg.commands):
        # With the 20% upward override these become ~35.3% for 1 m and
        # ~16.5% for each lateral endpoint.  The corner and continuous shares
        # remain unchanged from the accepted orientation-stage distribution.
        mixed_short_max_probability = 0.375
        mixed_lateral_endpoint_probability = 0.175


class Solo12V31XYYawRandV2S4InertialFullForwardFarCfgPPO(
    Solo12V31XYYawRandV2S4InertialFullForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg(
    Solo12V31XYYawRandV2S4InertialFullForwardCfg
):
    """Balanced, continuous Forward command rehearsal without endpoint bias."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_inertial_full_forward_balanced"

    class commands(Solo12V31XYYawRandV2S4InertialFullForwardCfg.commands):
        balanced_command_sampling = True
        # Six exact endpoints share 50%; four continuous groups share 50%.
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfgPPO(
    Solo12V31XYYawRandV2S4InertialFullForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump15UpCfg(
    Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg
):
    """Introduce Go2-style has_jumped randomization, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump15_up"

    class commands(Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg.commands):
        # 50% true upward commands; the other nine endpoint/continuous groups
        # share the remaining 50% equally.
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfg.domain_rand):
        randomize_has_jumped = True
        has_jumped_random_prob = 0.15
        reset_has_jumped = True
        manual_has_jumped_reset_time = 0


class Solo12V31XYYawRandV2S5HasJump15UpCfgPPO(
    Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfgPPO
):
    class algorithm(Solo12V31XYYawRandV2S4InertialFullForwardBalancedCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S5HasJump15ForwardCfg(
    Solo12V31XYYawRandV2S5HasJump15UpCfg
):
    """Rehearse the 15% has_jumped policy with the balanced Forward mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump15_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump15UpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump15ForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump15UpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump30UpCfg(
    Solo12V31XYYawRandV2S5HasJump15ForwardCfg
):
    """Increase Go2-style has_jumped randomization to 30%, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_up"

    class commands(Solo12V31XYYawRandV2S5HasJump15ForwardCfg.commands):
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump15ForwardCfg.domain_rand):
        has_jumped_random_prob = 0.3


class Solo12V31XYYawRandV2S5HasJump30UpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump15ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfg(
    Solo12V31XYYawRandV2S5HasJump30UpCfg
):
    """Low-rate consolidation after 200 effective 30% Upward rounds."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_up_consolidate"


class Solo12V31XYYawRandV2S5HasJump30UpConsolidateCfgPPO(
    Solo12V31XYYawRandV2S5HasJump30UpCfgPPO
):
    class algorithm(Solo12V31XYYawRandV2S5HasJump30UpCfgPPO.algorithm):
        learning_rate = 2.e-5


class Solo12V31XYYawRandV2S5HasJump30ForwardCfg(
    Solo12V31XYYawRandV2S5HasJump30UpCfg
):
    """Rehearse the 30% has_jumped policy with the balanced Forward mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump30_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump30UpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump30ForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump30UpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45UpCfg(
    Solo12V31XYYawRandV2S5HasJump30ForwardCfg
):
    """Bridge the conditional has_jumped distribution from 30% to 60%."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_up"

    class commands(Solo12V31XYYawRandV2S5HasJump30ForwardCfg.commands):
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump30ForwardCfg.domain_rand):
        has_jumped_random_prob = 0.45


class Solo12V31XYYawRandV2S5HasJump45UpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump30ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfg(
    Solo12V31XYYawRandV2S5HasJump30ForwardCfg
):
    """Retry 45% has_jumped with a viable synthetic-state duration."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_shortreset_up"

    class commands(Solo12V31XYYawRandV2S5HasJump30ForwardCfg.commands):
        # Keep the original attributable ten-bucket Upward distribution.
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump30ForwardCfg.domain_rand):
        has_jumped_random_prob = 0.45
        # Controlled evaluation shows steps 1--5 remain viable; by step 10
        # take-off vz is halved and by step 20 no valid jump remains.
        has_jumped_reset_step_range = [1, 5]


class Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump30ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg(
    Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfg
):
    """Transfer the corrected 45% state randomization to Forward commands."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_shortreset_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ShortResetUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfg(
    Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg
):
    """Raise only the corrected has_jumped probability from 45% to 60%."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_shortreset_up"

    class commands(Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg.commands):
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfg.domain_rand):
        has_jumped_random_prob = 0.6


class Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ShortResetForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg(
    Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfg
):
    """Rehearse corrected 60% has_jumped with the balanced Forward mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_shortreset_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump60ShortResetUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2LandingRecoveryV1Cfg(
    Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg
):
    """Recover stable real landings without changing model17900 randomisation."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_landing_recovery_v1"

    class env(Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.env):
        # Leave enough time for take-off, flight, 0.25 s impact settling and a
        # complete three-second standing window.
        episode_length_s = 5.0
        landing_stability_seconds = 3.0
        landing_reward_grace_seconds = 0.25
        landing_reward_stability_seconds = 3.0

    class rewards(Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.rewards):
        stance_height_target = 0.30
        landing_lin_vel_sigma = 0.04
        landing_ang_vel_sigma = 0.25
        landing_level_orientation_sigma = 0.04
        landing_dof_vel_sigma = 1.0

        class scales(
            Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.rewards.scales
        ):
            # These terms are gated by a real flight-to-contact transition and
            # begin only after the impact grace period. Synthetic has_jumped
            # states retain the exact short-reset behaviour learned by 17900.
            post_landing_contact_fraction = 6.0
            post_landing_all_feet = 6.0
            post_landing_low_lin_vel = 2.0
            post_landing_low_ang_vel = 4.0
            post_landing_level_orientation = 4.0
            post_landing_low_dof_vel = 2.0
            stable_standing_complete = 500.0
            # Applied after Ji22 reward shaping, analogous to termination.
            post_landing_failure = -200.0


class Solo12V31XYYawRandV2LandingRecoveryV1CfgPPO(
    Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO
):
    class algorithm(
        Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO.algorithm
    ):
        learning_rate = 2.e-5
        schedule = "fixed"


class Solo12V31XYYawRandV2LandingRecoveryV2Cfg(
    Solo12V31XYYawRandV2LandingRecoveryV1Cfg
):
    """Learn landing recovery from dense, mildly disturbed stance resets."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_landing_recovery_v2"

    class domain_rand(Solo12V31XYYawRandV2LandingRecoveryV1Cfg.domain_rand):
        # Training-only: the remaining 80% retain the complete model17900 jump
        # distribution and all randomisation inherited by landing recovery v1.
        landing_recovery_state_probability = 0.20
        landing_recovery_stance_height = 0.30
        landing_recovery_height_offset_range = [-0.015, 0.015]
        landing_recovery_roll_pitch_range = [-0.05, 0.05]
        landing_recovery_yaw_error_range = [-0.05, 0.05]
        landing_recovery_lin_vel_range = [-0.15, 0.15]
        landing_recovery_vertical_vel_range = [-0.15, 0.05]
        landing_recovery_ang_vel_range = [-0.30, 0.30]
        landing_recovery_dof_pos_offset_range = [-0.03, 0.03]
        landing_recovery_dof_vel_range = [-0.50, 0.50]


class Solo12V31XYYawRandV2LandingRecoveryV2CfgPPO(
    Solo12V31XYYawRandV2LandingRecoveryV1CfgPPO
):
    pass


class Solo12V31XYYawRandV2S6SurfaceMildUpCfg(
    Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg
):
    """Add only mild ground friction and restitution randomization."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s6_surface_mild_up"

    class commands(Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.commands):
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.domain_rand):
        randomize_friction = True
        randomize_restitution = True

        class ranges(Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfg.domain_rand.ranges):
            friction_range = [0.5, 1.5]
            restitution_range = [0.0, 0.2]


class Solo12V31XYYawRandV2S6SurfaceMildUpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump60ShortResetForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S6SurfaceMildForwardCfg(
    Solo12V31XYYawRandV2S6SurfaceMildUpCfg
):
    """Transfer mild surface robustness back to the Forward command mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s6_surface_mild_forward"

    class commands(Solo12V31XYYawRandV2S6SurfaceMildUpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S6SurfaceMildForwardCfgPPO(
    Solo12V31XYYawRandV2S6SurfaceMildUpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardCfg(
    Solo12V31XYYawRandV2S5HasJump45UpCfg
):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump45UpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump45ForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45UpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardCfg
):
    """Recover Forward flight margin without changing the balanced command mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_vz350"

    class rewards(Solo12V31XYYawRandV2S5HasJump45ForwardCfg.rewards):
        class scales(Solo12V31XYYawRandV2S5HasJump45ForwardCfg.rewards.scales):
            # The 45% Forward policy is accurate but settles at 0.29--0.30 s
            # for lateral/mixed randomized commands.  Strengthen only the
            # existing 1.8 m/s take-off target; all command buckets and domain
            # randomisation remain identical to the accepted Forward stage.
            takeoff_vertical_velocity = 350.0


class Solo12V31XYYawRandV2S5HasJump45ForwardVz350CfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardVz400Cfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg
):
    """Add a small final take-off margin for randomized mixed commands."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_vz400"

    class rewards(Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg.rewards):
        class scales(Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg.rewards.scales):
            takeoff_vertical_velocity = 400.0


class Solo12V31XYYawRandV2S5HasJump45ForwardVz400CfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardVz350CfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg
):
    """Add uniformly balanced mixed endpoints while retaining 50% continuity."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners"

    class commands(Solo12V31XYYawRandV2S5HasJump45ForwardVz350Cfg.commands):
        balanced_include_corner_anchors = True
        balanced_corner_x = 0.5
        # Ten exact endpoints share 50%; four continuous groups share 50%.
        balanced_command_probabilities = [0.05] * 10 + [0.125] * 4


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardVz350CfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg
):
    """Balance endpoint command families while retaining 50% continuity."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners_grouped"

    class commands(Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg.commands):
        # Zero, x=1 m, lateral, yaw, and signed mixed endpoints each receive
        # 10% as a command family.  Signs are uniform within each family.
        balanced_command_probabilities = (
            [0.10, 0.10]
            + [0.05] * 4
            + [0.025] * 4
            + [0.125] * 4
        )


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersGroupedCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg
):
    """Retain learned corners while restoring uniform primary endpoints."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump45_forward_corners_retain"

    class commands(Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfg.commands):
        # The six primary endpoints are strictly uniform (48% total), the
        # learned signed corners receive a 2% retention share, and the four
        # continuous families remain exactly 50% in total.
        balanced_command_probabilities = (
            [0.08] * 6
            + [0.005] * 4
            + [0.125] * 4
        )


class Solo12V31XYYawRandV2S5HasJump45ForwardCornersRetainCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardCornersCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump60UpCfg(
    Solo12V31XYYawRandV2S5HasJump45ForwardCfg
):
    """Increase Go2-style has_jumped randomization to 60%, upward-heavy."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_up"

    class commands(Solo12V31XYYawRandV2S5HasJump45ForwardCfg.commands):
        balanced_command_probabilities = [0.5] + [0.5 / 9.0] * 9

    class domain_rand(Solo12V31XYYawRandV2S5HasJump45ForwardCfg.domain_rand):
        has_jumped_random_prob = 0.6


class Solo12V31XYYawRandV2S5HasJump60UpCfgPPO(
    Solo12V31XYYawRandV2S5HasJump45ForwardCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfg(
    Solo12V31XYYawRandV2S5HasJump60UpCfg
):
    """Low-rate consolidation after the best first 100 effective 60% Upward rounds."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_up_consolidate"


class Solo12V31XYYawRandV2S5HasJump60UpConsolidateCfgPPO(
    Solo12V31XYYawRandV2S5HasJump60UpCfgPPO
):
    class algorithm(Solo12V31XYYawRandV2S5HasJump60UpCfgPPO.algorithm):
        learning_rate = 2.e-5


class Solo12V31XYYawRandV2S5HasJump60ForwardCfg(
    Solo12V31XYYawRandV2S5HasJump60UpCfg
):
    """Rehearse the 60% has_jumped policy with the balanced Forward mix."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_forward"

    class commands(Solo12V31XYYawRandV2S5HasJump60UpCfg.commands):
        balanced_command_probabilities = [1.0 / 12.0] * 6 + [1.0 / 8.0] * 4


class Solo12V31XYYawRandV2S5HasJump60ForwardCfgPPO(
    Solo12V31XYYawRandV2S5HasJump60UpCfgPPO
):
    pass


class Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfg(
    Solo12V31XYYawRandV2S5HasJump60ForwardCfg
):
    """Low-rate consolidation after 500 effective 60% Forward rounds."""

    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_hasjump60_forward_consolidate"


class Solo12V31XYYawRandV2S5HasJump60ForwardConsolidateCfgPPO(
    Solo12V31XYYawRandV2S5HasJump60ForwardCfgPPO
):
    class algorithm(Solo12V31XYYawRandV2S5HasJump60ForwardCfgPPO.algorithm):
        learning_rate = 2.e-5


class Solo12V31XYYawRandV2S3InitialStateCfg(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s3_initial_state"

    class domain_rand(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg.domain_rand):
        pos_vel_random_prob = 0.4
        randomize_robot_ori = True
        randomize_dof_pos = True

        class ranges(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfg.domain_rand.ranges):
            min_ori_euler = [0.0, 0.0, -0.1]
            max_ori_euler = [0.0, 0.0, 0.1]


class Solo12V31XYYawRandV2S3InitialStateCfgPPO(Solo12V31XYYawRandV2S2Latency30Jitter5ForwardCfgPPO):
    pass


class Solo12V31XYYawRandV2S4CalibrationCfg(Solo12V31XYYawRandV2S3InitialStateCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_calibration"

    class domain_rand(Solo12V31XYYawRandV2S3InitialStateCfg.domain_rand):
        randomize_spring_params = True
        randomize_motor_offset = True

        class ranges(Solo12V31XYYawRandV2S3InitialStateCfg.domain_rand.ranges):
            spring_stiffness_percentage = 0.3
            spring_damping_percentage = 0.3
            spring_rest_pos_range = [-0.1, 0.1]
            motor_offset_range = [-0.02, 0.02]


class Solo12V31XYYawRandV2S4CalibrationCfgPPO(Solo12V31XYYawRandV2S3InitialStateCfgPPO):
    pass


class Solo12V31XYYawRandV2S4NoiseCfg(Solo12V31XYYawRandV2S4CalibrationCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s4_noise"

    class noise(Solo12V31XYYawRandV2S4CalibrationCfg.noise):
        add_noise = True


class Solo12V31XYYawRandV2S4NoiseCfgPPO(Solo12V31XYYawRandV2S4CalibrationCfgPPO):
    pass


class Solo12V31XYYawRandV2S5InertialMildCfg(Solo12V31XYYawRandV2S4NoiseCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_inertial_mild"

    class domain_rand(Solo12V31XYYawRandV2S4NoiseCfg.domain_rand):
        randomize_base_mass = True
        randomize_com = True
        randomize_link_mass = True

        class ranges(Solo12V31XYYawRandV2S4NoiseCfg.domain_rand.ranges):
            added_mass_range = [-0.25, 0.5]
            com_displacement_range = [-0.02, 0.02]
            added_link_mass_range = [0.9, 1.1]


class Solo12V31XYYawRandV2S5InertialMildCfgPPO(Solo12V31XYYawRandV2S4NoiseCfgPPO):
    pass


class Solo12V31XYYawRandV2S5InertialFullCfg(Solo12V31XYYawRandV2S5InertialMildCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s5_inertial_full"

    class domain_rand(Solo12V31XYYawRandV2S5InertialMildCfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S5InertialMildCfg.domain_rand.ranges):
            added_mass_range = [-0.5, 1.0]
            com_displacement_range = [-0.04, 0.04]
            added_link_mass_range = [0.8, 1.2]


class Solo12V31XYYawRandV2S5InertialFullCfgPPO(Solo12V31XYYawRandV2S5InertialMildCfgPPO):
    pass


class Solo12V31XYYawRandV2S6ContactMildCfg(Solo12V31XYYawRandV2S5InertialFullCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s6_contact_mild"

    class domain_rand(Solo12V31XYYawRandV2S5InertialFullCfg.domain_rand):
        randomize_friction = True
        randomize_restitution = True
        randomize_joint_friction = True
        randomize_joint_damping = True

        class ranges(Solo12V31XYYawRandV2S5InertialFullCfg.domain_rand.ranges):
            friction_range = [0.5, 1.5]
            restitution_range = [0.0, 0.2]
            joint_friction_range = [0.0, 0.02]
            joint_damping_range = [0.0, 0.005]


class Solo12V31XYYawRandV2S6ContactMildCfgPPO(Solo12V31XYYawRandV2S5InertialFullCfgPPO):
    pass


class Solo12V31XYYawRandV2S6ContactFullCfg(Solo12V31XYYawRandV2S6ContactMildCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s6_contact_full"

    class domain_rand(Solo12V31XYYawRandV2S6ContactMildCfg.domain_rand):
        class ranges(Solo12V31XYYawRandV2S6ContactMildCfg.domain_rand.ranges):
            friction_range = [0.01, 3.0]
            restitution_range = [0.0, 0.4]
            joint_friction_range = [0.0, 0.04]
            joint_damping_range = [0.0, 0.01]


class Solo12V31XYYawRandV2S6ContactFullCfgPPO(Solo12V31XYYawRandV2S6ContactMildCfgPPO):
    pass


class Solo12V31XYYawRandV2S7HasJump15Cfg(Solo12V31XYYawRandV2S6ContactFullCfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s7_hasjump15"

    class domain_rand(Solo12V31XYYawRandV2S6ContactFullCfg.domain_rand):
        randomize_has_jumped = True
        has_jumped_random_prob = 0.15
        reset_has_jumped = True
        manual_has_jumped_reset_time = 0


class Solo12V31XYYawRandV2S7HasJump15CfgPPO(Solo12V31XYYawRandV2S6ContactFullCfgPPO):
    class algorithm(Solo12V31XYYawRandV2S6ContactFullCfgPPO.algorithm):
        learning_rate = 5.e-5


class Solo12V31XYYawRandV2S7HasJump30Cfg(Solo12V31XYYawRandV2S7HasJump15Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s7_hasjump30"

    class domain_rand(Solo12V31XYYawRandV2S7HasJump15Cfg.domain_rand):
        has_jumped_random_prob = 0.3


class Solo12V31XYYawRandV2S7HasJump30CfgPPO(Solo12V31XYYawRandV2S7HasJump15CfgPPO):
    pass


class Solo12V31XYYawRandV2S7HasJump60Cfg(Solo12V31XYYawRandV2S7HasJump30Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s7_hasjump60"

    class domain_rand(Solo12V31XYYawRandV2S7HasJump30Cfg.domain_rand):
        has_jumped_random_prob = 0.6


class Solo12V31XYYawRandV2S7HasJump60CfgPPO(Solo12V31XYYawRandV2S7HasJump30CfgPPO):
    pass


class Solo12V31XYYawRandV2S8Push02Cfg(Solo12V31XYYawRandV2S7HasJump60Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s8_push02"

    class domain_rand(Solo12V31XYYawRandV2S7HasJump60Cfg.domain_rand):
        push_robots = True
        push_interval_s = 1.0
        max_push_vel_xy = 0.2


class Solo12V31XYYawRandV2S8Push02CfgPPO(Solo12V31XYYawRandV2S7HasJump60CfgPPO):
    pass


class Solo12V31XYYawRandV2S8Push04Cfg(Solo12V31XYYawRandV2S8Push02Cfg):
    task_name = "solo12_v3_1_xyyaw_rand_v2_s8_push04"

    class domain_rand(Solo12V31XYYawRandV2S8Push02Cfg.domain_rand):
        max_push_vel_xy = 0.4


class Solo12V31XYYawRandV2S8Push04CfgPPO(Solo12V31XYYawRandV2S8Push02CfgPPO):
    pass
