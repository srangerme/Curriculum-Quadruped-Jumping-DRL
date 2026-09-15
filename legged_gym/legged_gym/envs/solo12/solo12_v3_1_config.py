"""Second-stage forward-jump training configuration for Solo12 v3_1."""

from legged_gym.envs.go2.go2_config import Go2Cfg, Go2CfgPPO
from legged_gym.envs.solo12.solo12_v3_1_common import (
    Solo12V31Asset,
    Solo12V31Control,
    Solo12V31Imu,
    Solo12V31InitState,
    Solo12V31Morphology,
    Solo12V31PhysicalRandomizationRanges,
    Solo12V31Sim,
    Solo12V31Viewer,
)


class Solo12V31Cfg(Go2Cfg):
    task_name = "solo12_v3_1_forward"

    class init_state(Solo12V31InitState):
        # Match the accepted upward c13800 reset distribution.
        pos = [0.0, 0.0, 0.32]
    morphology = Solo12V31Morphology
    control = Solo12V31Control
    imu = Solo12V31Imu
    asset = Solo12V31Asset
    sim = Solo12V31Sim
    viewer = Solo12V31Viewer

    class env(Go2Cfg.env):
        reset_height = 0.12
        settled_height_threshold = 0.38
        debug_draw = True
        debug_draw_line_goal = False
        debug_draw_ground_grid = True
        ground_grid_spacing = 0.1
        ground_grid_major_spacing = 0.5
        ground_grid_x_range = [-0.5, 1.0]
        ground_grid_y_range = [-0.5, 0.5]
        settled_contact_count = 3
        # Preserve the reset/action phase used by the accepted upward policy.
        initial_zero_action_steps = 1
        initial_contact_history_probability = 1.0
        initial_contact_settle_steps = 1
        initial_contact_settle_steps_max = 1
        initial_contact_base_height_offset = 0.0
        initial_stance_height_range = [0.28, 0.34]
        initial_stance_height_probability = 1.0
        contact_observation_delay_max_steps = 2
        # F1 must reach terminal rewards throughout its 0-0.30 m range.
        reset_landing_error = 0.35
        # Position commands occupy one column per axis in a 1014-D actor
        # observation. Bring the 0-0.30 m F1 signal to order-one magnitude.
        command_position_observation_scale = 25.0
        # Use the same metric-position scale for body-frame x and y commands.
        # The previous 4.0 scale made lateral targets 6.25x less visible than
        # longitudinal targets and flattened the learned y-to-action mapping.
        command_lateral_observation_scale = 25.0
        observe_joint_friction_in_command_y = False

    class commands(Go2Cfg.commands):
        # Forward is a command-conditioned task, so the full active stage
        # range must remain represented. The legacy success curriculum moves
        # every environment into its highest bin and destroys that mixture.
        curriculum = False
        randomize_commands = True
        # F4 commands are expressed in the robot's pre-takeoff heading frame.
        # Keep the existing world-frame reward internals while presenting
        # body-frame x/y and relative yaw to the policy.
        initial_body_command_frame = True
        randomize_yaw = False
        yaw_range = [-1.5707963267948966, 1.5707963267948966]
        # Preserve continuous coverage while injecting high-contrast command
        # pairs that prevent an Upward-initialized actor from optimizing a
        # single shared compromise displacement.
        endpoint_sampling_probability = 0.4
        endpoint_sampling_min_x = 0.1
        endpoint_sampling_max_x = 0.3
        intermediate_anchor_sampling_probability = 0.3
        intermediate_anchor_sampling_x = [0.3, 0.6]
        # Optional F4 categorical command mixture. Modes are ordered as:
        # pure_x, carrier_y, pure_y, carrier_yaw, pure_yaw, joint_xyyaw.
        # Disabled by default so accepted F3 behaviour is unchanged.
        structured_mode_probabilities = []
        structured_carrier_x = 0.3
        y_endpoint_sampling_probability = 0.0
        yaw_endpoint_sampling_probability = 0.0

        class ranges(Go2Cfg.commands.ranges):
            pos_dx_ini = [0.0, 0.3]
            pos_dy_ini = [0.0, 0.0]

    class domain_rand(Go2Cfg.domain_rand):
        # Forward deployment and acceptance do not inject velocity impulses.
        # Disable both legacy helpers that directly overwrite base velocity;
        # physical/sensor domain randomization remains enabled below.
        push_robots = False
        push_towards_goal = False
        push_towards_goal_probability = 0.0
        push_towards_goal_final_probability = 0.0
        # Preserve the latency coverage that made the 20/30/40 ms endpoints
        # learnable in the upward stage.  The command line selects the current
        # maximum; these probabilities select zero/max/interior regimes.
        zero_latency_probability = 0.5
        max_latency_probability = 0.25
        joint_friction_endpoint_probability = 0.0
        # Preserve continuous coverage while explicitly exposing the limiting
        # high-friction endpoint. Do not oversample the zero-friction endpoint.
        joint_friction_min_endpoint_probability = 0.0
        joint_friction_max_endpoint_probability = 0.2
        # Diagnostic-only residual angular impulse at the transition from
        # crouch to propulsion. Disabled by default for all accepted runs.
        takeoff_angvel_kick_max = [0.0, 0.0, 0.0]
        takeoff_angvel_kick_probability = 0.0
        # Diagnostic-only, one-control-step redistribution of vertical foot
        # forces during late stance. The four forces sum to zero, so this
        # perturbs contact timing/angular impulse without adding net thrust.
        takeoff_contact_force_perturb_max = [0.0, 0.0]
        takeoff_contact_force_perturb_probability = 0.0
        takeoff_contact_force_perturb_trigger_vz = 0.5

        class ranges(Solo12V31PhysicalRandomizationRanges, Go2Cfg.domain_rand.ranges):
            # Match the accepted c8800 Upward distribution at the Forward
            # transition instead of reverting to the broad legacy Solo COM.
            # Forward assumes no external payload. Keep base-mass uncertainty
            # symmetric and bounded to roughly +/-5% of total URDF mass.
            added_mass_range = [-0.5, 0.5]
            # The inherited 0.01 endpoint is traction-limited for the Solo12
            # far-jump task. Fixed-factor sweeps show 0.3 is the lowest tested
            # endpoint that remains viable with the full Forward dynamics
            # distribution. The 0.3 and 0.5 endpoints fail recovery when the
            # other randomized dynamics are active; 0.7 is the tested floor.
            friction_range = [0.7, 3.0]
            com_displacement_range = [
                [-0.02, -0.02, -0.02],
                [0.02, 0.02, 0.02],
            ]
            latency_range = [0.0, 40.0]
            restitution_range = [0.0, 0.2]
            joint_friction_range = [0.0, 0.02]
            joint_damping_range = [0.0, 0.01]

    class rewards(Go2Cfg.rewards):
        # The legacy 0.05 kernel gives nearly full terminal position reward
        # even at 6-10 cm error, so PPO can optimize one compromise jump for
        # every command. Keep the reward maximum unchanged while making the
        # F1 5 cm acceptance tolerance visible in the reward landscape.
        command_pos_tracking_sigma = 0.01
        task_pos_at_first_landing = False
        # Forward targets controlled distance rather than maximum jump height.
        # Keep the existing target reward centered in the accepted 0.60-0.70 m
        # band instead of inheriting Go2's 0.90 m target.
        max_height_target = 0.65
        # Preserve the nominal objective while preventing high joint-friction
        # samples from being ignored by the mixed-domain PPO objective.
        joint_friction_position_reward_boost = 0.0
        stance_height_target = 0.30
        squat_height_target = 0.188
        feet_tuck_height_target = -0.141
        feet_tuck_activation_height = 0.42
        # Solo's shorter legs need more time to leave the tucked pose before
        # the calf links can reach the ground.  This only matters when the
        # optional feet_landing_pose reward scale is enabled.
        feet_landing_pose_activation_height = 0.50
        max_contact_force = 120.0

        # Free bands are deliberately permissive. Their scales remain zero
        # until the independent Forward reward ablations establish causality.
        takeoff_pitch_cancellation_free_band = 0.05
        takeoff_front_rear_timing_free_band = 0.025
        takeoff_roll_rate_free_band = 0.25
        # Normalize squared timing excess before applying the event scale.
        # Without this, millisecond-scale errors produce rewards near 1e-5
        # and the configured penalty has no meaningful PPO gradient.
        takeoff_front_rear_timing_sigma = 0.0004
        takeoff_front_rear_vertical_impulse_free_band = 0.15
        takeoff_front_rear_vertical_impulse_sigma = 0.04
        # check_jump() computes this shared diagnostic for both jump tasks.
        # Forward inherits Go2Cfg rather than Go2UpwardsCfg, so preserve the
        # accepted c8800 event parameters explicitly without enabling its
        # Upward-only reward.
        takeoff_predicted_height_min = 0.60
        takeoff_predicted_height_max = 0.70
        takeoff_predicted_height_sigma = 0.0025
        takeoff_event_pitch_limit = 0.20943951023931956
        takeoff_event_pitch_sigma = 0.007615435494667714

        # Forward-only joint terminal reward. Keep the default disabled until
        # its c8800 single-variable ablation establishes causality.
        forward_jump_quality_height_min = 0.60
        forward_jump_quality_height_max = 0.70
        forward_jump_quality_height_sigma = 0.0025
        forward_jump_quality_height_above_weight = 0.5
        forward_jump_quality_timing_free_band = 0.025
        forward_jump_quality_timing_sigma = 0.0004
        forward_jump_quality_impulse_free_band = 0.15
        forward_jump_quality_impulse_sigma = 0.04
        forward_jump_quality_flight_time = 0.55
        # The horizontal sigma is a broad displacement-error band in metres.
        # Distance is primary; height and synchronization are guardrails.
        forward_jump_quality_horizontal_sigma = 0.40
        forward_jump_quality_vertical_weight = 0.2
        forward_jump_quality_sync_weight = 0.1
        forward_jump_quality_horizontal_weight = 0.7
        forward_takeoff_height_floor_start = 0.35
        forward_takeoff_height_floor_target = 0.50
        # Cross-distribution fit on c15500: landing_xy is predicted from the
        # first-takeoff state with 0.84 cm RMSE. The scale stays disabled until
        # the command-conditioning ablation validates it.
        forward_takeoff_position_bias = 0.0
        forward_takeoff_position_bias_friction_adaptive = True
        forward_takeoff_position_bias_at_max_joint_friction = 0.0
        # At the accepted high-friction capability boundary, preserve the
        # command direction and monotonic response without forcing physically
        # unrealistic full-distance tracking.  This only changes the takeoff
        # landing-position target; safety and stability rewards are unchanged.
        forward_takeoff_position_retention_at_max_joint_friction = 0.70
        forward_takeoff_position_sigma = 0.01
        # Coarse command-position score used only to bring distant commands
        # into the existing Gaussian reward's useful basin.  It is kept
        # separate and disabled by default so the accepted fine reward is
        # unchanged outside explicit ablations.
        forward_takeoff_position_coarse_range = 0.60
        # Optional training-only alignment gate.  When enabled, real episodes
        # must clear jump_success_height before terminal task and post-landing
        # rewards are credited.  Synthetic already-jumped recovery starts are
        # exempt so the recovery curriculum keeps its learning signal.
        task_event_require_success_height = False
        task_pos_lateral_sigma = 0.01
        task_pos_lateral_coarse_range = 0.60
        forward_takeoff_lateral_sigma = 0.01
        # One-shot credit at first takeoff, while ground contact can still
        # generate the commanded yaw angular momentum.
        forward_takeoff_yaw_rate_sigma = 0.09
        pre_takeoff_dof_vel_envelope_free_band = 1.5
        pre_takeoff_dof_vel_envelope_full_cost_ratio = 2.0
        class scales(Go2Cfg.rewards.scales):
            post_landing_dof_vel = -0.5
            post_landing_base_xy_vel = 0.0
            post_landing_target_pos = 0.0
            pre_takeoff_overspeed_drive = 0.0
            pre_takeoff_dof_vel_envelope = 0.0
            takeoff_pitch_cancellation = 0.0
            takeoff_front_rear_timing = 0.0
            takeoff_front_rear_vertical_impulse = 0.0
            takeoff_roll_rate = 0.0
            forward_jump_quality = 0.0
            forward_takeoff_height_floor = 0.0
            forward_takeoff_joint = 0.0
            forward_takeoff_position = 0.0
            forward_takeoff_position_coarse = 0.0
            # F4 requires a lateral gradient that is not suppressed by a
            # simultaneous longitudinal error. The original 2-D task reward
            # remains active and unchanged.
            task_pos_lateral = 0.0
            task_pos_lateral_coarse = 0.0
            forward_takeoff_lateral = 0.0
            forward_takeoff_yaw_rate = 0.0


class Solo12V31CfgPPO(Go2CfgPPO):
    class runner(Go2CfgPPO.runner):
        experiment_name = "test_solo12_v3_1"
