"""Robot-specific settings shared by the Solo12 v3_1 jumping tasks."""

import torch

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.model_interface import (
    QUADRUPED_POLICY_DOF_NAMES,
    QUADRUPED_POLICY_FOOT_NAMES,
)


SOLO12_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.0,
    "RL_hip_joint": 0.0,
    "FR_hip_joint": 0.0,
    "RR_hip_joint": 0.0,
    "FL_thigh_joint": 0.7220,
    "RL_thigh_joint": 0.7220,
    "FR_thigh_joint": 0.7220,
    "RR_thigh_joint": 0.7220,
    "FL_calf_joint": -1.4441,
    "RL_calf_joint": -1.4441,
    "FR_calf_joint": -1.4441,
    "RR_calf_joint": -1.4441,
}


class Solo12V31InitState(LeggedRobotCfg.init_state):
    # With q_thigh=0.722 and q_calf=-1.444, the two 0.2 m links put
    # the feet approximately 0.30 m below the hip axes.
    pos = [0.0, 0.0, 0.30]
    rel_foot_pos = [
        [0.17675, 0.17675, -0.1775, -0.1775],
        [0.1324, -0.1324, 0.1324, -0.1324],
        [-0.30, -0.30, -0.30, -0.30],
    ]
    default_joint_angles = SOLO12_DEFAULT_JOINT_ANGLES

    K_HIP = 16.0
    K_THIGH = 16.0
    K_CALF = 20.0
    D_HIP = 0.4
    D_THIGH = 0.4
    D_CALF = 0.5
    DEFAULT_HIP_ANGLE = 0.0
    DEFAULT_THIGH_ANGLE = 0.7220
    DEFAULT_CALF_ANGLE = -1.4441
    spring_stiffness = torch.tensor([K_HIP, K_THIGH, K_CALF]).repeat(1, 4)
    spring_damping = torch.tensor([D_HIP, D_THIGH, D_CALF]).repeat(1, 4)
    spring_rest_pos = torch.tensor(
        [DEFAULT_HIP_ANGLE, DEFAULT_THIGH_ANGLE, DEFAULT_CALF_ANGLE]
    ).repeat(1, 4)


class Solo12V31Morphology(LeggedRobotCfg.morphology):
    hip_link_length = 0.0799
    thigh_link_length = 0.20
    calf_link_length = 0.20
    hip_x = 0.1770
    hip_y = 0.0525


class Solo12V31Control(LeggedRobotCfg.control):
    control_type = "P"
    # Validated common Solo12 control authority for upward and forward stages.
    stiffness = {"joint": 16.0}
    damping = {"joint": 0.5}
    action_scale = 0.25
    hip_scale_reduction = 0.5
    decimation = 4
    use_action_filter = True
    filter_freq = 8.0
    filter_type = "EMA"
    butterworth_order = 2
    safety_clip_actions = True
    # Disabled for baseline compatibility; repair experiments opt in through
    # train.py so the deployment-side slew limit can use the same value.
    max_action_delta = 0.0
    velocity_torque_envelope = False
    velocity_torque_limit_scale = 1.0
    velocity_torque_envelope_blend = 1.0
    velocity_cost_source = "envelope_rejection"
    velocity_cost_free_band = 0.90
    velocity_cost_full_ratio = 1.0
    velocity_torque_soft_limit_ratio = 0.9
    # Do not inject controller-side braking above the rated speed.  The
    # envelope only removes torque that would accelerate farther into
    # overspeed; policy-requested braking remains available unchanged.
    velocity_torque_overspeed_kd = 0.0


class Solo12V31Sim(LeggedRobotCfg.sim):
    """Physics settings shared by Solo12 upward and forward jump tasks."""

    dt = 0.005
    substeps = 1

    class physx(LeggedRobotCfg.sim.physx):
        solver_type = 1
        # Four TGS position iterations allowed Solo's small spherical feet to
        # penetrate the flat triangle mesh during landing.  Eight iterations
        # removes that non-physical calf-contact termination without changing
        # the physics step or policy/control frequency.
        num_position_iterations = 8
        max_depenetration_velocity = 1.0


class Solo12V31PhysicalRandomizationRanges:
    """Solo12 model uncertainty shared by every jump-training stage."""

    # Match Go2's relative uncertainty strength. Absolute mass and COM
    # offsets are scaled by the Solo/Go2 URDF base masses and link geometry.
    motor_strength_ranges = [0.9, 1.1]
    p_gains_range = [0.9, 1.1]
    d_gains_range = [0.9, 1.1]
    latency_range = [0.0, 20.0]
    added_mass_range = [-0.635, 1.906]
    com_displacement_range = [
        [-0.0915, -0.1129, -0.0939],
        [0.0915, 0.1129, 0.0939],
    ]
    added_link_mass_range = [0.7, 1.3]
    physical_dof_velocity_limit_scale_range = [1.0, 1.0]


class Solo12V31Asset(LeggedRobotCfg.asset):
    file = "/workspace/models/solo12_v3_1/urdf/solo12_v3_1.urdf"
    name = "solo12_v3_1"
    base_body_name = "Body"
    foot_name = "foot"
    policy_dof_names = QUADRUPED_POLICY_DOF_NAMES
    policy_foot_names = QUADRUPED_POLICY_FOOT_NAMES
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["Body", "thigh", "calf", "hip"]
    collapse_fixed_joints = True
    self_collisions = 0
    flip_visual_attachments = False
    fix_base_link = False
    armature = 0.0
    use_physx_armature = False
    physical_dof_velocity_limit_override = None
    physical_dof_velocity_limit_scale = None


class Solo12V31Imu(LeggedRobotCfg.imu):
    """Physical Solo12 IMU mount; policy measurements remain base-frame."""

    body_name = "imu"
    parent_body_name = "Body"
    position_in_base = [-0.063487, -0.031976, -0.011172]
    rpy_in_base = [0.0, 1.5707963, 3.1415927]


class Solo12V31Viewer:
    camera_track_robot = False
    ref_env = 0
    pos = [2.5, -2.0, 0.45]
    lookat = [0.0, 0.0, 0.30]
    simulate_camera = False
