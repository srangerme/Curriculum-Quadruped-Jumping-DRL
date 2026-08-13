"""Robot-specific settings shared by the Solo12 v3_1 jumping tasks."""

import torch

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


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
    # More responsive first-stage controller for explosive take-off. These
    # values still respect the effort limits declared by the URDF.
    stiffness = {"joint": 20.0}
    damping = {"joint": 0.5}
    action_scale = 0.30
    hip_scale_reduction = 0.5
    decimation = 4
    use_action_filter = True
    filter_freq = 8.0
    filter_type = "EMA"
    butterworth_order = 2
    safety_clip_actions = True


class Solo12V31Asset(LeggedRobotCfg.asset):
    file = "/workspace/models/solo12_v3_1/urdf/solo12_v3_1.urdf"
    name = "solo12_v3_1"
    base_body_name = "Body"
    foot_name = "foot"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["Body", "thigh", "calf", "hip"]
    collapse_fixed_joints = True
    self_collisions = 0
    flip_visual_attachments = False
    fix_base_link = False
    armature = 0.0
    use_physx_armature = False


class Solo12V31Viewer:
    camera_track_robot = False
    ref_env = 0
    pos = [2.5, -2.0, 0.45]
    lookat = [0.0, 0.0, 0.30]
    simulate_camera = False
