"""Helpers for the robot-independent policy/model interface contract."""

import json
import math
import os
import xml.etree.ElementTree as ET


QUADRUPED_POLICY_DOF_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

QUADRUPED_POLICY_FOOT_NAMES = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)


def build_name_permutation(source_names, target_names, interface_name):
    """Return indices that reorder source values into target-name order."""
    source_names = tuple(source_names)
    target_names = tuple(target_names)

    if len(source_names) != len(set(source_names)):
        raise ValueError(f"duplicate names in {interface_name} source order: {source_names}")
    if len(target_names) != len(set(target_names)):
        raise ValueError(f"duplicate names in {interface_name} target order: {target_names}")

    missing = [name for name in target_names if name not in source_names]
    unexpected = [name for name in source_names if name not in target_names]
    if missing or unexpected:
        raise ValueError(
            f"{interface_name} names do not match: missing={missing}, "
            f"unexpected={unexpected}"
        )

    return tuple(source_names.index(name) for name in target_names)


def rpy_to_rotation_matrix(rpy):
    """Return R_base_imu for a URDF fixed-axis roll/pitch/yaw mount."""
    if len(rpy) != 3:
        raise ValueError(f"IMU rpy must contain three values, got {rpy}")

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rpy_to_quaternion_xyzw(rpy):
    """Return the URDF child-to-parent rotation as an xyzw quaternion."""
    if len(rpy) != 3:
        raise ValueError(f"IMU rpy must contain three values, got {rpy}")

    roll, pitch, yaw = (angle / 2.0 for angle in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def urdf_fixed_joint_mount(xml_text, child_body_name):
    """Return parent/xyz/rpy for the fixed joint mounting a URDF body."""

    root = ET.fromstring(xml_text)
    matching_joints = []
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link") == child_body_name:
            matching_joints.append(joint)
    if len(matching_joints) != 1:
        raise ValueError(
            "expected exactly one parent joint for URDF body {!r}, found {}".format(
                child_body_name, len(matching_joints)
            )
        )
    joint = matching_joints[0]
    if joint.get("type") != "fixed":
        raise ValueError(
            "URDF body {!r} must be mounted by a fixed joint, got {!r}".format(
                child_body_name, joint.get("type")
            )
        )
    parent = joint.find("parent")
    if parent is None or not parent.get("link"):
        raise ValueError("URDF IMU joint is missing its parent link")
    origin = joint.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError("URDF joint origin xyz and rpy must each have three values")
    return parent.get("link"), xyz, rpy


def validate_imu_mount_in_urdf(
    urdf_path, body_name, parent_body_name, position_in_base, rpy_in_base,
    tolerance=1e-7
):
    """Fail fast when deployment IMU metadata diverges from the robot URDF."""

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        actual_parent, actual_position, actual_rpy = urdf_fixed_joint_mount(
            urdf_file.read(), body_name
        )
    expected_position = tuple(float(value) for value in position_in_base)
    expected_rpy = tuple(float(value) for value in rpy_in_base)
    if actual_parent != parent_body_name:
        raise ValueError(
            "configured IMU parent {!r} does not match URDF parent {!r}".format(
                parent_body_name, actual_parent
            )
        )
    if len(expected_position) != 3 or len(expected_rpy) != 3:
        raise ValueError("configured IMU xyz and rpy must each have three values")
    position_error = max(
        abs(actual - expected)
        for actual, expected in zip(actual_position, expected_position)
    )
    rpy_error = max(
        abs(actual - expected)
        for actual, expected in zip(actual_rpy, expected_rpy)
    )
    if position_error > tolerance or rpy_error > tolerance:
        raise ValueError(
            "configured IMU mount does not match URDF: position_error={}, "
            "rpy_error={}".format(position_error, rpy_error)
        )
    return {
        "parent_body_name": actual_parent,
        "position_in_base": actual_position,
        "rpy_in_base": actual_rpy,
    }


def model_interface_dict(cfg):
    """Build serializable deployment metadata from an environment config."""
    policy_dof_names = tuple(cfg.asset.policy_dof_names)
    policy_foot_names = tuple(cfg.asset.policy_foot_names)
    if not policy_dof_names or not policy_foot_names:
        raise ValueError("policy DOF and foot names must be explicit before export")
    if cfg.imu.policy_frame != "base":
        raise ValueError(
            f"unsupported policy IMU frame {cfg.imu.policy_frame!r}; expected 'base'"
        )

    position = tuple(float(value) for value in cfg.imu.position_in_base)
    rpy = tuple(float(value) for value in cfg.imu.rpy_in_base)
    if len(position) != 3:
        raise ValueError(
            f"IMU position_in_base must contain three values, got {position}"
        )

    return {
        "schema_version": 1,
        "policy_joint_names": list(policy_dof_names),
        "policy_foot_names": list(policy_foot_names),
        "policy_imu_frame": "base",
        "imu_mount": {
            "body_name": cfg.imu.body_name,
            "parent_body_name": cfg.imu.parent_body_name,
            "position_in_base_m": list(position),
            "rpy_in_base_rad": list(rpy),
            "rotation_imu_to_base": [list(row) for row in rpy_to_rotation_matrix(rpy)],
            "quaternion_imu_to_base_xyzw": list(rpy_to_quaternion_xyzw(rpy)),
        },
    }


def export_model_interface_config(cfg, path):
    """Write the deployment-side model interface next to an exported policy."""
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, "model_interface.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(model_interface_dict(cfg), output_file, indent=2)
        output_file.write("\n")
    return output_path
