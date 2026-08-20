import unittest
from types import SimpleNamespace

import isaacgym  # noqa: F401; project imports require Isaac Gym before torch.
import torch
from legged_gym.envs import task_registry  # noqa: F401; complete task registration first.
from legged_gym.envs.base.legged_robot import LeggedRobot

from legged_gym.utils.checkpoint_evaluation import (
    PROTOCOL_VERSION,
    build_episode_quotas,
    compare_evaluations,
    summarize_episodes,
)
from legged_gym.utils.model_interface import (
    build_name_permutation,
    urdf_fixed_joint_mount,
)


def result(task, mean, median, success):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "jump_type": "upward",
        "load_run": "run",
        "checkpoint": 3000,
        "mode": "nominal",
        "seed": 1,
        "num_envs": 128,
        "eval_episodes": 512,
        "target": {"x": 0.0, "y": 0.0, "z": 0.0},
        "success_height": 0.5,
        "action_noise_scale": 0.0,
        "fixed_physics": {
            "restitution": None,
            "joint_friction": None,
            "joint_damping": None,
        },
        "episode_sampling": "balanced-per-environment",
        "action_source": "policy",
        "metrics": {
            "height_mean": mean,
            "height_median": median,
            "success_rate": success,
        },
    }


class CheckpointEvaluationTest(unittest.TestCase):
    def test_descending_landing_pose_reward_phase_gate(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 4
        robot.device = "cpu"
        robot.root_states = torch.zeros(4, 13)
        robot.root_states[:, 2] = 0.45
        robot.root_states[:, 6] = 1.0
        # Descending, ascending, never took off, and already landed.
        robot.root_states[:, 9] = torch.tensor([-1.0, 1.0, -1.0, -1.0])
        robot.base_quat = robot.root_states[:, 3:7]
        target = [
            [0.17675, 0.17675, -0.1775, -0.1775],
            [0.1324, -0.1324, 0.1324, -0.1324],
            [-0.30, -0.30, -0.30, -0.30],
        ]
        feet_body = torch.tensor(target).transpose(1, 0)
        robot.feet_pos = feet_body.unsqueeze(0).repeat(4, 1, 1)
        robot.feet_pos[:, :, 2] += robot.root_states[:, 2].view(-1, 1)
        robot.was_in_flight = torch.tensor([True, True, False, True])
        robot.mid_air = torch.tensor([True, True, True, False])
        robot.has_jumped = torch.tensor([False, False, False, True])
        robot.cfg = SimpleNamespace(
            init_state=SimpleNamespace(rel_foot_pos=target),
            rewards=SimpleNamespace(
                feet_landing_pose_sigma=0.01,
                feet_landing_pose_activation_height=0.50,
            ),
        )
        robot.get_terrain_height = lambda xy: torch.zeros(xy.shape[0])

        reward = robot._reward_feet_landing_pose()
        torch.testing.assert_close(
            reward, torch.tensor([1.0, 0.0, 0.0, 0.0])
        )

        robot.feet_pos[0, :, 2] += 0.10
        shifted_reward = robot._reward_feet_landing_pose()
        self.assertGreater(shifted_reward[0].item(), 0.0)
        self.assertLess(shifted_reward[0].item(), reward[0].item())

    def test_episode_quotas_are_balanced_and_exact(self):
        quotas = build_episode_quotas(4, 10)
        self.assertEqual(quotas.tolist(), [3, 3, 2, 2])
        self.assertEqual(int(quotas.sum()), 10)
        self.assertLessEqual(int(quotas.max() - quotas.min()), 1)

    def test_name_permutation_round_trip(self):
        simulator_order = ("FR", "FL", "RR", "RL")
        policy_order = ("FL", "FR", "RL", "RR")
        to_policy = build_name_permutation(
            simulator_order, policy_order, "test joint"
        )
        to_simulator = build_name_permutation(
            policy_order, simulator_order, "test joint"
        )
        values = [10, 20, 30, 40]
        policy_values = [values[index] for index in to_policy]
        round_trip = [policy_values[index] for index in to_simulator]
        self.assertEqual(round_trip, values)

    def test_urdf_imu_mount_parser(self):
        xml_text = """
        <robot name="test">
          <link name="base"/>
          <link name="imu"/>
          <joint name="imu_joint" type="fixed">
            <origin xyz="1 2 3" rpy="0.1 0.2 0.3"/>
            <parent link="base"/>
            <child link="imu"/>
          </joint>
        </robot>
        """
        parent, xyz, rpy = urdf_fixed_joint_mount(xml_text, "imu")
        self.assertEqual(parent, "base")
        self.assertEqual(xyz, (1.0, 2.0, 3.0))
        self.assertEqual(rpy, (0.1, 0.2, 0.3))

    def test_summary(self):
        metrics = summarize_episodes([0.4, 0.6], [float("nan"), 0.1], 0.5)
        self.assertAlmostEqual(metrics["height_mean"], 0.5)
        self.assertAlmostEqual(metrics["success_rate"], 0.5)
        self.assertAlmostEqual(metrics["successful_landing_error_mean"], 0.1)

    def test_same_level_gate_passes_at_boundaries(self):
        reference = result("go2_upwards", 0.6, 0.62, 0.9)
        candidate = result("solo12_v3_1_upwards", 0.57, 0.589, 0.85)
        gate = compare_evaluations(candidate, reference)
        self.assertTrue(gate["pass"])

    def test_gate_rejects_lower_mean_even_if_success_passes(self):
        reference = result("go2_upwards", 0.6, 0.62, 0.9)
        candidate = result("solo12_v3_1_upwards", 0.56, 0.62, 0.9)
        gate = compare_evaluations(candidate, reference)
        self.assertFalse(gate["pass"])

    def test_protocol_mismatch_is_not_comparable(self):
        reference = result("go2_upwards", 0.6, 0.62, 0.9)
        candidate = result("solo12_v3_1_upwards", 0.6, 0.62, 0.9)
        candidate["seed"] = 2
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            compare_evaluations(candidate, reference)


if __name__ == "__main__":
    unittest.main()
