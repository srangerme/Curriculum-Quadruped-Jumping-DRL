#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"
image=localhost:8080/trains/projects/quadruped-jumping:ada-py38-cu118-v1
container_name=nohard-v4-forward-f1-corrected-costoff-c15850-100
parent_run=Sep15_01-55-29_nohard_v4_fromzero_17200chain_low065_filter6speed_arm002_seed33_c15750_to16050
run_name=nohard_v4_forward_f1_corrected_costoff_from_upward_c15850_100

docker run -d --name "$container_name" --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/trains/no_wandb:/workspace/project/legged_gym:/workspace/project/rsl_rl \
  -e LD_LIBRARY_PATH=/opt/conda/lib -e MPLBACKEND=Agg \
  -v "$workspace/projects/quadruped-jumping:/workspace/project" \
  -v "$workspace/outputs/quadruped-jumping-ada/logs:/workspace/project/legged_gym/logs" \
  -v "$workspace/models:/workspace/models:ro" \
  -v "$workspace/artifacts:/workspace/project/artifacts" \
  -v "$workspace/cache/torch:/root/.cache/torch" \
  -v "$workspace/cache/pip:/root/.cache/pip" \
  -w /workspace/project/legged_gym/legged_gym/scripts \
  "$image" \
  python train.py --headless --task solo12_v3_1_forward \
  --num_envs 4096 --seed 33 \
  --resume --load_run "$parent_run" --checkpoint 15850 \
  --max_iterations 100 --save_interval_override 25 \
  --base_height_override .32 \
  --initial_stance_height_range_override .28,.34 \
  --initial_stance_height_probability_override 1 \
  --initial_contact_history_probability_override 1 \
  --initial_contact_settle_steps_override 1 \
  --initial_contact_base_height_offset_override 0 \
  --contact_observation_delay_max_steps_override 2 \
  --latency_range_min_ms 0 --latency_range_max_ms 40 \
  --zero_latency_probability_override .35 --max_latency_probability_override .30 \
  --restitution_range_override 0,.2 \
  --joint_friction_range_override 0,.01 \
  --joint_damping_range_override 0,.01 \
  --friction_range_override .5,1.25 \
  --joint_armature_range_override .002,.002 \
  --max_contact_force_override 120 \
  --post_landing_dof_vel_scale_override -.5 \
  --reset_landing_error_override .35 \
  --command_position_observation_scale_override 25 \
  --command_pos_dx_min_override 0 --command_pos_dx_max_override .30 \
  --task_pos_scale_override 1500 \
  --forward_jump_quality_scale_override 0 \
  --forward_takeoff_height_floor_scale_override 0 \
  --forward_takeoff_joint_scale_override 0 \
  --forward_takeoff_position_scale_override 500 \
  --forward_takeoff_position_coarse_scale_override 0 \
  --takeoff_front_rear_timing_scale_override 0 \
  --task_max_height_scale_override 1000 \
  --base_height_flight_scale_override 20 \
  --push_towards_goal_probability_override 0 \
  --push_towards_goal_final_probability_override 0 \
  --filter_freq_override 6 --clip_actions_override 100 \
  --physical_dof_velocity_limit_override 200 \
  --velocity_torque_envelope 1 --velocity_torque_limit_scale 1 \
  --dof_vel_limits_scale_override 0 \
  --pre_takeoff_dof_vel_envelope_scale_override 0 \
  --velocity_torque_rejection_scale_override 0 \
  --velocity_cost_enabled 0 \
  --run_name "$run_name" \
  --effective_config_output /workspace/project/artifacts/velocity_recovery_20260913/nohard_v4_forward_f1_corrected_costoff_c15850_100_effective.json
