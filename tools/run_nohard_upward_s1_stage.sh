#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 8 ]]; then
  echo "usage: $0 CONTAINER SEED PARENT_RUN PARENT_CHECKPOINT ITERATIONS RUN_NAME SIGMA EFFECTIVE_JSON [EXTRA_ARGS...]" >&2
  exit 2
fi

container_name=$1
seed=$2
parent_run=$3
parent_checkpoint=$4
iterations=$5
run_name=$6
height_sigma=$7
effective_json=$8
shift 8

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"
image=localhost:8080/trains/projects/quadruped-jumping:ada-py38-cu118-v1

docker_args=(
  run -d --name "$container_name" --gpus all
  -e PYTHONDONTWRITEBYTECODE=1
  -e PYTHONUNBUFFERED=1
  -e PYTHONPATH=/opt/trains/no_wandb:/workspace/project/legged_gym:/workspace/project/rsl_rl
  -e LD_LIBRARY_PATH=/opt/conda/lib
  -e MPLBACKEND=Agg
  -v "$workspace/projects/quadruped-jumping:/workspace/project"
  -v "$workspace/outputs/quadruped-jumping-ada/logs:/workspace/project/legged_gym/logs"
  -v "$workspace/models:/workspace/models:ro"
  -v "$workspace/artifacts:/workspace/project/artifacts"
  -v "$workspace/cache/torch:/root/.cache/torch"
  -v "$workspace/cache/pip:/root/.cache/pip"
  -w /workspace/project/legged_gym/legged_gym/scripts
  "$image"
)

train_args=(
  python train.py --headless --task solo12_v3_1_upwards
  --num_envs 4096 --seed "$seed"
  --max_iterations "$iterations"
  --fixed_target_x 0 --fixed_target_y 0
  --base_height_override .32
  --settled_contact_count_override 4
  --initial_contact_history_probability_override 0
  --initial_contact_settle_steps_override 1
  --initial_contact_base_height_offset_override 0
  --contact_observation_delay_max_steps_override 0
  --latency_range_min_ms 0 --latency_range_max_ms 20
  --restitution_range_override 0,.4
  --joint_friction_range_override 0,.02
  --friction_range_override .5,1.25
  --joint_damping_range_override 0,.01
  --filter_freq_override 8 --clip_actions_override 100
  --takeoff_vz_pitch_quality_scale_override 1000
  --takeoff_pitch_angular_impulse_scale_override -200
  --takeoff_pitch_angular_impulse_abs_scale_override -100
  --takeoff_predicted_height_sigma_override "$height_sigma"
  --dof_vel_limits_scale_override 0
  --velocity_torque_rejection_scale_override 0
  --velocity_torque_envelope 0
  --run_name "$run_name"
  --effective_config_output "/workspace/project/artifacts/velocity_recovery_20260913/$effective_json"
)

if [[ $parent_run == - ]]; then
  if [[ $parent_checkpoint != 0 ]]; then
    echo "from-zero stages require PARENT_CHECKPOINT=0" >&2
    exit 2
  fi
else
  train_args+=(
    --resume --load_run "$parent_run" --checkpoint "$parent_checkpoint"
  )
fi

docker "${docker_args[@]}" "${train_args[@]}" "$@"
