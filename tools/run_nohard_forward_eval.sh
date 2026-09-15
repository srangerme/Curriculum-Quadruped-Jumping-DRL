#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 RUN CHECKPOINT TARGET_X MODE EPISODES OUTPUT_JSON [EXTRA_ARGS...]" >&2
  exit 2
fi

run_name=$1
checkpoint=$2
target_x=$3
mode=$4
episodes=$5
output_json=$6
shift 6

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"
image=localhost:8080/trains/projects/quadruped-jumping:ada-py38-cu118-v1

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/trains/no_wandb:/workspace/project/legged_gym:/workspace/project/rsl_rl \
  -e LD_LIBRARY_PATH=/opt/conda/lib -e MPLBACKEND=Agg \
  -v "$workspace/projects/quadruped-jumping:/workspace/project" \
  -v "$workspace/outputs/quadruped-jumping-ada/logs:/workspace/project/legged_gym/logs" \
  -v "$workspace/models:/workspace/models:ro" \
  -v "$workspace/artifacts:/workspace/project/artifacts" \
  -v "$workspace/cache/torch:/root/.cache/torch" \
  -w /workspace/project/legged_gym/legged_gym/scripts \
  "$image" \
  python evaluate_checkpoint.py --headless --task solo12_v3_1_forward \
  --load_run "$run_name" --checkpoint "$checkpoint" \
  --num_envs "$episodes" --eval_episodes "$episodes" --seed 48721 \
  --episode_sampling balanced-per-environment --eval_max_steps 400 \
  --target_x "$target_x" --target_y 0 \
  --eval_mode "$mode" \
  --observation_noise_profile none \
  --filter_freq_override 6 --clip_actions_override 100 \
  --base_height_override .32 \
  --initial_contact_settle_steps 1 \
  --initial_contact_base_height_offset 0 \
  --initial_contact_history_probability 1 \
  --contact_observation_delay_max_steps 0 \
  --reset_landing_error_override .35 \
  --command_position_observation_scale_override 25 \
  --forward_position_tolerance .05 \
  --fixed_joint_armature .002 \
  --physical_dof_velocity_limit_override 200 \
  --velocity_torque_envelope 1 \
  --velocity_torque_limit_scale 1 \
  --output_json "$output_json" --diagnostics "$@"
