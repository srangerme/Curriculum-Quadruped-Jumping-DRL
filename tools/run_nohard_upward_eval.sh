#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 RUN CHECKPOINT CONTACT_HISTORY_PROB MODE EPISODES OUTPUT_JSON [EXTRA_ARGS...]" >&2
  exit 2
fi

run_name=$1
checkpoint=$2
contact_history_probability=$3
eval_mode=$4
episodes=$5
output_json=$6
shift 6

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"
image=localhost:8080/trains/projects/quadruped-jumping:ada-py38-cu118-v1

docker run --rm --gpus all \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/trains/no_wandb:/workspace/project/legged_gym:/workspace/project/rsl_rl \
  -e LD_LIBRARY_PATH=/opt/conda/lib \
  -e MPLBACKEND=Agg \
  -v "$workspace/projects/quadruped-jumping:/workspace/project" \
  -v "$workspace/outputs/quadruped-jumping-ada/logs:/workspace/project/legged_gym/logs" \
  -v "$workspace/models:/workspace/models:ro" \
  -v "$workspace/artifacts:/workspace/project/artifacts" \
  -v "$workspace/cache/torch:/root/.cache/torch" \
  -w /workspace/project/legged_gym/legged_gym/scripts \
  "$image" \
  python evaluate_checkpoint.py \
  --headless --task solo12_v3_1_upwards \
  --num_envs "$episodes" --seed 24680 \
  --load_run "$run_name" --checkpoint "$checkpoint" \
  --eval_mode "$eval_mode" --eval_episodes "$episodes" \
  --target_x 0 --target_y 0 \
  --base_height_override .32 --settled_contact_count 4 \
  --initial_contact_history_probability "$contact_history_probability" \
  --initial_contact_settle_steps 1 --initial_contact_base_height_offset 0 \
  --contact_observation_delay_max_steps 0 \
  --filter_freq_override 8 --clip_actions_override 100 \
  --velocity_torque_envelope 0 --action_noise_scale 0 \
  --diagnostics --output_json "$output_json" "$@"
