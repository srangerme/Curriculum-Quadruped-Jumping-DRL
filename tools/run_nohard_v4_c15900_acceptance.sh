#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"

phase=${1:-main}
run=Sep15_01-55-29_nohard_v4_fromzero_17200chain_low065_filter6speed_arm002_seed33_c15750_to16050
checkpoint=${2:-15900}
host_out="$workspace/artifacts/velocity_recovery_20260913/c${checkpoint}_formal_acceptance_v2"
container_out=/workspace/project/artifacts/velocity_recovery_20260913/c${checkpoint}_formal_acceptance_v2
mkdir -p "$host_out/logs"

seeds=(24680 24681 24682)
histories=(0 1)
common=(
  --friction_range_override .5,1.25
  --restitution_range_override 0,.2
  --joint_friction_range_override 0,.02
  --latency_range_max_ms 40
  --contact_observation_delay_max_steps 2
  --fixed_joint_armature .002
  --filter_freq_override 6
  --velocity_torque_envelope 1
  --velocity_torque_limit_scale 1
  --physical_dof_velocity_limit_override 200
  --landing_stability_seconds 1.0
)

run_eval() {
  local name=$1
  local history=$2
  local episodes=$3
  local seed=$4
  shift 4
  local output="$container_out/${name}.json"
  local log="$host_out/logs/${name}.log"
  if [[ -s "$host_out/${name}.json" ]]; then
    echo "SKIP $name"
    return
  fi
  echo "START $name $(date --iso-8601=seconds)"
  bash "$script_dir/run_nohard_upward_eval.sh" \
    "$run" "$checkpoint" "$history" robust "$episodes" "$output" \
    --seed "$seed" "${common[@]}" "$@" >"$log" 2>&1
  echo "DONE  $name $(date --iso-8601=seconds)"
}

case "$phase" in
  main)
    for bucket in A B Full; do
      bucket_args=()
      if [[ "$bucket" == A ]]; then
        bucket_args=(--physx_contact_offset_override .010 --physx_rest_offset_override 0 --physx_friction_offset_threshold_override .010)
      elif [[ "$bucket" == B ]]; then
        bucket_args=(--physx_contact_offset_override .005 --physx_rest_offset_override 0 --physx_friction_offset_threshold_override .005)
      fi
      for history in "${histories[@]}"; do
        for seed in "${seeds[@]}"; do
          run_eval "main_${bucket}_p${history}_s${seed}_128" "$history" 128 "$seed" "${bucket_args[@]}"
        done
      done
    done
    ;;
  endpoints)
    for latency in 0 40; do
      for restitution in 0 .2; do
        rest_name=${restitution//./p}
        for history in "${histories[@]}"; do
          for seed in "${seeds[@]}"; do
            run_eval "endpoint_lat${latency}_rest${rest_name}_jf002_p${history}_s${seed}_128" \
              "$history" 128 "$seed" --fixed_latency_ms "$latency" \
              --fixed_restitution "$restitution" --fixed_joint_friction .02
          done
        done
      done
    done
    ;;
  cap_pair)
    for cap in cap2 cap200; do
      cap_args=(--physical_dof_velocity_limit_scale 2)
      if [[ "$cap" == cap200 ]]; then
        cap_args=(--physical_dof_velocity_limit_override 200)
      fi
      for history in "${histories[@]}"; do
        for seed in "${seeds[@]}"; do
          run_eval "pair_${cap}_Full_p${history}_s${seed}_128" "$history" 128 "$seed" "${cap_args[@]}"
        done
      done
    done
    ;;
  heights)
    for height in .28 .29 .30 .31 .32 .33 .34; do
      height_name=${height//./p}
      margin=0
      if [[ "$height" == .33 ]]; then
        margin=.02
      elif [[ "$height" == .34 ]]; then
        margin=.02
      fi
      for seed in "${seeds[@]}"; do
        run_eval "height_${height_name}_Full_p1_s${seed}_96" 1 96 "$seed" \
          --initial_stance_height_probability_override 1 \
          --initial_stance_height_range_override "$height,$height" \
          --initial_stance_kinematic_margin_override "$margin" \
          --initial_contact_settle_steps 0
      done
    done
    ;;
  screen)
    for candidate in 15800 15850 15950 16000 16050; do
      checkpoint=$candidate
      for history in "${histories[@]}"; do
        run_eval "screen_c${candidate}_Full_p${history}_s24680_128" "$history" 128 24680
      done
    done
    ;;
  screen_height34)
    for candidate in 15850 15900 15950 16000 16050; do
      checkpoint=$candidate
      run_eval "screen_height_p34_c${candidate}_Full_p1_s24680_96" 1 96 24680 \
        --base_height_override .34
    done
    ;;
  contact_height34)
    checkpoint=${2:-15800}
    run_eval "contact_height_p34_c${checkpoint}_Full_p1_s24680_96" 1 96 24680 \
      --initial_stance_height_probability_override 1 \
      --initial_stance_height_range_override .34,.34 \
      --initial_contact_settle_steps 8
    ;;
  contact_height34_direct)
    checkpoint=${2:-15800}
    run_eval "contact_height_p34_direct_c${checkpoint}_Full_p1_s24680_96" 1 96 24680 \
      --initial_stance_height_probability_override 1 \
      --initial_stance_height_range_override .34,.34 \
      --initial_contact_settle_steps 0
    ;;
  contact_height34_margin_scan)
    checkpoint=${2:-15800}
    for margin in .002 .004 .006 .008; do
      margin_name=${margin//./p}
      run_eval "contact_height_p34_margin${margin_name}_c${checkpoint}_p1_s24680_32" \
        1 32 24680 \
        --initial_stance_height_probability_override 1 \
        --initial_stance_height_range_override .34,.34 \
        --initial_stance_kinematic_margin_override "$margin" \
        --initial_contact_settle_steps 0
    done
    ;;
  contact_height34_margin_scan2)
    checkpoint=${2:-15800}
    for margin in .012 .020 .030 .040; do
      margin_name=${margin//./p}
      run_eval "contact_height_p34_margin${margin_name}_c${checkpoint}_p1_s24680_32" \
        1 32 24680 \
        --initial_stance_height_probability_override 1 \
        --initial_stance_height_range_override .34,.34 \
        --initial_stance_kinematic_margin_override "$margin" \
        --initial_contact_settle_steps 0
    done
    ;;
  screen_height_bounds)
    for candidate in 15850 15900 15950 16000 16050; do
      checkpoint=$candidate
      run_eval "screen_height_p28_c${candidate}_p1_s24680_96" 1 96 24680 \
        --initial_stance_height_probability_override 1 \
        --initial_stance_height_range_override .28,.28 \
        --initial_stance_kinematic_margin_override 0 \
        --initial_contact_settle_steps 0
      run_eval "screen_height_p34_contact_c${candidate}_p1_s24680_96" 1 96 24680 \
        --initial_stance_height_probability_override 1 \
        --initial_stance_height_range_override .34,.34 \
        --initial_stance_kinematic_margin_override .02 \
        --initial_contact_settle_steps 0
    done
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
