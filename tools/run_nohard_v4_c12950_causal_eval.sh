#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"

run=Sep14_18-56-36_nohard_v4_fromzero_17200chain_final_restendpointmix_p050_seed33_c12800_to13100
checkpoint=12950
artifact_dir=/workspace/project/artifacts/velocity_recovery_20260913

for cap in cap2 cap200; do
  for history_probability in 0 1; do
    for seed in 24680 24681 24682; do
      output="$artifact_dir/nohard_v4_c12950_final_causal_${cap}_Full_p${history_probability}_s${seed}_128.json"
      extra_cap_args=(--physical_dof_velocity_limit_scale 2)
      if [[ "$cap" == cap200 ]]; then
        extra_cap_args=(--physical_dof_velocity_limit_override 200)
      fi
      bash "$script_dir/run_nohard_upward_eval.sh" \
        "$run" "$checkpoint" "$history_probability" robust 128 "$output" \
        --seed "$seed" \
        --friction_range_override .5,1.25 \
        --restitution_range_override 0,.2 \
        --joint_friction_range_override 0,.02 \
        --latency_range_max_ms 40 \
        --contact_observation_delay_max_steps 2 \
        "${extra_cap_args[@]}" >/tmp/nohard_v4_c12950_causal_eval.log
    done
  done
done
