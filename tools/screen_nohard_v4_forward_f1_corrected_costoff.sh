#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"

run=Sep15_09-40-59_nohard_v4_forward_f1_corrected_costoff_from_upward_c15850_100
host_out="$workspace/artifacts/velocity_recovery_20260913/f1_corrected_costoff_screen"
container_out=/workspace/project/artifacts/velocity_recovery_20260913/f1_corrected_costoff_screen
mkdir -p "$host_out/logs"

for checkpoint in 15875 15900; do
  for target in .1 .2 .3; do
    target_name=${target//./p}
    name=c${checkpoint}_x${target_name}_nominal_32
    "$script_dir/run_nohard_forward_eval.sh" \
      "$run" "$checkpoint" "$target" nominal 32 \
      "$container_out/$name.json" >"$host_out/logs/$name.log" 2>&1
  done
done
