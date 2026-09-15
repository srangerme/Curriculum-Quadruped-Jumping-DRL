#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"

run=Sep15_10-06-21_nohard_v4_forward_f1_heightgate_joint500_taskheight5000_costoff_from_c15850_100
checkpoint=15875
host_out="$workspace/artifacts/velocity_recovery_20260913/f1_heightgate_taskheight5000_screen"
container_out=/workspace/project/artifacts/velocity_recovery_20260913/f1_heightgate_taskheight5000_screen
mkdir -p "$host_out/logs"

for target in .1 .2 .3; do
  target_name=${target//./p}
  name=c${checkpoint}_x${target_name}_nominal_32
  "$script_dir/run_nohard_forward_eval.sh" \
    "$run" "$checkpoint" "$target" nominal 32 \
    "$container_out/$name.json" >"$host_out/logs/$name.log" 2>&1
done
