#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/workspace_env.sh"

run=Sep15_10-14-46_nohard_v4_forward_f1_legacychain_aligned_costoff_from_c15850_300
host_out="$workspace/artifacts/velocity_recovery_20260913/f1_legacychain_aligned_screen"
container_out=/workspace/project/artifacts/velocity_recovery_20260913/f1_legacychain_aligned_screen
mkdir -p "$host_out/logs"

for checkpoint in 15900 16150; do
  for target in .1 .2 .3; do
    target_name=${target//./p}
    name=c${checkpoint}_x${target_name}_nominal_32
    "$script_dir/run_nohard_forward_eval.sh" \
      "$run" "$checkpoint" "$target" nominal 32 \
      "$container_out/$name.json" >"$host_out/logs/$name.log" 2>&1
  done
done
