#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)

if [[ -n ${TRAINS_WORKSPACE_ROOT:-} ]]; then
  workspace=$TRAINS_WORKSPACE_ROOT
else
  workspace=$(cd -- "$project_root/../.." && pwd)
fi

if [[ ! -d "$workspace/trains" || ! -d "$workspace/projects/quadruped-jumping" ]]; then
  echo "Unable to locate robotcontrol-trains; set TRAINS_WORKSPACE_ROOT." >&2
  return 2
fi
