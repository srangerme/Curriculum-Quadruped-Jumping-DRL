#!/usr/bin/env bash
set -euo pipefail

task=$1
run=$2
checkpoint=$3

run_case() {
    local label=$1
    local seed=$2
    local mode=$3
    shift 3
    local random_args=()
    local max_pos=0.15
    if [[ "$mode" == "random" ]]; then
        random_args+=(--eval_with_randomization)
        max_pos=0.25
    fi

    python3 -m trains.cli play quadruped-jumping-ada \
        --task "$task" \
        --load_run "$run" \
        --checkpoint "$checkpoint" \
        --headless \
        --seed "$seed" \
        --landing_stability_seconds 3 \
        --landing_contact_grace_seconds 0.25 \
        --landing_min_all_feet_contact_ratio 0.95 \
        --landing_max_leg_torque_cv 0.25 \
        --landing_min_success_rate 0.95 \
        --enforce_landing_stability \
        "${random_args[@]}" \
        "$@" 2>&1 | awk \
        -v label="$label" -v seed="$seed" -v mode="$mode" -v max_pos="$max_pos" '
            {buffer = buffer $0 "\n"}
            / - jump_landing_error:/ {pos=$3}
            / - jump_landing_yaw_error:/ {yaw=$3}
            / - jump_flight_time:/ {flight=$3}
            / - jump_stable_standing_success:/ {stable=$3}
            / - jump_post_landing_all_feet_contact_ratio:/ {contact=$3}
            / - jump_post_landing_final_all_feet_contact:/ {final_contact=$3}
            / - jump_post_landing_leg_torque_cv:/ {torque_cv=$3}
            / - jump_post_landing_leg_torque_fl:/ {torque_fl=$3}
            / - jump_post_landing_leg_torque_fr:/ {torque_fr=$3}
            / - jump_post_landing_leg_torque_rl:/ {torque_rl=$3}
            / - jump_post_landing_leg_torque_rr:/ {torque_rr=$3}
            / - jump_post_landing_natural_failure:/ {failure=$3}
            END {
                if (pos == "" || yaw == "" || flight == "" || stable == "" ||
                    contact == "" || final_contact == "" || torque_cv == "" ||
                    failure == "") {
                    printf "%s", buffer > "/dev/stderr"
                    exit 2
                }
                pass = (pos <= max_pos && yaw <= 0.45 && flight >= 0.30 &&
                        stable >= 0.95 && contact >= 0.95 &&
                        final_contact >= 0.95 && torque_cv <= 0.25 && failure == 0)
                printf "%s seed=%s mode=%s pos=%s yaw=%s flight=%s stable=%s four_feet=%s final_four_feet=%s torque=[FL:%s,FR:%s,RL:%s,RR:%s] torque_cv=%s natural_failure=%s result=%s\n",
                    label, seed, mode, pos, yaw, flight, stable, contact,
                    final_contact, torque_fl, torque_fr, torque_rl, torque_rr,
                    torque_cv, failure, pass ? "PASS" : "FAIL"
                if (!pass) exit 2
            }
        '
}

grid() {
    local mode=$1
    local seed=$2
    run_case x0 "$seed" "$mode" --jump_distance 0.0
    run_case x025 "$seed" "$mode" --jump_distance 0.25
    run_case x05 "$seed" "$mode" --jump_distance 0.5
    run_case x1 "$seed" "$mode" --jump_distance 1.0
    run_case yneg "$seed" "$mode" --jump_distance_y -0.5
    run_case ypos "$seed" "$mode" --jump_distance_y 0.5
    run_case yawneg "$seed" "$mode" --jump_yaw_deg -90
    run_case yawpos "$seed" "$mode" --jump_yaw_deg 90
    run_case comboneg "$seed" "$mode" --jump_distance 0.5 --jump_distance_y -0.5 --jump_yaw_deg -90
    run_case combopos "$seed" "$mode" --jump_distance 0.5 --jump_distance_y 0.5 --jump_yaw_deg 90
}

grid nominal 1
grid random 1
for seed in 2 3; do
    run_case x1 "$seed" random --jump_distance 1.0
    run_case ypos "$seed" random --jump_distance_y 0.5
    run_case yawpos "$seed" random --jump_yaw_deg 90
    run_case combopos "$seed" random \
        --jump_distance 0.5 --jump_distance_y 0.5 --jump_yaw_deg 90
done
