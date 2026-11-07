#!/bin/bash
# Resumes the fresh-seed elliptic measurement after the Nov 6 interruptions. Same runner and
# arguments as _fresh_elliptic_nov6.sh; each run continues from its own file. Rules: results.md,
# "Pre-registered, measurement only: noslip pass on fresh seeds", its elliptic addendum, and
# "Amendment before any fresh-seed elliptic row exists".
# Changes from the Nov 6 script, none affecting results:
# - A STOP file in this folder halts the queue before its next attempt or run, so a pause holds.
#   (On Nov 6 the retry loop restarted a run two minutes after it was stopped.)
# - A 10-minute cool-down between runs, because the laptop has frozen under sustained load.
# - The interrupted noslip runs are left out. They wait for a separate decision.
OUT="/c/Users/oskic/PI-TCN-SlipNet/Data/diag_fresh_seeds"
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"
PY=../../venv/Scripts/python.exe
SCENE=(
  --mjcf ../../third_party/orcahand_description/v2/scene_slip.xml
  --fingertips right_T-DP_b7429e50 right_I-FingerTipAssembly_ec49c16c
               right_M-FingerTipAssembly_34afb748 right_M-FingerTipAssembly_424a8e75
               right_P-FingerTipAssembly_cd219176
  --actuators right_t-mcp_actuator right_t-pip_actuator
              right_i-mcp_actuator right_i-pip_actuator
              right_m-mcp_actuator right_m-pip_actuator
              right_r-mcp_actuator right_r-pip_actuator
              right_p-mcp_actuator right_p-pip_actuator
  --mesh-dir ../../assets --out-dir ../../Data/diag_contact_physics/scenes
)
TCN=(--checkpoint ../../src/training/model_a.pt --in-channels 22)

stop_if_asked() {
  if [ -f "$OUT/STOP" ]; then
    echo "$(date): STOP file found, queue stopped"
    exit 0
  fi
}

cool_down() {
  echo "$(date): cooling down for 10 minutes"
  sleep 600
  stop_if_asked
}

run() {  # tag variant, then controller arguments. Same arguments as the Nov 4 and Nov 6 scripts.
  local tag="$1" variant="$2" attempt ec=1
  shift 2
  for attempt in 1 2 3 4 5 6; do
    stop_if_asked
    "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" "${TCN[@]}" "$@" --variant "$variant" \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && return 0
    stop_if_asked
    sleep 120
  done
  return 1
}

grip() {  # the fixed grip, no policy, tactile input or TCN
  local tag="grip_TIRP_fast_elliptic_impratio10" attempt ec=1
  for attempt in 1 2 3 4 5 6; do
    stop_if_asked
    "$PY" diagnostic_scripted_grip.py "${SCENE[@]}" --variant elliptic_impratio10 --grips TIRP_fast \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && return 0
    stop_if_asked
    sleep 120
  done
  return 1
}

run handA_elliptic_impratio10 elliptic_impratio10 --controller hand
cool_down
run seed2_elliptic_impratio10 elliptic_impratio10 --controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip
cool_down
run seed3_elliptic_impratio10 elliptic_impratio10 --controller policy --policy ../../Data/ppo_slip_policy_seed3.zip
echo FRESH_ELLIPTIC_DONE
cool_down
grip
echo FRESH_GRIP_DONE
