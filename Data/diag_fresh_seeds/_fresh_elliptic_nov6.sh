#!/bin/bash
# Read-only measurement, one job at a time after the Nov 4 crash. Rules: results.md,
# "Pre-registered, measurement only: noslip pass on fresh seeds", its elliptic addendum, and
# "Amendment before any fresh-seed elliptic row exists". Every run resumes from its own file,
# so rerunning this script after an interruption continues where it stopped.
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

run() {  # tag variant, then controller arguments. Same arguments as the Nov 4 scripts.
  local tag="$1" variant="$2" attempt ec=1
  shift 2
  for attempt in 1 2 3 4 5 6; do
    "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" "${TCN[@]}" "$@" --variant "$variant" \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && return 0
    sleep 120
  done
  return 1
}

grip() {  # the fixed grip, no policy, tactile input or TCN
  local tag="grip_TIRP_fast_elliptic_impratio10" attempt ec=1
  for attempt in 1 2 3 4 5 6; do
    "$PY" diagnostic_scripted_grip.py "${SCENE[@]}" --variant elliptic_impratio10 --grips TIRP_fast \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && return 0
    sleep 120
  done
  return 1
}

run seed1_elliptic_impratio10 elliptic_impratio10 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip
run handA_elliptic_impratio10 elliptic_impratio10 --controller hand
run seed2_elliptic_impratio10 elliptic_impratio10 --controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip
run seed3_elliptic_impratio10 elliptic_impratio10 --controller policy --policy ../../Data/ppo_slip_policy_seed3.zip
echo FRESH_ELLIPTIC_DONE
grip
echo FRESH_GRIP_DONE
# Lowest priority: finish the noslip runs the crash interrupted, same arguments as before.
run seed1_noslip20 noslip20 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip
run seed2_noslip20 noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip
run seed3_noslip20 noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seed3.zip
echo FRESH_NOSLIP_FINISHED
echo ALL_DONE
