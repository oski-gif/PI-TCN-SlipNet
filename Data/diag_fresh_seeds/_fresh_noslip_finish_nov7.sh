#!/bin/bash
# Finishes the three fresh-seed noslip runs the Nov 4 crash interrupted (user's decision, Nov 7).
# Waits until _fresh_resume_nov7.sh has finished the fixed grip, so only one job runs at a time.
# Same runner and arguments as _fresh_noslip_nov4.sh; each run continues from its own file.
# A STOP file in this folder halts it while waiting, between attempts, and between runs.
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
  --checkpoint ../../src/training/model_a.pt --in-channels 22
)

stop_if_asked() {
  if [ -f "$OUT/STOP" ]; then
    echo "$(date): STOP file found, noslip queue stopped"
    exit 0
  fi
}

cool_down() {
  echo "$(date): cooling down for 10 minutes"
  sleep 600
  stop_if_asked
}

run() {  # tag, then controller arguments. Same arguments as _fresh_noslip_nov4.sh.
  local tag="$1" attempt ec=1
  shift
  for attempt in 1 2 3 4 5 6; do
    stop_if_asked
    "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" "$@" --variant noslip20 \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && return 0
    stop_if_asked
    sleep 120
  done
  return 1
}

until grep -q FRESH_GRIP_DONE "$OUT/_fresh_resume_nov7.out" 2>/dev/null; do
  stop_if_asked
  sleep 60
done
echo "$(date): fixed grip finished, starting the noslip runs"
cool_down
run seed1_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip
cool_down
run seed2_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip
cool_down
run seed3_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seed3.zip
echo FRESH_NOSLIP_DONE
