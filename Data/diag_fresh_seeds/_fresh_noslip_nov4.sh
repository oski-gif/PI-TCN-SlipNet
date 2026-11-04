#!/bin/bash
# Read-only measurement. Rules: results.md, "Pre-registered, measurement only: noslip pass on
# fresh seeds". Four single-process runs side by side, about the same load as RL training's
# 4 workers. Each run resumes from its own file, so a failed attempt loses at most one trial.
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

run() {  # tag, then controller arguments
  local tag="$1" attempt ec=1
  shift
  for attempt in 1 2 3 4 5 6; do
    "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" "$@" --variant noslip20 \
      --n-trials-per-object 67 --seed-start 2000 --out-json "$OUT/$tag.jsonl" >> "$OUT/$tag.log" 2>&1
    ec=$?
    echo "$(date): $tag attempt $attempt exited $ec"
    [ "$ec" -eq 0 ] && break
    sleep 120
  done
}

run seed1_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip &
run seed2_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip &
run seed3_noslip20 --controller policy --policy ../../Data/ppo_slip_policy_seed3.zip &
run handA_noslip20 --controller hand &
wait
echo FRESH_DONE
