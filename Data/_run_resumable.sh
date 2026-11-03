#!/bin/bash
# Runs a resumable evaluation script until complete, relaunching a fresh process
# after a crash or failure streak. Progress lives in --out-json, so a relaunch
# resumes it. Stops after 3 relaunches in a row that add no completed trial.
# Usage: _run_resumable.sh <out.jsonl> <log> <script.py> <extra args...>
# Paths are relative to the Simulation directory. Pass --checkpoint and
# --in-channels explicitly.
OUT_JSON="$1"; LOG="$2"; SCRIPT="$3"; shift 3
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"

COMMON_ARGS=(
  --mjcf ../../third_party/orcahand_description/v2/scene_slip.xml
  --fingertips right_T-DP_b7429e50 right_I-FingerTipAssembly_ec49c16c
               right_M-FingerTipAssembly_34afb748 right_M-FingerTipAssembly_424a8e75
               right_P-FingerTipAssembly_cd219176
  --actuators right_t-mcp_actuator right_t-pip_actuator
              right_i-mcp_actuator right_i-pip_actuator
              right_m-mcp_actuator right_m-pip_actuator
              right_r-mcp_actuator right_r-pip_actuator
              right_p-mcp_actuator right_p-pip_actuator
  --mesh-dir ../../assets --out-dir ../../Data/closed_loop_scenes
)

count_done() { if [ -f "$OUT_JSON" ]; then grep -vc '"error":' "$OUT_JSON"; else echo 0; fi; }

no_progress=0
while true; do
  before=$(count_done)
  python "$SCRIPT" "${COMMON_ARGS[@]}" "$@" --out-json "$OUT_JSON" >> "$LOG" 2>&1
  EC=$?
  after=$(count_done)
  if [ "$EC" -eq 0 ]; then
    echo "RUN_EXIT=0 completed_rows=$after" >> "$LOG"
    exit 0
  fi
  echo "=== process exited $EC, completed rows $before -> $after ===" >> "$LOG"
  if [ "$after" -le "$before" ]; then no_progress=$(( no_progress + 1 )); else no_progress=0; fi
  if [ "$no_progress" -ge 3 ]; then
    echo "RUN_EXIT=$EC STOPPED: 3 relaunches with no progress" >> "$LOG"
    exit 1
  fi
done
