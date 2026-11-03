#!/bin/bash
# Orchestrates the remaining RL training as a sequence of safe(r) segments,
# each a fresh process resuming from the previous segment's final policy.
# Segment length is risk-REDUCTION, not a guaranteed-safe threshold (see
# results.md's "solo retry -- a genuine SIGSEGV" entry, 2026-10-30) --
# ~40-50k steps per segment per the user's explicit decision, after
# EvalCallback (implicated in two SIGSEGV crashes) was removed from
# train_rl_policy.py entirely. Stops immediately on any non-zero exit
# rather than continuing past a failure.
# (No `set -e` -- the exit-code check below needs to run and print its own
# message even on failure, which `set -e` would short-circuit past.)
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"

SEGMENTS=(175000 215000 255000 300000)
RESUME_FROM="../../Data/ppo_slip_policy_latest.zip"
LOGDIR="../../Data/rl_logs_production"

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
  --checkpoint ../../src/training/model_a.pt --in-channels 22
  --n-envs 4
)

SEG_IDX=0
for TARGET in "${SEGMENTS[@]}"; do
  SEG_IDX=$(( SEG_IDX + 1 ))
  echo "=== SEGMENT ${SEG_IDX}: resuming toward total_timesteps=${TARGET} ==="

  # Fresh per-segment reset-count tracking.
  rm -f ../../Data/_worker_diag_*.log

  python train_rl_policy.py "${COMMON_ARGS[@]}" \
    --total-timesteps "${TARGET}" \
    --resume-from "${RESUME_FROM}" \
    --save-path ../../Data/ppo_slip_policy_latest --log-dir "${LOGDIR}"

  EC=$?
  if [ "${EC}" -ne 0 ]; then
    echo "=== SEGMENT toward ${TARGET} FAILED with exit code ${EC} -- stopping, not continuing ==="
    exit "${EC}"
  fi
  echo "=== SEGMENT toward ${TARGET} completed cleanly ==="
done

echo "=== ALL SEGMENTS COMPLETE -- full 300000-step run finished ==="
