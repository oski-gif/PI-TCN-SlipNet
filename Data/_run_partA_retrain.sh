#!/bin/bash
# Full fresh retrain for Part A (char_length added to the observation
# space, reward unchanged). MUST be a fresh LOGDIR, distinct from
# Data/rl_logs_fresh_retrain (which holds checkpoints from the OLD
# 115-dim observation space + MIN_FINGER_FRACTION_COEF=3.0 attempt --
# loading one of those into the new 116-dim env would be a shape
# mismatch). Same self-healing auto-resume-on-crash pattern as before.
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"

TARGET=300000
LOGDIR="../../Data/rl_logs_partA_full"
SAVE_PATH="../../Data/ppo_slip_policy_partA"
CHECKPOINT_FREQ=1250
MAX_CONSECUTIVE_NO_PROGRESS=5
MIN_PROGRESS_STEPS=3000

mkdir -p "$LOGDIR"

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
  --n-envs 4 --checkpoint-freq $CHECKPOINT_FREQ
)

get_latest_ckpt_steps() {
  ls "$LOGDIR"/ppo_slip_ckpt_*_steps.zip 2>/dev/null \
    | sed -E 's/.*ppo_slip_ckpt_([0-9]+)_steps\.zip/\1/' \
    | sort -n | tail -1
}

last_steps=0
consecutive_no_progress=0
attempt=0

while true; do
  attempt=$(( attempt + 1 ))
  latest_steps=$(get_latest_ckpt_steps)
  if [ -n "$latest_steps" ]; then
    latest_ckpt="$LOGDIR/ppo_slip_ckpt_${latest_steps}_steps.zip"
    RESUME_ARGS=(--resume-from "$latest_ckpt")
    echo "=== Attempt ${attempt}: resuming from ${latest_ckpt} (steps=${latest_steps}) ==="
  else
    RESUME_ARGS=()
    echo "=== Attempt ${attempt}: FRESH START, no checkpoint yet ==="
  fi

  rm -f ../../Data/_worker_diag_*.log
  python train_rl_policy.py "${COMMON_ARGS[@]}" "${RESUME_ARGS[@]}" \
    --total-timesteps $TARGET --save-path "$SAVE_PATH" --log-dir "$LOGDIR" \
    > "../../Data/_partA_retrain_attempt_${attempt}.log" 2>&1
  EC=$?

  new_latest_steps=$(get_latest_ckpt_steps)
  new_latest_steps=${new_latest_steps:-0}

  if [ "$EC" -eq 0 ]; then
    echo "=== Attempt ${attempt} completed cleanly (exit 0), latest checkpoint steps=${new_latest_steps} ==="
    echo "=== LOOP DONE -- verify ${SAVE_PATH}.zip's real num_timesteps separately ==="
    break
  fi

  progress_this_attempt=$(( new_latest_steps - last_steps ))
  echo "=== Attempt ${attempt} CRASHED (exit ${EC}) -- latest checkpoint now at ${new_latest_steps} steps (${progress_this_attempt} steps of progress this attempt) ==="

  if [ "$progress_this_attempt" -lt "$MIN_PROGRESS_STEPS" ]; then
    consecutive_no_progress=$(( consecutive_no_progress + 1 ))
  else
    consecutive_no_progress=0
  fi
  last_steps=$new_latest_steps

  if [ "$consecutive_no_progress" -ge "$MAX_CONSECUTIVE_NO_PROGRESS" ]; then
    echo "=== STOPPING: ${consecutive_no_progress} consecutive attempts each made <${MIN_PROGRESS_STEPS} steps of progress. Needs a decision, not more auto-retrying. ==="
    exit 1
  fi
done

echo "=== ORCHESTRATOR LOOP EXITED ==="
