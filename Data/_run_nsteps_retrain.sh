#!/bin/bash
# Longer-rollout retrain: 3 fresh seeds, n_steps 512 instead of 256, nothing else
# changed from the baseline seeds (_run_seed3_retrain.sh). Waits for the evaluation
# chain so only one job runs at a time, then scores each seed on the dev set.
# Rules: results.md, "Pre-registered: longer rollouts".
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
OUT="$DATA/_run_rl_1005.out"
RUN="$DATA/_run_resumable.sh"
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"

# Keeps waiting if the chain goes quiet, since it may only be paused by laptop
# sleep, but says so instead of hanging silently.
until grep -q ALL_DONE "$OUT"; do
  if [ -z "$(find "$DATA" -maxdepth 1 \( -name '_rl_*.log' -o -name '_evalpath_*.log' \) -mmin -60)" ]; then
    echo "$(date): evaluation chain logs quiet for over 60 minutes, still waiting"
  fi
  sleep 120
done

python ../../Data/_apply_nsteps_patch.py || { echo "PATCH_FAILED, no training run"; exit 1; }

TARGET=300000
CHECKPOINT_FREQ=1250
MAX_CONSECUTIVE_NO_PROGRESS=5
MIN_PROGRESS_STEPS=3000

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
  --n-envs 4 --checkpoint-freq $CHECKPOINT_FREQ --n-steps 512
)

train_seed() {
  local name="nsteps512_seed$1"
  local logdir="../../Data/rl_logs_$name"
  local save_path="../../Data/ppo_slip_policy_$name"
  local last_steps=0 consecutive_no_progress=0 attempt=0 latest_steps new_steps progress ec
  mkdir -p "$logdir"
  while true; do
    attempt=$(( attempt + 1 ))
    latest_steps=$(ls "$logdir"/ppo_slip_ckpt_*_steps.zip 2>/dev/null \
      | sed -E 's/.*ppo_slip_ckpt_([0-9]+)_steps\.zip/\1/' | sort -n | tail -1)
    local resume_args=()
    if [ -n "$latest_steps" ]; then
      resume_args=(--resume-from "$logdir/ppo_slip_ckpt_${latest_steps}_steps.zip")
      echo "=== $name attempt $attempt: resuming at $latest_steps steps ==="
    else
      echo "=== $name attempt $attempt: fresh start ==="
    fi
    rm -f ../../Data/_worker_diag_*.log
    python train_rl_policy.py "${COMMON_ARGS[@]}" "${resume_args[@]}" \
      --total-timesteps $TARGET --save-path "$save_path" --log-dir "$logdir" \
      >> "../../Data/_${name}_attempt_${attempt}.log" 2>&1
    ec=$?
    if [ "$ec" -eq 0 ]; then
      # A crash between the last checkpoint and the final save makes the next
      # attempt report "nothing to do" and exit 0 with no final policy.
      if [ ! -f "$save_path.zip" ] && [ -n "$latest_steps" ]; then
        cp "$logdir/ppo_slip_ckpt_${latest_steps}_steps.zip" "$save_path.zip"
        echo "=== $name: no final save, using its $latest_steps-step checkpoint ==="
      fi
      echo "=== $name finished after $attempt attempt(s) ==="
      return 0
    fi
    new_steps=$(ls "$logdir"/ppo_slip_ckpt_*_steps.zip 2>/dev/null \
      | sed -E 's/.*ppo_slip_ckpt_([0-9]+)_steps\.zip/\1/' | sort -n | tail -1)
    new_steps=${new_steps:-0}
    progress=$(( new_steps - last_steps ))
    echo "=== $name attempt $attempt exited $ec, now at $new_steps steps ==="
    if [ "$progress" -lt "$MIN_PROGRESS_STEPS" ]; then
      consecutive_no_progress=$(( consecutive_no_progress + 1 ))
    else
      consecutive_no_progress=0
    fi
    last_steps=$new_steps
    if [ "$consecutive_no_progress" -ge "$MAX_CONSECUTIVE_NO_PROGRESS" ]; then
      echo "=== $name STOPPED: no progress in $consecutive_no_progress attempts ==="
      return 1
    fi
  done
}

for s in 1 2 3; do
  name="nsteps512_seed$s"
  if train_seed "$s" && [ -f "../../Data/ppo_slip_policy_$name.zip" ]; then
    bash "$RUN" "../../Data/rl_${name}_dev_deterministic.jsonl" "../../Data/_rl_${name}_dev_deterministic.log" \
      evaluate_rl_policy.py --checkpoint ../../src/training/model_a.pt --in-channels 22 \
      --policy "../../Data/ppo_slip_policy_$name.zip" \
      --n-trials-per-object 10 --seed-start 1000 --continue-past-formation-drop
    echo "=== $name dev evaluation exited $? ==="
  else
    echo "=== $name NOT SCORED: no final policy ==="
  fi
done
echo RETRAIN_DONE
