#!/bin/bash
# Read-only diagnostics on the dev set, one process at a time. Replaces _chain3_nov4.sh, which
# never started: chain 2's later steps did run, so chain 3 would have appended duplicate rows.
# Nothing here trains, edits a script, changes a setting anywhere else or touches a stored result.
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
DIAG="$DATA/diag_contact_physics"
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
TCN_A=(--checkpoint ../../src/training/model_a.pt --in-channels 22)

until grep -q CHAIN2_DONE "$DIAG/_chain2_nov4.out"; do sleep 20; done

grip_run() {  # set_name variant grips...
  local set_name="$1" variant="$2"
  shift 2
  "$PY" diagnostic_scripted_grip.py "${SCENE[@]}" --variant "$variant" --n-trials-per-object 10 --seed-start 1000 \
    --grips "$@" --out-json "../../Data/diag_scripted_grip/$set_name.jsonl" \
    >> "../../Data/diag_scripted_grip/$set_name.log" 2>&1
  echo "$(date): grips $set_name $variant exited $?"
}

# No TCN. Without --checkpoint the policy sees a slip probability of 0 and hand-tuned never corrects.
confirm_no_tcn() {  # file_tag variant controller arguments...
  local tag="$1" variant="$2"
  shift 2
  "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" --in-channels 22 "$@" --variant "$variant" \
    --n-trials-per-object 10 --seed-start 1000 --out-json "$DIAG/$tag.jsonl" >> "$DIAG/$tag.log" 2>&1
  echo "$(date): $tag exited $?"
}

SEED1=(--controller policy --policy ../../Data/ppo_slip_policy_latest.zip)
SEED2=(--controller policy --policy ../../Data/ppo_slip_policy_seedcheck.zip)
SEED3=(--controller policy --policy ../../Data/ppo_slip_policy_seed3.zip)
HAND=(--controller hand)

# 1. Does RL beat a fixed grip when the contact settings are stricter?
grip_run physics_on_grip noslip20 TIRP_fast seed1_like TIMRP_fast TIR_fast hand_tuned_like

# 2. What do the n_steps 512 policies do with their fingers? With the TCN, so the rows must reproduce.
for s in 1 2 3; do
  "$PY" diagnostic_contact_physics.py "${SCENE[@]}" "${TCN_A[@]}" --policy "../../Data/ppo_slip_policy_nsteps512_seed$s.zip" \
    --variants default --n-trials-per-object 1 --seed-start 1000 \
    --stored-default "../../Data/rl_nsteps512_seed${s}_dev_deterministic.jsonl" \
    --out-json "$DIAG/nsteps512_seed${s}_dev_15trials.jsonl" > "$DIAG/nsteps512_seed${s}_dev_15trials.log" 2>&1
  echo "$(date): nsteps512 seed $s traces exited $?"
done

# 3. Does the TCN matter under noslip, for RL and for hand-tuned? And for hand-tuned under default?
confirm_no_tcn seed1_dev150_noslip20_noTCN noslip20 "${SEED1[@]}"
confirm_no_tcn seed2_dev150_noslip20_noTCN noslip20 "${SEED2[@]}"
confirm_no_tcn seed3_dev150_noslip20_noTCN noslip20 "${SEED3[@]}"
confirm_no_tcn handA_dev150_default_noTCN default "${HAND[@]}"
confirm_no_tcn handA_dev150_noslip20_noTCN noslip20 "${HAND[@]}"

# 4. Is the noslip result sensitive to the iteration count, or to the friction cone?
confirm_no_tcn seed1_dev150_noslip5_noTCN noslip5 "${SEED1[@]}"
confirm_no_tcn seed1_dev150_noslip50_noTCN noslip50 "${SEED1[@]}"
confirm_no_tcn seed1_dev150_elliptic_noslip20_noTCN elliptic_noslip20 "${SEED1[@]}"
confirm_no_tcn handA_dev150_noslip5_noTCN noslip5 "${HAND[@]}"
confirm_no_tcn handA_dev150_noslip50_noTCN noslip50 "${HAND[@]}"

# 5. Do seeds 2 and 3 differ from seed 1 mainly in which fingers they close?
grip_run explain_seeds default seed2_like seed3_like TIP_fast
grip_run explain_seeds noslip20 seed2_like seed3_like TIP_fast

echo CHAIN4_DONE
