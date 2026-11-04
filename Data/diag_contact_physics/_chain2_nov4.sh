#!/bin/bash
# Read-only sensitivity runs on the dev set, one process at a time, after _chain_nov4.sh.
# Nothing here trains, edits a script or touches a stored result. Every run has its own
# output file. A run that fails is reported here and the next one still goes.
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
OUT="$DATA/diag_contact_physics"
cd "/c/Users/oskic/PI-TCN-SlipNet/Synthetic data creation/Simulation"

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

until grep -q CHAIN_DONE "$OUT/_chain_nov4.out"; do sleep 20; done

rl_run() {  # name policy variant stored
  python diagnostic_contact_physics.py "${SCENE[@]}" "${TCN_A[@]}" --policy "../../Data/ppo_slip_policy_$2.zip" \
    --variants "$3" --n-trials-per-object 10 --seed-start 1000 --stored-default "../../Data/rl_$1_dev_deterministic.jsonl" \
    --out-json "../../Data/diag_contact_physics/$1_dev150_$3.jsonl" > "../../Data/diag_contact_physics/$1_dev150_$3.log" 2>&1
  echo "$(date): $1 $3 exited $?"
}

# The same policies with the TCN signal removed: no --checkpoint means slip probability is always 0.
no_tcn_run() {  # name policy
  python diagnostic_contact_physics.py "${SCENE[@]}" --in-channels 22 --policy "../../Data/ppo_slip_policy_$2.zip"     --variants default --n-trials-per-object 10 --seed-start 1000 --stored-default "../../Data/rl_$1_dev_deterministic.jsonl"     --out-json "../../Data/diag_contact_physics/$1_dev150_noTCN.jsonl" > "../../Data/diag_contact_physics/$1_dev150_noTCN.log" 2>&1
  echo "$(date): $1 without TCN exited $?"
}

no_tcn_run seed1 latest
no_tcn_run seed2 seedcheck
no_tcn_run seed3 seed3

rl_run seed1 latest default
rl_run seed1 latest noslip20

python diagnostic_handtuned_physics.py "${SCENE[@]}" "${TCN_A[@]}" --squeeze-intensity 0.95 \
  --variants default --n-trials-per-object 1 --seed-start 0 \
  --out-json ../../Data/diag_contact_physics/handA_repro15_default.jsonl \
  > ../../Data/diag_contact_physics/handA_repro15_default.log 2>&1
echo "$(date): hand-tuned reproduction exited $?"
for variant in default noslip20; do
  python diagnostic_handtuned_physics.py "${SCENE[@]}" "${TCN_A[@]}" --squeeze-intensity 0.95 \
    --variants "$variant" --n-trials-per-object 10 --seed-start 1000 \
    --out-json "../../Data/diag_contact_physics/handA_dev150_$variant.jsonl" \
    > "../../Data/diag_contact_physics/handA_dev150_$variant.log" 2>&1
  echo "$(date): hand-tuned $variant exited $?"
done

rl_run seed2 seedcheck noslip20
rl_run seed3 seed3 noslip20
rl_run seed1 latest elliptic_impratio10
echo CHAIN2_DONE
