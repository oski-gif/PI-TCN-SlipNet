#!/bin/bash
# Read-only diagnostics, one process at a time, alongside the longer-rollout retrain.
# Nothing here trains, edits a script or touches a stored result. Waits for the
# contact-physics run, then traces seeds 2 and 3, then runs the scripted grips.
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
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

until grep -q "^== noslip20" "$DATA/diag_contact_physics/seed1_dev_15trials.log"; do sleep 20; done

for pair in "seed2 seedcheck" "seed3 seed3"; do
  set -- $pair
  python diagnostic_contact_physics.py "${SCENE[@]}" --checkpoint ../../src/training/model_a.pt --in-channels 22 \
    --policy "../../Data/ppo_slip_policy_$2.zip" --variants default --n-trials-per-object 1 --seed-start 1000 \
    --stored-default "../../Data/rl_$1_dev_deterministic.jsonl" \
    --out-json "../../Data/diag_contact_physics/$1_dev_15trials.jsonl" \
    > "../../Data/diag_contact_physics/$1_dev_15trials.log" 2>&1
  echo "$(date): $1 trace exited $?"
done

python diagnostic_scripted_grip.py "${SCENE[@]}" --n-trials-per-object 10 --seed-start 1000 \
  --out-json ../../Data/diag_scripted_grip/dev_first_grips.jsonl \
  > ../../Data/diag_scripted_grip_dev_first_grips.log 2>&1
echo "$(date): scripted grips exited $?"
echo CHAIN_DONE
