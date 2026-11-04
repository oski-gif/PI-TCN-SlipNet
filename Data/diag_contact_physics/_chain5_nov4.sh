#!/bin/bash
# Read-only diagnostics on the dev set: the elliptic-cone, impratio 10 setting for everything
# that has only been measured under noslip so far. Starts when the hand-tuned fresh-seed run
# frees a slot, so the machine's load stays where it was. Nothing here trains or adopts anything.
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
DIAG="$DATA/diag_contact_physics"
FRESH="$DATA/diag_fresh_seeds"
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

until grep -q -E "handA_noslip20 attempt .* exited 0|handA_noslip20 attempt 6" "$FRESH/_fresh_noslip_nov4.out"; do sleep 30; done

rl_run() {  # name policy variant
  "$PY" diagnostic_contact_physics.py "${SCENE[@]}" "${TCN_A[@]}" --policy "../../Data/ppo_slip_policy_$2.zip" \
    --variants "$3" --n-trials-per-object 10 --seed-start 1000 --stored-default "../../Data/rl_$1_dev_deterministic.jsonl" \
    --out-json "$DIAG/$1_dev150_$3.jsonl" > "$DIAG/$1_dev150_$3.log" 2>&1
  echo "$(date): RL $1 $3 exited $?"
}

grip_run() {  # set_name variant grips...
  local set_name="$1" variant="$2"
  shift 2
  "$PY" diagnostic_scripted_grip.py "${SCENE[@]}" --variant "$variant" --n-trials-per-object 10 --seed-start 1000 \
    --grips "$@" --out-json "../../Data/diag_scripted_grip/$set_name.jsonl" \
    >> "../../Data/diag_scripted_grip/$set_name.log" 2>&1
  echo "$(date): grips $set_name $variant exited $?"
}

# 1. The rest of the controllers under this setting.
rl_run seed2 seedcheck elliptic_impratio10
rl_run seed3 seed3 elliptic_impratio10
"$PY" diagnostic_handtuned_physics.py "${SCENE[@]}" "${TCN_A[@]}" --squeeze-intensity 0.95 \
  --variants elliptic_impratio10 --n-trials-per-object 10 --seed-start 1000 \
  --out-json "$DIAG/handA_dev150_elliptic_impratio10.jsonl" > "$DIAG/handA_dev150_elliptic_impratio10.log" 2>&1
echo "$(date): hand-tuned elliptic_impratio10 exited $?"

# 2. Does RL beat a fixed grip under this setting?
grip_run physics_on_grip_elliptic elliptic_impratio10 TIRP_fast seed1_like TIMRP_fast TIR_fast hand_tuned_like

# 3. Is the result sensitive to impratio? 10 is the setting in three Menagerie files, and LEAP uses 100.
# VARIANTS is extended in this command rather than in the file, because live runs depend on that file.
for pair in "elliptic_impratio3 3.0" "elliptic_impratio100 100.0"; do
  set -- $pair
  "$PY" -c "
import mujoco, sys
import diagnostic_confirm_runs as d
d.VARIANTS['$1'] = {'cone': int(mujoco.mjtCone.mjCONE_ELLIPTIC), 'impratio': $2}
d.main()
" "${SCENE[@]}" --in-channels 22 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip --variant "$1" \
    --n-trials-per-object 10 --seed-start 1000 --out-json "$DIAG/seed1_dev150_$1_noTCN.jsonl" >> "$DIAG/seed1_dev150_$1_noTCN.log" 2>&1
  echo "$(date): seed1 $1 no TCN exited $?"
done
"$PY" diagnostic_confirm_runs.py "${SCENE[@]}" --in-channels 22 --controller policy --policy ../../Data/ppo_slip_policy_latest.zip \
  --variant elliptic_impratio10 --n-trials-per-object 10 --seed-start 1000 \
  --out-json "$DIAG/seed1_dev150_elliptic_impratio10_noTCN.jsonl" >> "$DIAG/seed1_dev150_elliptic_impratio10_noTCN.log" 2>&1
echo "$(date): seed1 elliptic_impratio10 no TCN exited $?"

echo CHAIN5_DONE
