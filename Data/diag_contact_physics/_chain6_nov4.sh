#!/bin/bash
# Read-only diagnostics on the dev set, after chain 4 finishes. Nothing here trains or adopts anything.
# 1. Does a seed's held rate climb, level off or bounce during training? Saved checkpoints with the
#    TCN removed for speed: that changed 0 of 450 outcomes on the final policies. 45 trials each, so
#    this is a coarse first look, about 7 points of standard error per checkpoint.
# 2. What can a fixed grip reach, under the setting that's the likelier candidate? Choosing among
#    these on 150 trials inflates the best one, so a confirmation would need fresh seeds.
DATA="/c/Users/oskic/PI-TCN-SlipNet/Data"
DIAG="$DATA/diag_contact_physics"
CURVES="$DATA/diag_checkpoint_curves"
GRIPS="$DATA/diag_scripted_grip"
mkdir -p "$CURVES"
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

until grep -q CHAIN4_DONE "$DIAG/_chain4_nov4.out"; do sleep 30; done

curve() {  # run_name log_dir steps...
  local run="$1" dir="$2" step
  shift 2
  for step in "$@"; do
    "$PY" diagnostic_confirm_runs.py "${SCENE[@]}" --in-channels 22 --controller policy \
      --policy "../../Data/$dir/ppo_slip_ckpt_${step}_steps.zip" --variant default \
      --n-trials-per-object 3 --seed-start 1000 --out-json "$CURVES/${run}_${step}.jsonl" >> "$CURVES/${run}_${step}.log" 2>&1
    echo "$(date): curve $run $step exited $?"
  done
}

grip_run() {  # set_name variant grips...
  local set_name="$1" variant="$2"
  shift 2
  "$PY" diagnostic_scripted_grip.py "${SCENE[@]}" --variant "$variant" --n-trials-per-object 10 --seed-start 1000 \
    --grips "$@" --out-json "$GRIPS/$set_name.jsonl" >> "$GRIPS/$set_name.log" 2>&1
  echo "$(date): grips $set_name $variant exited $?"
}

curve seed1 rl_logs_production 40000 80000 135360 295168
curve seed1 rl_logs_continue 300224 350224 400224 450224
curve seed2 rl_logs_seedcheck 50000 100000 150000 200000 225000 250000 275000 300000
curve seed3 rl_logs_seed3 50000 100000 150000 200000 225000 250000 275000 300000
curve n512_seed1 rl_logs_nsteps512_seed1 100000 200000 300000
curve n512_seed2 rl_logs_nsteps512_seed2 100000 200000 300000
curve n512_seed3 rl_logs_nsteps512_seed3 100000 200000 300000

grip_run poses_elliptic elliptic_impratio10 S_thumb_c S_thumb_c2 S_inward S_thumb_c2_inward M_thumb_c M_thumb_c2 M_inward M_thumb_c2_inward
grip_run subsets_elliptic elliptic_impratio10 TIP_fast TRP_fast TI_fast TR_fast IRP_fast TMR_fast TIMR_fast TMRP_fast
grip_run timing_elliptic elliptic_impratio10 S_pinky30 S_pinky45 S_pinky80 S_ring60 S_ring100 S_index80 S_thumb80
grip_run middle_and_soft_elliptic elliptic_impratio10 S_M_late30 S_M_late60_slow S_M_partial S_soft90 S_soft80
grip_run poses default S_thumb_c S_thumb_c2 S_inward S_thumb_c2_inward M_thumb_c M_thumb_c2 M_inward M_thumb_c2_inward

echo CHAIN6_DONE
