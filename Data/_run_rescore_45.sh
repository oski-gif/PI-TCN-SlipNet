#!/bin/bash
# Reproduction check for --continue-past-formation-drop, one job at a time, on the
# original 45 trials. Default mode first, because that is the code path training uses.
RUN="/c/Users/oskic/PI-TCN-SlipNet/Data/_run_resumable.sh"
TCN=(--checkpoint ../../src/training/model_a.pt --in-channels 22)

bash "$RUN" ../../Data/rescore_check_seed3_default.jsonl ../../Data/_rescore_check_seed3_default.log \
  evaluate_rl_policy.py "${TCN[@]}" --policy ../../Data/ppo_slip_policy_seed3.zip

for pair in "seed1 latest" "seed2 seedcheck" "seed3 seed3"; do
  set -- $pair
  bash "$RUN" "../../Data/rescore_45_$1.jsonl" "../../Data/_rescore_45_$1.log" \
    evaluate_rl_policy.py "${TCN[@]}" --policy "../../Data/ppo_slip_policy_$2.zip" \
    --continue-past-formation-drop
done
echo ALL_DONE
