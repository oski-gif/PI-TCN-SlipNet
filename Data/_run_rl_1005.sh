#!/bin/bash
# Evaluation only, no training, one job at a time, chained so it keeps going
# without anyone watching. Paths and rules: results.md, "Pre-registered evaluation rules".
RUN="/c/Users/oskic/PI-TCN-SlipNet/Data/_run_resumable.sh"
TCN=(--checkpoint ../../src/training/model_a.pt --in-channels 22)
POLICIES=("seed1 latest" "seed2 seedcheck" "seed3 seed3")

# Existing policies on the 15 x 67 locked set, deterministic. Trials run past a 5 cm
# fall so each can be scored at any cutoff. Seed 1 is the 64.4% policy: report it as
# "best of 3 seeds, deterministic".
for pair in "${POLICIES[@]}"; do
  set -- $pair
  bash "$RUN" "../../Data/rl_$1_1005trial_results.jsonl" "../../Data/_rl_$1_1005trial.log" \
    evaluate_rl_policy.py "${TCN[@]}" --policy "../../Data/ppo_slip_policy_$2.zip" \
    --n-trials-per-object 67 --seed-start 0 --continue-past-formation-drop
done
echo LOCKED_DONE

# Eval-path check on the 2.2% policy, original 45 trials, default trial handling.
VN=(evaluate_rl_policy.py "${TCN[@]}" --policy ../../Data/ppo_slip_policy_vecnorm.zip)
bash "$RUN" ../../Data/evalpath_direct.jsonl ../../Data/_evalpath_direct.log "${VN[@]}" --eval-path direct
bash "$RUN" ../../Data/evalpath_wrapped.jsonl ../../Data/_evalpath_wrapped.log "${VN[@]}" --eval-path wrapped
bash "$RUN" ../../Data/evalpath_wrapped_stochastic.jsonl ../../Data/_evalpath_wrapped_stochastic.log \
  "${VN[@]}" --eval-path wrapped --stochastic
echo EVALPATH_DONE

# Deterministic versus stochastic on the dev set, seeds 1000-1009.
for pair in "${POLICIES[@]}"; do
  set -- $pair
  DEV=(evaluate_rl_policy.py "${TCN[@]}" --policy "../../Data/ppo_slip_policy_$2.zip"
       --n-trials-per-object 10 --seed-start 1000 --continue-past-formation-drop)
  bash "$RUN" "../../Data/rl_$1_dev_deterministic.jsonl" "../../Data/_rl_$1_dev_deterministic.log" "${DEV[@]}"
  bash "$RUN" "../../Data/rl_$1_dev_stochastic.jsonl" "../../Data/_rl_$1_dev_stochastic.log" "${DEV[@]}" --stochastic
done
echo ALL_DONE
