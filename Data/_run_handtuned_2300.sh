#!/bin/bash
# Usage: _run_handtuned_2300.sh A|B  -- one hand-tuned run, 15 x 67 seeds, RL's horizon.
if [ "$1" = "A" ]; then CK="model_a.pt"; CH=22; else CK="model_b.pt"; CH=25; fi
bash "/c/Users/oskic/PI-TCN-SlipNet/Data/_run_resumable.sh" \
  "../../Data/handtuned_model$1_2300step_results.jsonl" "../../Data/_handtuned_model$1_2300step.log" \
  batch_closed_loop_trials.py --n-trials-per-object 67 --max-steps 2300 --squeeze-intensity 0.95 \
  --checkpoint "../../src/training/$CK" --in-channels $CH
