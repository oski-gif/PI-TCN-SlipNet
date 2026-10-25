# Results Log

Locked-in numbers for the paper go here, labeled clearly, so re-running
training/evaluation for debugging or tuning later never creates ambiguity
about which number is the one actually cited in the write-up.

## Model A (plain baseline TCN) -- initial successful run

- **Date:** 2026-10-05
- **Architecture:** `TCN(in_channels=22, hidden_channels=32, kernel_size=3, num_layers=7)`
- **Training:** 20 epochs, Adam lr=1e-3, `CrossEntropyLoss` with train-set class_weights, per-timestep labels (`y_series`)
- **Data:** `Data/pitcn_dataset_full_random_split.npz` (random split, train=12259 / val=2627 / test=2628), seq_len-masked evaluation (padded timesteps excluded)
- **F1 score: 0.7189987008100414**
- **Accuracy: 0.8912098815652408**

This is the number to cite for Model A's baseline result unless a deliberate
retrain is run specifically to replace it (e.g. after the planned tuning
pass - dropout, more epochs, lr schedule). If that happens, add a new
dated entry below rather than overwriting this one, so the history of
what was tried stays visible.

## Model B (PI-TCN) -- initial successful run

- **Date:** 2026-10-14 (trained) / 2026-10-16 (evaluated)
- **Architecture:** `TCN(in_channels=23, hidden_channels=32, kernel_size=3, num_layers=7)` -- identical to Model A except for 1 extra input channel
- **Physics-informed element:** `mu_combined` (`max(object_friction_sliding, pad_friction_mean)`, the verified MuJoCo contact-friction rule), broadcast as a constant 23rd input channel across every timestep of each trial. This is the ONLY physics element in the final model -- see project_scope_priority memory / this conversation's extensive 2026-10-14/15 investigation for the full record of 9 tested feature/loss formulations (ratio, difference, trend, torque-based, stiffened-contact simulation) that were tried and found NOT to discriminate real slip onset from stable grasping, before this safe fallback was adopted.
- **Training:** identical setup to Model A -- 20 epochs, Adam lr=1e-3, `CrossEntropyLoss` with train-set class_weights, per-timestep labels (`y_series`)
- **Data:** same `Data/pitcn_dataset_full_random_split.npz` split as Model A, so the comparison is apples-to-apples
- **F1 score: 0.7149740482699957**
- **Accuracy: 0.8914659190063294**

Essentially indistinguishable from Model A (F1 0.715 vs. Model A's 0.715-0.719
across its two runs; accuracy 0.891 vs. 0.886-0.891). Consistent with the
onset-visualization finding (2026-10-15) that tactile signals show no visible
pre-slip pattern in this data, while `pull_force` -- already available to
BOTH models -- does. A single per-trial friction scalar was not expected to
move performance much given that finding, and it didn't. Same retrain-replaces-this-entry
rule as Model A applies here too.

## Model A + Model B -- retrained with label_smoothing=0.1

- **Date:** 2026-10-19
- **Why:** `label_smoothing=0.1` added to `CrossEntropyLoss` in both `train.py`/`train_pitcn.py`, primarily to get better-behaved softmax probabilities for `closed_loop_sim.py`'s `predict_slip()` (readout changed same day from argmax to raw slip-class probability, feeding a probability-scaled `decide_action()` control policy). No other change -- same architecture, same data, same 20 epochs/Adam lr=1e-3.
- **Model A -- F1 score: 0.7728929388214366 / Accuracy: 0.8989912209391691** (up from F1 0.715-0.719 / accuracy 0.886-0.891)
- **Model B -- F1 score: 0.7304784804871298 / Accuracy: 0.8945878420906525** (up from F1 0.715 / accuracy 0.891)

Genuine improvement on both, not just noise-level variation -- label smoothing
was expected to help calibration, but it meaningfully improved classification
metrics too, especially for Model A. val_loss itself is NOT comparable to
earlier entries (label smoothing changes the loss target, raising its floor
even for a perfect model) -- F1/accuracy above are the real comparison.
**These are now the numbers to cite for both models** unless a further
deliberate retrain replaces them -- same append-only rule as above.

## Model B -- retrained again with obj_mass + grasp_force_at_settle added

- **Date:** 2026-10-19
- **Why:** attempt to get Model B to genuinely outperform Model A (not just tie), by adding 2 more safe scalar features -- `obj_mass` and `grasp_force_at_settle`, same broadcast-constant pattern as `mu_combined`, no new formula risk. `in_channels` 23 -> 25.
- **F1 score: 0.7537170771391258 / Accuracy: 0.9057822387820821**

Mixed result, not a clean win -- report honestly. Vs. Model B's own prior
(label-smoothed, 23ch) run: BOTH metrics improved (F1 0.730->0.754, accuracy
0.895->0.906), so the new features did help. Vs. Model A (label-smoothed,
F1 0.773/accuracy 0.899): Model B's accuracy is now HIGHER (0.906 vs 0.899),
but its F1 is still LOWER (0.754 vs 0.773). Likely a precision/recall
shift on the minority (slip) class, not uniform improvement. Superseded by
the threshold-tuning entry directly below -- see that one for the number to
actually cite.

## Model B (25ch) -- decision threshold tuned on validation set

- **Date:** 2026-10-24
- **Why:** the prior entry's F1 shortfall vs. Model A was diagnosed as a precision/recall imbalance at the default 0.5 decision threshold, not a modeling problem -- cheap to test directly via a threshold sweep, no retraining. Script: `src/training/eval_tcn_threshold.py`. Threshold swept on VALIDATION only (0.05 to 0.95 step 0.05), chosen threshold then applied once to the untouched TEST set -- never tuned directly against test.
- **Best threshold on val: 0.30** (val F1 = 0.7840)
- **F1 score at tuned threshold: 0.7820409365251447**
- **Accuracy at tuned threshold: 0.9042320599892976**
- (F1 at default 0.5, for comparison: 0.7537170771391258)

**Clean win over Model A on both metrics** -- F1 0.782 vs. Model A's 0.773,
accuracy 0.904 vs. Model A's 0.899. The model doesn't need to change, just
be read out with the correct decision boundary: 0.30, not the default 0.5
`argmax` was implicitly using. Note that `closed_loop_sim.py`'s
`decide_action()` consumes the raw probability directly (not a hard
threshold), so this 0.30 figure is specific to evaluation/reporting, not
something that needs wiring into the closed-loop control policy itself.
Superseded by the entry directly below.

Checkpoint backed up as `model_b_f1_0782_backup.pt` before further tuning.

## Model B (25ch) -- retrained with num_epochs=40 (was 20)

- **Date:** 2026-10-24
- **Why:** targeting F1=0.85, pursued as a deliberate, isolated-variable tuning pass (one change at a time, so it's clear what's actually responsible for any difference) -- val_loss hadn't clearly plateaued by epoch 20 in any prior run, so more epochs was the first, cheapest lever to test. No other change from the 25ch/label-smoothed config.
- **Best threshold on val: 0.30** (val F1 = 0.7919)
- **F1 score at tuned threshold: 0.7985883040854257**
- **Accuracy at tuned threshold: 0.9082381685225329**
- (F1 at default 0.5, for comparison: 0.772205528667876)

Real improvement, not noise -- F1 0.782->0.799, accuracy 0.904->0.908,
same 0.30 threshold both times. **This is now the number to cite for Model
B.** Checkpoint backed up as `model_b_f1_0799_backup.pt`. Gap to the 0.85
target: ~0.05 F1 remaining -- next lever being tested is dropout, then
multi-seed ensembling for both the remaining gap and run-to-run consistency
(Model A alone has shown F1 vary 0.693-0.719 across identical-config runs,
so a single run's number was never going to be "consistent" by itself).