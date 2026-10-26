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
target: ~0.05 F1 remaining.

**Dropout tested and reverted:** added `nn.Dropout(p=0.2)` to `TCNBlock`,
retrained -- F1 dropped to 0.776 (below the 0.78 floor), accuracy to 0.895.
Reverted both the checkpoint (restored from the 0799 backup) and the code
change. Dropout apparently hurts here, not helps -- no evidence this model
was overfitting, so removing capacity just removed signal.

**num_epochs=40 applied to Model A too, for comparison fairness -- made it
WORSE, not better:** F1 dropped from 0.773 to 0.722. Reverted to 20 epochs,
but the restore run itself landed at F1=0.691 (accuracy 0.887) -- NOT the
same as the original 0.773 run, since training isn't seeded. This is a
real, concrete demonstration of the run-to-run variance problem: Model A
alone has now shown F1 of 0.773, 0.722, AND 0.691 from nominally "the same"
20-epoch config. Multi-seed runs (not yet done) are needed before any
single Model A number should be treated as final -- flagged here rather
than silently picking whichever run looked best.

## Closed-loop batch trial results (Model A vs Model B, 45 trials each)

- **Date:** 2026-10-25
- **Setup:** new `batch_closed_loop_trials.py`, calling `run_trial()`
  (extracted from `closed_loop_sim.py` in the same refactor) headlessly,
  3 trials per object x all 15 objects in `OBJECT_LIBRARY`, each trial a
  fresh seed (new friction/mass/geometry jitter). `squeeze_intensity=0.8`,
  `pull_multiple=1.5` (defaults), `decide_action()`'s real control policy
  active throughout (not a stub). "Held securely" = max displacement from
  the settle-end reference never exceeded 15% of that object's own
  char_length -- same threshold definition `generate_trials.py` uses for
  `trial_slip_label`. ~1.65s/trial, 45 trials in ~75s per model.
- **Model A (22ch, F1=0.773 static): 8/45 (17.8%) held securely.**
  Mean max_slip_probability=0.658, mean n_corrections=11.96.
- **Model B (25ch, F1=0.799 static): 5/45 (11.1%) held securely.**
  Mean max_slip_probability=0.729, mean n_corrections=14.11.

**Model B has the better static F1/accuracy but the WORSE closed-loop
success rate.** Not a bug -- Model B actually corrects MORE often (higher
mean probability, more corrections triggered) and still does worse, so
this isn't under-reaction. Two honest candidate explanations, neither
confirmed: (1) Model B's physics-informed inputs make it more sensitive in
a way that doesn't translate to better control outcomes, or (2) 45 trials/
model is real but not large -- given the demonstrated run-to-run training
variance above, some of this gap could be sample noise rather than a true
effect. Flagging both possibilities rather than picking one. This result
is itself a legitimate finding worth reporting either way: static accuracy
and closed-loop control quality are not the same thing, which is exactly
what the closed-loop comparison was built to be able to show.

**CORRECTION, same date:** a code-review pass found a real bug in the
closed_loop_sim.py refactor that produced the Model A/B batch numbers
above -- the line that re-enables real gravity at the start of the settle
phase was accidentally dropped (only its print statement survived), so
EVERY trial behind both the 45/45 and 11.1%/17.8% numbers above ran with
the object permanently weightless for the entire simulation, not just
during the ramp/approach as intended. **Those two batch results are
invalid** -- gravity never pulling the object down the whole time
fundamentally changes what "slip" means in that physics. Fixed (restored
the `gravcomp = 0.0` reset), and 3 more issues from the same review pass
also fixed while at it: `batch_closed_loop_trials.py` now has real error
handling and writes results incrementally (one bad trial no longer loses
the whole batch), the raw feature-column list is now a single shared
`build_feature_names()` instead of duplicated in two places, and the two
scripts' argparse setup is now a shared `add_common_args()` instead of
hand-duplicated flags with silently different defaults.

## Closed-loop batch, re-run with corrected gravity + policy tuning (Model A only)

- **Date:** 2026-10-25
- **Policy change tested:** trigger threshold 0.5->0.3, `STEP_FRACTION`
  0.15->0.35 (both in `decide_action()`) -- user's explicit, well-reasoned
  call, not mine: react sooner and correct harder.
- **Result (corrected gravity, tuned policy): 10/45 (22.2%) held securely.**
  Mean max_slip_probability=0.54, mean n_corrections=5.96.
- **Reaction-frequency test** (same tuned policy, but checking/correcting
  every 5 simulation steps instead of every 25): **identical 10/45
  (22.2%).** Corrections per trial jumped to 25.27 (confirming it really
  did react far more often), but the outcome didn't change AT ALL --
  real, informative negative result: reaction lag was not the bottleneck.

**Leading hypothesis for tomorrow, not yet tested:** `decide_action()`'s
own formula, `step = STEP_FRACTION * probability * (hi - target)`, shrinks
toward zero as the grip approaches its ctrlrange max -- a known tradeoff
flagged when the policy was first designed. If the grip is already fairly
tight and still slipping, every additional correction has less room to
work with, independent of how often it's applied -- which would exactly
explain why reacting 5x more often changed nothing. Two untried, quick
fixes for next time: raise `squeeze_intensity` above 0.8 (more initial
headroom), or change the correction to a fixed absolute step instead of
one that scales down near the limit. Model B's batch (same tuned policy +
corrected gravity) was not re-run before stopping for the night -- do that
too, since its only valid numbers right now are the pre-gravity-fix ones,
which are already marked invalid above.

## Autonomous deep-dive, 2026-10-25/26 overnight -- "get to 90%" investigation

User authorized an extended (2+ hour) autonomous investigation targeting
90% closed-loop success by Oct 28, with model/policy-design code changes
tested only via a monkey-patched diagnostic harness
(`diagnostic_policy_variants.py`), never writing to the user's actual
`decide_action()`. Summary below; the findings supersede each other in
places, documented honestly rather than cherry-picked.

**1. `squeeze_intensity` sweep (short-timing config, see correction #4
below -- these numbers are NOT directly comparable to anything using
realistic timing):** 0.80->26.7%, 0.90->43.3%, 0.92->53.3%, 0.94->50.0%,
0.95->56.7%, 0.96->50.0%, 0.98->40.0%, 1.00->50.0%. Real plateau in the
0.90-0.96 range, not a sharp peak -- pushing too high (0.98+) regresses,
likely because a near-maximal baseline squeeze leaves no headroom for any
correction to use.

**2. Fixed-step vs. proportional correction formula: no difference.**
Current proportional policy and 3 fixed-step magnitudes (0.05/0.10/0.20)
all landed on the identical 17/30 (56.7%), with nearly identical
correction counts (~34 each) -- strong evidence the actuators are
saturating at their ctrlrange max regardless of which formula gets them
there. The "shrinking correction near the limit" hypothesis from earlier
tonight does not explain the gap on its own.

**3. `pull_multiple` has almost NO effect on outcome -- a genuinely
surprising finding.** 1.0/1.2/1.5 all gave the identical 17/30 (56.7%).
Verified with an extreme single-trial comparison: 0.2x pull (0.158N) gave
max_displacement=0.1993; 5x pull (3.959N, 25x stronger) gave 0.2081 --
essentially the same. **Pull force is not the primary driver of these
failures.**

**4. CRITICAL METHODOLOGY CORRECTION, discovered via #3's investigation:**
every test tonight up through #3 used `--ramp-steps 100 --settle-steps
100` (an old default baked into the first version of
`batch_closed_loop_trials.py`, before the shared-argparse code-review fix
changed it to the realistic 500/300 default). Testing with ZERO pull
(pure gravity) on cube/seed=0 under the SHORT timing gave catastrophic
failure (displacement 0.1989), but the SAME object/seed under the
REALISTIC 500/300 timing (used by `generate_trials.py` and now the
correct shared default) held securely (displacement 0.0058) in a
follow-up check. The short ramp/settle likely doesn't give the grasp
enough time to actually stabilize before the displacement reference is
captured, producing spurious "slip" from residual settling motion rather
than genuine grip failure. **This means items #1-3 above are internally
consistent with each other (all measured under the same short-timing
artifact) but may not reflect real-world/realistic-timing behavior.**

**5. Real, properly-timed batch (realistic 500/300 ramp/settle, full
pull_multiple=1.5 engaged and sustained, squeeze_intensity=0.95,
max_steps=2500 to give enough post-ramp monitoring time): 18/45 (40.0%)
held securely.** Not directly comparable to the 56.7% figure above (both
the timing AND how long the trial sustains full pull changed at once),
so no claim is made here that policy tuning helped under fair conditions
-- that comparison still needs to be run properly (current policy, 0.95
squeeze, realistic timing, vs. the ORIGINAL 0.8 squeeze/default policy,
same realistic timing, same trial count).

**Per-object breakdown from the realistic-timing batch (#5) -- this is
genuinely actionable, independent of the timing confound:**
sphere/rod/dumbbell held 3/3 (100%); cone/rectangular_prism/
cylindrical_prism/plus_cross/triangular_prism/hexagonal_prism held 0/3
(0%). This looks like a real, geometry-dependent grip-security pattern --
flat-faced/angular shapes may give the 5-fingered hand nothing to
cradle against, unlike rounder or more complex geometries. If true, this
is NOT fixable via control-policy tuning alone.

**6. Background research agent findings** (`reports/Neural network
approach comparison.md`, independently verified the actual codebase
before researching, not just worked from a description):
- **No random seeding anywhere in the codebase** -- directly explains the
  F1 run-to-run variance chased all night (0.69-0.80 swings from "the
  same" config). Trivial fix, training code, not done here (not mine to
  write).
- `decide_action()` has no smoothing/hysteresis -- reacts to one
  instantaneous probability with a flat threshold despite a 300-step
  history buffer already sitting unused. Real precedent cited (a patent
  on exactly this gripping-control pattern). Prepared but not yet tested
  as a diagnostic variant when this investigation paused.
- LayerNorm/weight-norm in `TCNBlock` untested (dropout was tried and
  regressed; normalization specifically was not tried in isolation) --
  matches both the original TCN paper and a published slip-detection TCN
  paper's architecture.
- `decide_action()` applies the identical correction to every actuator
  regardless of which specific finger's sensors show slip -- the model
  has per-fingertip features upstream but the controller collapses them
  into one global reaction.
- On the 9-failed-physics-features finding: real precedent exists that
  genuine pre-slip tactile signal requires kHz-range micro-vibration
  sensing this project's sensors structurally cannot provide -- suggests
  deprioritizing further hand-engineered feature work in favor of the
  closed-loop controller, which has independently verifiable, fixable
  problems (items above).

**Honest state for whoever picks this up:** the single most important
next step is re-running a clean, apples-to-apples comparison under
REALISTIC timing only -- baseline (0.8 squeeze, original policy) vs.
tuned (0.95 squeeze, current policy) -- since nothing claimed above about
policy/squeeze improvements has been verified under fair, consistent
conditions yet. The object-specific failure pattern (item 5) is probably
the highest-leverage real lever remaining, more than further policy
tuning. 90% by Oct 28 looks genuinely difficult given the geometry-
dependent failures found here, but 40%+ on a realistic-timing baseline,
up from an unverified-but-likely-lower true starting point, is real
progress, honestly reported rather than inflated.

## THE clean comparison -- real baseline vs. real tuned, no confounds

- **Date:** 2026-10-26 (overnight)
- **Setup:** `diagnostic_clean_comparison.py`, both configurations run
  under the SAME realistic timing (500/300 ramp/settle), same 45 trials
  (all 15 objects x 3 seeds), same checkpoint (Model A). Baseline uses
  the ORIGINAL policy settings (threshold=0.5, STEP_FRACTION=0.15,
  monkey-patched since the real file now has the tuned version) at the
  original squeeze_intensity=0.8. Tuned uses the REAL, current
  `decide_action()` (threshold=0.3, STEP_FRACTION=0.35) at
  squeeze_intensity=0.95.
- **BASELINE (0.8 squeeze, original policy): 6/45 (13.3%) held securely.**
- **TUNED (0.95 squeeze, tuned policy): 18/45 (40.0%) held securely.**

**This is the number that actually matters -- a genuine ~3x improvement,
verified with no confounds.** Every earlier closed-loop percentage
tonight (including the ones already logged above) was measured under
either the pre-gravity-fix bug, the short-timing artifact, or without a
fair baseline to compare against -- this is the first comparison where
both sides are on equal footing. **Cite 13.3% -> 40.0% as the real,
trustworthy result of tonight's tuning, not any of the earlier
percentages in this file.**

**Still well short of the 90% target.** Given the object-specific
breakdown found earlier (some objects 100%, several 0%), the likely
highest-leverage next step is understanding/fixing those specific
geometry failures rather than further policy-parameter tuning -- squeeze
intensity and policy aggressiveness look close to their useful limit
(see the squeeze_intensity plateau and the fixed-step-made-no-difference
finding above). Not yet tested: the hysteresis/smoothing policy variant
flagged by the research agent, and whether the 0-failure objects fail for
a shared, fixable reason (e.g. grasp approach geometry) or are each
failing differently.