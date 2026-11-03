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

**Which number to cite for Model A, given the above (added
2026-10-30 to resolve an ambiguity this entry left open -- nothing new
run, just stating what already follows from the numbers above):**
0.7728929388214366 (the label-smoothed, 20-epoch run from the entry
above this section) remains the number currently in use for Model A,
since the num_epochs=40 change was reverted rather than adopted. But
unlike Model B's citation two sections up, this is NOT a settled number
-- it's one sample from a config now demonstrated to swing between 0.773
and 0.691 (a ~0.08 spread) run to run. Cite it as provisional / report
it as a range (0.69-0.77) rather than a single clean value, until seeded
multi-run training exists to actually determine mean+-std (see "Neural
network approach comparison.md" section 4.1 -- this is the user's
training-loop domain to fix, not something resolved here).

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
finding above).

## Hysteresis/smoothing policy variant -- tested, no improvement

- **Date:** 2026-10-25/26 overnight
- **Setup:** `diagnostic_hysteresis.py`, same 45 trials (15 objects x 3
  seeds, realistic timing, squeeze_intensity=0.95) as the clean
  comparison. Tests the research agent's flagged gap -- `decide_action()`
  reacts to one instantaneous probability with a flat threshold despite a
  300-step history buffer sitting unused -- by adding EMA smoothing
  (alpha=0.3) plus separate engage/disengage thresholds (0.3/0.15)
  instead of one flat 0.3 cutoff.
- **Current tuned policy (no smoothing): 18/45 (40.0%).**
- **Hysteresis (EMA alpha=0.3, engage=0.3/disengage=0.15): 17/45 (37.8%).**
- **No improvement -- a 1-trial difference out of 45, noise-level.**
  Smoothing/hysteresis is not the lever that moves this number. Combined
  with the squeeze_intensity plateau and the fixed-step-made-no-difference
  finding, three independent control-policy changes have now all failed
  to move the needle -- the bottleneck is very unlikely to be in
  `decide_action()`'s formula itself.

## Object-geometry investigation -- mesh collision is a real, large risk factor

- **Date:** 2026-10-25/26 overnight
- Cross-referencing the per-object breakdown (from the 45-trial realistic-
  timing batch, `Data/closed_loop_scenes/batch_results.jsonl`) against
  each object's `kind` in `OBJECT_LIBRARY` (`generate_trials.py`):
  - **Mesh-kind objects (cone, square_pyramid, triangular_prism,
    hexagonal_prism): 1/12 trials held securely (8.3%).**
  - **Primitive + compound objects (the other 11): 17/33 held securely
    (51.5%).**
  - That's a large, real split -- but not a clean rule. `plus_cross`
    (compound) and `rectangular_prism`/`cylindrical_prism` (primitive)
    also failed 0/3, so "mesh" is a strong risk factor, not the whole
    story. Also notable: object size (`char_length`) does NOT explain it
    -- `sphere`, `cone`, `cube`, `square_pyramid`, `hexagonal_prism` all
    share the identical char_length=0.04 (confirmed: their displacement
    threshold is exactly `0.15 * char_length` in every case), yet their
    success rates range from 3/3 (sphere) to 0/3 (cone, hexagonal_prism).
    Shape, not size, is driving this.
- **Follow-up diagnostic (`diagnostic_contact_geometry.py`, read-only
  inspection of `mjData.contact`, never writes to closed_loop_sim.py):**
  compared contact points at end-of-settle (grip closed, object only
  under gravity, before any pull) between sphere (succeeds) and cone
  (fails), same seed. Sphere: 6 contacts with the object, each at a
  distinct position. Cone: 12 contacts with the object -- but several are
  duplicated at near-identical positions (e.g. 3 contacts within 0.0001
  of each other at one spot, another pair/triple likewise) -- a known
  MuJoCo mesh-collision pattern where the solver can generate multiple
  near-redundant contact points on the same mesh face instead of one
  clean contact. This is *suggestive* that mesh collision is giving a
  less clean/stable contact geometry than analytic primitives -- but it's
  a single snapshot (n=1 per object), not a proven causal mechanism.
  Genuine shape difficulty (cone's apex and sloped sides give fingers a
  worse surface to cradle against even in a perfect simulator) remains an
  equally plausible, non-exclusive explanation -- this diagnostic can't
  cleanly separate the two hypotheses, only confirm mesh contact geometry
  looks messier.
- **Unrelated anomaly surfaced while running this diagnostic, flagged
  rather than chased further tonight:** the diagnostic rebuilt `cube`
  seed=0's scene independently and found it on the floor with ZERO
  finger contacts at end-of-settle -- but the actual 45-trial batch
  recorded that exact object/seed as held securely (displacement=0.0059).
  Same object, same seed, same build path, different outcome. Possible
  explanations not yet investigated: floating-point/solver
  non-associativity near an unstable equilibrium (cube's grip may be
  genuinely marginal, consistent with its mixed 1/3 batch result), or a
  procedural difference between this diagnostic's manual step loop and
  `run_trial()`'s. Worth a careful look before trusting any single-seed
  reproduction in this codebase -- the batch-level percentages (45 trials)
  are far more trustworthy than any one-off single-trial check.

**Honest state for whoever picks this up:** three independent
control-policy changes (squeeze sweep, fixed-step correction, hysteresis)
all failed to move the 40% number -- the policy is very likely near its
ceiling given the current model and object set. The object-geometry angle
is real (mesh objects fail far more) but only partially explains the
pattern, and the mechanism (sim artifact vs. genuine shape difficulty) is
still unresolved. Not yet tested: whether the 0-failure NON-mesh objects
(plus_cross, rectangular_prism, cylindrical_prism) fail for a shared
reason with each other, or with the mesh objects, or independently. 90%
by Oct 28 still looks genuinely difficult from here -- the honest
recommendation is to treat 40% (3x the real baseline) as tonight's
result, and decide fresh with the user which thread (geometry/mesh
collision, the model itself, or accepting a lower target with an honest
writeup of why) is worth the remaining time before the deadline.

## Per-finger localized correction -- tested, no improvement

- **Date:** 2026-10-30
- **Setup:** `diagnostic_per_finger_correction.py`, same 45-trial benchmark
  (15 objects x 3 seeds, realistic 500/300 ramp/settle timing,
  squeeze_intensity=0.95, Model A/22ch) as every prior closed-loop
  comparison above. Tests the research-agent-flagged gap (`decide_action()`
  applies the identical correction to every actuator from one pooled
  `slip_probability`, discarding the per-fingertip features the pipeline
  already computes) by weighting each finger's correction step by that
  finger's own current `tangential / |normal|` ratio relative to the
  5-finger mean (clipped to [0.5, 2.0]), while leaving the existing global
  gate (`slip_probability <= 0.3`) and `STEP_FRACTION=0.35` exactly as in
  the real, unmodified `decide_action()` -- isolating ONE variable, same
  discipline as the hysteresis/fixed-step/squeeze diagnostics.
- **CURRENT tuned policy (global scalar): 18/45 (40.0%).**
- **PER-FINGER localized correction: 18/45 (40.0%).**
- **Identical result, trial-for-trial -- not even noise-level, exactly
  the same count.** This is now the FOURTH independent control-policy
  change (after squeeze sweep, fixed-step, hysteresis) to fail to move the
  40% number. Taken together, these four results are strong, convergent
  evidence that the ceiling isn't in `decide_action()`'s formula in any
  form tried so far -- not the threshold, not the step size, not the
  timing/smoothing, and now not which finger it's applied to either. This
  further supports the decision (already made, see the RL env v2 scope
  expansion) that the real lever left is grasp FORMATION itself (what grip
  gets built in the first place, object-adaptively) rather than any more
  post-formation correction-policy tuning -- there is very likely nothing
  left to find by tuning `decide_action()` further.

## Cube seed=0 reproducibility anomaly -- investigated, not resolved

- **Date:** 2026-10-30
- **Re-ran `diagnostic_contact_geometry.py` for cube/seed=0 again, fresh:
  reproduces the SAME anomaly as before (ncon=1 object-contact, floor-only,
  no fingertip contact).** This rules out "it was a one-off fluke on the
  diagnostic's own side" -- the diagnostic's own result is itself
  deterministic/reproducible across separate invocations.
- **Compared `diagnostic_contact_geometry.py`'s `inspect_object()` loop
  against `run_trial()`'s ramp/settle code line-by-line -- found no
  procedural difference.** Same `cls.build_scene()` call (same
  `jitter_rng = random.Random(seed)`, so identical object pose/friction/
  size jitter), same gravcomp-toggle timing (`body_gravcomp` set to 0 at
  exactly `step == ramp_steps` in both), same `data.ctrl` assignment
  formula (`lo + frac*(target-lo)`) every step, same total step count
  through settle-end. `run_trial()` additionally zeroes `data.xfrc_applied`
  every step (pull=0.0 during ramp/settle) where the diagnostic never
  touches it -- inconsequential, since `MjData()` already zero-initializes
  it and nothing else writes to it before settle-end in either path.
  Neither script calls `mj_forward()` before the step loop; both rely on
  `mj_step()`'s own internal forward pass, so this isn't a difference
  between them either (it may be worth checking in general, but it's not
  what explains the cube/seed=0 split specifically).
- **Checked CLI defaults match.** Both this diagnostic (which hardcodes
  `args.squeeze_intensity = 0.95` in `main()`) and the batch run that
  produced the cited 40.0%/per-object numbers (`batch_closed_loop_trials.py`,
  called with `--squeeze-intensity 0.95` per the "realistic-timing batch"
  log entry above) land on the same ramp_steps=500/settle_steps=300
  defaults from the shared `add_common_args()` -- confirmed via
  `Data/closed_loop_scenes/batch_results.jsonl`'s cube/seed=0 record:
  `max_slip_probability=0.9076`, `n_corrections=333`, final
  `max_displacement=0.00588` (just under the 0.006 threshold) -- i.e. the
  real run shows a model that was VERY unsure/alarmed (90.76% slip
  probability) and corrected heavily (333 times) to just barely hold on,
  consistent with "marginal grip," not a confidently-secure one.
- **Still unresolved, and now better scoped:** the object ending up on the
  floor with zero finger contact at settle-end (diagnostic) is hard to
  reconcile with a grip that's merely "marginal but present" being rescued
  by 333 corrections during hold -- corrections nudge actuator targets by
  small increments; they have no mechanism to reach down and re-acquire an
  object that has already fallen out of the hand entirely. With the code
  paths verified identical and the CLI settings verified matching, the
  remaining candidate explanation is MuJoCo-engine-level
  non-reproducibility between separate process invocations building
  "the same" model from byte-identical XML (e.g. mesh/BVH construction or
  solver warm-start state that isn't purely a function of the XML content) --
  genuinely out of this investigation's bounded scope to chase further
  (would require instrumenting MuJoCo's own compilation/solver internals,
  not this project's code). Recorded here rather than resolved: treat any
  single-seed reproduction via a short standalone script in this codebase
  as suggestive, not proof, until/unless this is understood -- the
  batch-level percentages (45 trials, many seeds) remain the trustworthy
  numbers regardless.

## Mesh-collision vs. genuine-shape-difficulty -- one more step, still not separated

- **Date:** 2026-10-30
- **New data point:** counted actual vertices/faces in each mesh-kind
  object's `.obj` file (pure file parsing, no MuJoCo involved):
  cone=34v/64f, square_pyramid=6v/8f, triangular_prism=8v/12f,
  hexagonal_prism=14v/24f -- an ~8x spread in tessellation density
  across the 4 mesh objects (cone is a 32-sided "cone approximation"
  per `generate_trials.py`'s own comment; square_pyramid is a true
  4-sided pyramid, barely more faceted than a primitive box).
- **Cross-referenced against each object's actual per-object success
  rate** (`Data/closed_loop_scenes/batch_results.jsonl`, the same
  realistic-timing batch the per-object breakdown above is drawn from):
  cone 0/3, triangular_prism 0/3, hexagonal_prism 0/3, square_pyramid
  1/3. **No monotonic relationship between face count and failure rate**
  -- if contact-duplication severity scaled with tessellation density,
  cone (64 faces) should be the clear worst and square_pyramid (8 faces,
  barely different from a primitive) should look closest to a clean
  primitive's success rate; instead cone ties with the two
  medium-tessellation objects at 0/3, and square_pyramid's one extra
  success out of 3 trials is too small a sample to call a real effect
  either way. This weakens (does not refute) "worse tessellation ->
  more duplicate contacts -> more failure" as a clean causal story.
- **Stronger new evidence, from objects already in the existing
  per-object breakdown:** `rectangular_prism` and `cylindrical_prism`
  are PRIMITIVES -- exact analytic MuJoCo geometry, no mesh, no
  tessellation, none of the duplicate-contact-point artifact the cone
  vs. sphere comparison found is even possible for them -- yet both
  fail at 0/3, identical to the worst mesh objects. If mesh-collision
  artifacts were the main driver of the mesh-object failures, a clean
  analytic primitive sharing the same "flat-faced/angular" shape
  category should do noticeably better than a mesh object; it doesn't.
  **This is the most direct evidence yet that genuine shape difficulty
  (not a MuJoCo mesh-tessellation artifact) is doing most of the work**
  -- flat-faced, edged/cornered geometry seems to give the 5-fingered
  hand less to cradle against regardless of whether that geometry is
  represented as a mesh or an exact primitive.
- **Important complication, found while checking this:** `cylindrical_prism`
  (0/3) and `rod` (3/3) are BOTH primitive cylinders -- same curved-sided
  topology, same "flat circular end caps" -- yet opposite outcomes.
  The difference between them is size/aspect ratio (rod: r=0.008,
  half-length=0.045, long and thin; cylindrical_prism: r=0.02,
  half-length=0.03, short and fat), not flat-vs-curved shape category.
  So "flat/angular vs. curved" is not a clean rule on its own either --
  aspect ratio/size is a second, independent factor that can override it.
- **Honest conclusion, and where this investigation stops (bounded, per
  scope):** the balance of evidence shifted further toward genuine
  graspability difficulty (shape + aspect ratio) over simulation
  artifact as the dominant cause, mainly because exact analytic
  primitives fail at the same rate as the worst mesh objects. But this
  is a shift in balance of evidence, not a clean resolution -- the
  real pattern looks genuinely multi-factor (flat/angular geometry AND
  size/aspect ratio both matter, possibly interacting), and the original
  cone-vs-sphere duplicate-contact-point finding is still real, just
  probably more a symptom of an already-marginal grasp on a hard shape
  than an independent cause. Going further than this (e.g. swapping
  cone's mesh collision for a convex-hull-only primitive approximation
  to directly test grasp quality independent of tessellation) would
  require modifying how a mesh geom is declared/collided against in the
  scene-construction code, which is a real change to try next, not a
  read-only diagnostic -- out of this pass's bounded, read-only scope.

## Per-episode model rebuild in rl_env.py -- investigated, not changed

- **Date:** 2026-10-30
- **Question:** `SlipGraspEnv._build()` recompiles the full MuJoCo model
  from XML on every single `reset()`. Is there a bounded, safe way to
  cache/reuse the compiled model across episodes of the same object
  instead, varying only friction/mass/size/pose jitter without a full
  recompile?
- **Profiled `_build()` directly** (single process, alternating
  cube/cone/sphere/hexagonal_prism): 690-1234ms per call, roughly the
  SAME regardless of whether the object is a primitive or a mesh.
- **Isolated the cause: compiled the base hand scene ALONE, with no
  object at all, 3 times in the same warm process: 642-1491ms per
  compile** (44 bodies, 74 meshes, 35 joints -- the OrcaHand itself).
  This is the same order of magnitude as the full `_build()` numbers
  above. **The object's own jittered geometry is NOT the bottleneck --
  recompiling the hand's own fixed, never-changing 44-body/74-mesh
  structure from scratch every episode is.** The repeat-call numbers
  staying in the 640-900ms range (not dropping toward zero on a warm
  OS file cache) also rules out disk I/O as the dominant cost -- this
  is CPU-bound MuJoCo compile work (mesh/inertia/tree processing), not
  file reads, so `MjVFS`-style asset caching would not meaningfully
  help either.
- **Why this isn't a safe fix within scope, even though the bottleneck
  is now precisely identified:** MuJoCo's documented Python API has no
  partial/incremental recompile -- `MjModel.from_xml_string()` and
  `mjSpec.compile()` both rebuild the ENTIRE model (hand included) from
  scratch every call. The only way to actually skip paying the hand's
  recompile cost is to keep ONE already-compiled MjModel alive across
  resets and mutate ITS fields directly instead of rebuilding. That's
  safe for friction (`model.geom_friction`, a direct write, no derived
  quantities depend on it) and for initial pose (free-joint `qpos`, not
  baked into compilation at all) -- but NOT safe for the size/mass/
  mesh-scale jitter that's also part of every episode's randomization
  today: `geom_size` changes don't auto-update the `body_mass`/
  `body_inertia` MuJoCo derives from geometry at compile time (would
  need to hand-replicate that derivation per geom type), and mesh scale
  is baked into `model.mesh_vert` at compile time (would need to
  rescale vertex data AND rebuild the convex-hull/BVH data MuJoCo's own
  compiler computes from it). Either skipping those jitter dimensions
  (a real, silent change to the training distribution the policy sees
  -- it would stop seeing size/mass variation within a cached object's
  lifetime, which is the opposite of "same reward behavior") or
  hand-reimplementing MuJoCo's own compile-time geometry/inertia math
  (real risk of a subtly wrong, silently-corrupting bug, not a loud
  crash) is required to go further. **Conclusion: not attempting this.**
  The honest tradeoff is real engineering effort with real correctness
  risk for a speedup that, per the profiling above, would only remove
  the hand-recompile term -- worth revisiting with more time and a
  test harness that can detect subtle physics drift, not as a
  resource-capped side task right after stabilizing the crash fix.

## Production RL training run -- prepared command, not executed

- **Date:** 2026-10-30
- **Recommended launch** (run from `Synthetic data creation/Simulation/`):
  ```
  python train_rl_policy.py \
      --mjcf ../../third_party/orcahand_description/v2/scene_slip.xml \
      --fingertips right_T-DP_b7429e50 right_I-FingerTipAssembly_ec49c16c \
                   right_M-FingerTipAssembly_34afb748 right_M-FingerTipAssembly_424a8e75 \
                   right_P-FingerTipAssembly_cd219176 \
      --actuators right_t-mcp_actuator right_t-pip_actuator \
                  right_i-mcp_actuator right_i-pip_actuator \
                  right_m-mcp_actuator right_m-pip_actuator \
                  right_r-mcp_actuator right_r-pip_actuator \
                  right_p-mcp_actuator right_p-pip_actuator \
      --mesh-dir ../../assets --out-dir ../../Data/closed_loop_scenes \
      --checkpoint ../../src/training/model_a.pt --in-channels 22 \
      --n-envs 6 --total-timesteps 300000 --eval-freq 10000 \
      --save-path ../../Data/ppo_slip_policy --log-dir ../../Data/rl_logs_production
  ```
- **n-envs=6:** this machine has 8 logical cores; 6 workers + 1 main
  process (which also hosts the eval `DummyVecEnv`) leaves headroom for
  the OS rather than saturating all 8. This matches `train_rl_policy.py`'s
  own pre-existing default, not a new number invented here.
  **Caveat, stated honestly:** the crash fix was verified clean at
  `--n-envs 2` across two runs (10,240 and 48,640 steps, zero crashes) --
  it was NOT re-tested at n-envs 6-8 in this session, because doing so
  would exceed this session's own 2-3-concurrent-process resource cap.
  The original crashes happened specifically under 8-worker configs, so
  this gap is real, not hypothetical. The reasoning for expecting it to
  hold anyway: the fix removes the disk re-read race entirely
  (`MjModel.from_xml_string()` instead of `from_xml_path()`), and that
  race's root cause (suspected AV scanning of rapidly-written scene
  files) doesn't have an obvious reason to be worker-count-sensitive in
  a way the fix wouldn't also cover -- but this is reasoning from
  mechanism, not a direct empirical re-test at 6-8 workers. **If
  resource/time allows, re-verifying once at n-envs=6 for ~15-20k steps
  before committing to the full 300k run would close this gap cheaply.**
  If not, watching the first several minutes of the real run for the
  specific `Could not allocate memory` error is the fallback check.
- **total-timesteps=300000:** this is the script's existing default, not
  a new recommendation -- kept as the starting point given the deadline.
  Honest caveat from the 50k-step check: training reward (on train
  objects) was still climbing at the point it was stopped (~250-260 and
  rising), while holdout-eval reward was FLAT between the 20k and 40k
  checkpoints (~323 both times) -- 300k may or may not be enough for the
  policy to show real held-out generalization, not just training-object
  improvement. `train_rl_policy.py` has no resume-from-checkpoint
  support today, so if 300k turns out insufficient, continuing training
  would currently mean starting over, not extending -- worth knowing
  before committing to a number under deadline pressure, not something
  fixed here.
- **Wall-clock estimate, extrapolated from observed throughput (not a
  new measurement at n-envs=6):** the 10k and 50k checks both ran at
  ~51-55 combined env-steps/sec at n-envs=2. Scaling assumptions for
  n-envs=6 range from near-linear (~3x -> ~165 steps/sec -> 300,000/165
  [~]30 min) to conservative (~2x after 8-core contention -> ~110
  steps/sec -> 300,000/110 [~]45 min). **Best honest estimate: roughly
  30-45 minutes wall-clock**, not a single precise number, given no
  direct n-envs=6 measurement exists yet.
- **Not executed**, per instructions -- this is the exact command to run
  once the user gives the go-ahead.

## Production run attempt, n-envs=6 -- crashed immediately, cause unknown at the time

- **Date:** 2026-10-30
- **Command:** exact launch from the entry above, `--n-envs 6
  --total-timesteps 300000 --eval-freq 10000`, user-approved go-ahead.
- **Result:** crashed within the first rollout -- no training-progress
  table ever printed. Exit code 1.
  `BrokenPipeError: [WinError 109] The pipe has been ended` ->
  `EOFError` in the main process's `SubprocVecEnv.step_wait()`
  (`results = [remote.recv() for remote in self.remotes]`), meaning one
  of the 6 worker subprocesses died and its pipe closed. **The dying
  worker's own root-cause exception did not reach this log at all** --
  only this downstream symptom did. No checkpoint was saved. Stopped and
  reported rather than retried or silently falling back to n-envs=2, per
  instructions -- see the next two entries for the actual cause, found
  via instrumentation added afterward.

## Root-cause investigation: MjModel.from_xml_string() fails permanently after ~90-100 repeated calls in one long-lived process

- **Date:** 2026-10-30
- **Platform (exact, for reproduction):** Windows 11 Home, OS version
  10.0.26200, build 26200, 64-bit (`Get-CimInstance Win32_OperatingSystem`).
  Python 3.14.3. **`mujoco` 3.11.0** (`python -c "import mujoco; print(mujoco.__version__)"`).
  `stable-baselines3` 2.9.2.
- **Why the previous entry's crash was a black box:** on Windows,
  `multiprocessing`'s spawn-mode child processes don't reliably funnel
  their own stdout/stderr into the parent's redirected stream, so a
  worker's own traceback can vanish even though the parent sees the
  pipe break. To get real visibility, `rl_env.py` and `train_rl_policy.py`
  were instrumented (temporarily, for this investigation) with: (1)
  `faulthandler.enable()` per worker process, writing to a per-worker
  file, which would catch a native-level crash (segfault/access
  violation) that no Python `try/except` can catch; (2) print-and-flush
  checkpoints at every stage of scene construction/model compilation/
  reset, written to `Data/_worker_diag_<idx>.log` by opening, writing,
  and closing the file on every single call (not buffering) specifically
  so the last checkpoint before a hard process death is never lost.
- **Quick check first: is this a native crash?** Checked Windows'
  Application event log (`Get-WinEvent -FilterHashtable @{LogName='Application'; ...}`)
  for the exact timestamp window of the n-envs=6 crash above (18:46:00-18:50:00
  on 2026-10-30) -- **no matching events at all**, not even an unrelated
  one. Windows Error Reporting normally logs an "Application Error"
  event for a true segfault/access violation; its total absence is
  evidence against (not proof against) a native crash for that specific
  failure.
- **Retried at an intermediate worker count, n-envs=4, same command
  otherwise, with the instrumentation above in place.** Result: crashed
  again, but this time it got to **80,896 total timesteps** (vs. dying
  before the first rollout at n-envs=6) before dying the same way --
  `BrokenPipeError`/`EOFError` in the main process. **This time the
  per-worker diagnostic log for the dying worker (worker index 2)
  captured the actual root-cause exception directly, for the first time
  in this investigation:**
  ```
  19:16:24 [worker 2] reset attempt 0: object=capsule seed=152468801, about to call _build
  19:16:24 [worker 2] about to acquire build_lock
  19:16:24 [worker 2] lock acquired, compiling model
  19:16:24 [worker 2] _build FAILED on attempt 0: ValueError: Error: engine error: Could not allocate memory
  19:16:24 [worker 2] reset attempt 1: object=l_block seed=1755466623, about to call _build
  19:16:24 [worker 2] lock acquired, compiling model
  19:16:25 [worker 2] _build FAILED on attempt 1: ValueError: Error: engine error: Could not allocate memory
  19:16:25 [worker 2] reset attempt 2: object=capsule seed=477492862, about to call _build
  19:16:25 [worker 2] lock acquired, compiling model
  19:16:25 [worker 2] _build FAILED on attempt 2: ValueError: Error: engine error: Could not allocate memory
  19:16:25 [worker 2] reset attempt 3: object=square_pyramid seed=377189855, about to call _build
  19:16:25 [worker 2] lock acquired, compiling model
  19:16:26 [worker 2] _build FAILED on attempt 3: ValueError: Error: engine error: Could not allocate memory
  19:16:26 [worker 2] reset attempt 4: object=ellipsoid seed=437705471, about to call _build
  19:16:26 [worker 2] lock acquired, compiling model
  19:16:26 [worker 2] _build FAILED on attempt 4: ValueError: Error: engine error: Could not allocate memory
  19:16:26 [worker 2] all 5 reset attempts failed, re-raising
  ```
  This `ValueError` then propagates up out of `reset()`, killing worker
  2's process, which is what produces the `BrokenPipeError: [WinError 109]
  The pipe has been ended` -> `EOFError` seen in the main process's
  `subproc_vec_env.py::step_wait()` (`results = [remote.recv() for remote
  in self.remotes]`) -- the exact same downstream symptom as the n-envs=6
  crash above, now traced to its real cause.
- **The exact counts, which are the key piece of evidence:** worker 2's
  diagnostic log shows **91 successful resets** (`grep -c "_build
  succeeded" Data/_worker_diag_2.log` = 91) from process start
  (18:55:03) up through its last successful reset (completed 19:16:19)
  before the fatal attempt (started 19:16:24, a normal ~5-second gap
  from the prior episode ending). At that exact same wall-clock moment,
  the other 3 workers' logs show **97, 100, and 103** successful resets
  respectively (`Data/_worker_diag_0.log`, `_worker_diag_1.log`,
  `_worker_diag_3.log`). All four counts cluster tightly in the
  90s-to-low-100s range -- not wildly different, not random. This
  clustering, not any single worker's number alone, is the actual
  evidence: it means every worker was approaching the same threshold at
  roughly the same rate, and worker 2 simply crossed it first.
- **What this rules out, and the specific evidence for each:**
  - **(a) Real-time antivirus scanning / a disk-read race (the original
    hypothesis behind the `from_xml_string()` fix) -- ruled out.**
    `from_xml_string()` never touches disk for the compile step itself
    (only for the one-time `open(scene_path).read()` immediately before
    it, which is NOT where the failure occurs -- the diagnostic log
    shows the failure happening at "lock acquired, compiling model",
    after the file read already succeeded), and `build_lock` serializes
    every single compile across every worker, so there is no window for
    simultaneous disk access at all.
  - **(b) A native-level crash (segfault/access violation) -- ruled
    out for THIS run.** This was an ordinary, catchable Python
    `ValueError` raised by MuJoCo's own Python bindings, not a silent
    process death -- `faulthandler` never fired (no entry in any
    worker's log from it), and the process only died because the
    exception propagated up uncaught through `reset()`'s own retry loop
    after exhausting 5 attempts. Combined with the Application-log check
    above (no crash record for the earlier n-envs=6 failure either),
    there is no evidence of a native crash in either incident.
  - **(c) Random/intermittent per-call resource contention -- weakened,
    not fully ruled out, by the clustering.** A purely random failure
    mode (e.g. a one-off transient OS hiccup) would not be expected to
    hit all 4 workers within a similar ~90-105-call band of each other;
    a consistent accumulating threshold explains the clustering far
    better than chance.
- **Working hypothesis (stated as a hypothesis, not a proven mechanism):**
  something inside MuJoCo's native model-compiler is not being fully
  released between repeated `MjModel.from_xml_string()` calls within the
  same long-lived process. Each individual call is harmless and fast
  (~700-1500ms per this session's own earlier profiling), but something
  -- most plausibly an internal allocator/arena inside the compiled
  MuJoCo extension, possibly tied to mesh/asset processing given this
  scene's 74 meshes -- accumulates across calls regardless of which
  object or seed is being compiled, until a process-level limit is hit,
  at which point EVERY subsequent call in that process fails permanently
  (not intermittently -- all 5 retries with 5 different scenes failed
  identically, which a transient issue would not do).
- **What is genuinely NOT known, stated honestly:**
  - Whether this is specific to `from_xml_string()` or would also affect
    `from_xml_path()` -- not tested here, since the fix already moved
    away from the latter for an unrelated reason.
  - Whether this is Windows-specific or would reproduce on Linux/macOS.
  - Whether this is specific to mujoco 3.11.0 or affects other versions.
  - Whether explicit `del model` / `del data` / `gc.collect()` calls
    between episodes would mitigate or delay it -- untested; plausible
    if this is a Python-side reference-retention issue delaying native
    deallocation, but equally plausible to do nothing if the leak is
    entirely inside MuJoCo's own C/C++ allocator and not reachable from
    Python's garbage collector at all.
  - The exact internal mechanism -- nothing above identifies the actual
    leaking resource, only its observable behavior.
- **Practical workaround identified (not a fix):** recycling the worker
  process, or otherwise resuming/segmenting training in chunks well
  under the ~90-call-per-worker threshold observed here, avoids
  triggering it. This is a workaround for the symptom, not a resolution
  of the underlying cause.
- **Prior art check:** searched MuJoCo's GitHub issues and documentation
  on 2026-10-30 for this exact pattern ("MjModel.from_xml_string repeated
  calls in one process eventually fails with Could not allocate memory")
  -- **found no existing report matching it.** If this holds up under
  further testing, it is worth filing as a real upstream issue against
  `google-deepmind/mujoco` -- not done here, since filing a public
  external issue is the repo owner's call, not something to do
  unilaterally.
- **Positive result alongside the crash, worth recording:** holdout-eval
  reward (`EvalCallback`, deterministic, on `HOLDOUT_OBJECTS`) improved
  substantially over this run before it died -- 229.14596 at the 40,000-
  step checkpoint, 524.069416 at the 80,000-step checkpoint
  (`Data/rl_logs_production/evaluations.npz`) -- real generalization
  improvement, not just training-object reward climbing, unlike the
  earlier n-envs=2/50k-step check where holdout-eval reward was flat.
  Checkpoints at 40,000 and 80,000 steps plus `best_model.zip` are saved
  at `Data/rl_logs_production/` and are usable artifacts despite the crash.
- **Not done:** no retry, no fallback to n-envs=2, no attempt to fix the
  underlying leak, no GitHub issue filed. The temporary diagnostic
  instrumentation (`faulthandler`, the per-worker print-and-flush
  checkpoints, `_diag_log()` in `rl_env.py`) is left in place in the
  working tree rather than reverted, since it is almost certainly still
  needed for whatever investigation comes next.
## Failure-mode breakdown of the 27 failed trials in the 40.0% clean comparison

- **Date:** 2026-10-30
- **Why:** a research suggestion proposed that some "failures" might
  actually be late-but-successful catches -- the controller stops the
  slip, but the object had already crossed the displacement threshold
  moments earlier, so a genuinely near-miss trial scores identically to
  a total grip failure. Cheap to check: `Data/closed_loop_scenes/
  batch_results.jsonl` already has `max_displacement`/`threshold`/
  `n_corrections`/`max_slip_probability` per trial from the existing
  45-trial tuned-policy batch (confirmed 18/45 = 40.0% held securely,
  matching "THE clean comparison" entry above) -- no new simulation
  needed, just analysis of data already on disk.
- **Result: the near-miss theory does NOT hold here.** Of the 27 failed
  trials, only 3 (11%) overshot the threshold by <=1.5x. Median overshoot
  was 6.34x; 15 of 27 (56%) overshot by more than 3x, several by
  20-33x (`ellipsoid`, `cube` seed 1/2, `cylindrical_prism`, `plus_cross`
  all exceeded displacement 20x their threshold). Most failures are not
  close calls -- the object moves far past the line, not barely past it.
- **But splitting failures by `n_corrections` and `max_slip_probability`
  reveals two distinct failure modes that the single "slipped" boolean
  was hiding:**
  - **Detection misses (6 of 27 failures, the lowest-overshoot group,
    1.04x-2.57x):** `max_slip_probability` stayed low (0.20-0.40) and
    `n_corrections` was near zero -- `hexagonal_prism` seed 0 and
    `square_pyramid` seed 1 triggered **zero** corrections for the
    entire trial. The object slipped and the controller essentially
    never tried, because the model's own predicted probability never
    crossed `decide_action()`'s 0.3 trigger threshold strongly enough.
    This is a model-sensitivity problem, not a control-policy problem.
  - **Saturation despite correct detection (21 of 27 failures, nearly
    all the moderate-to-catastrophic ones):** `max_slip_probability`
    0.92-1.0 (the model was confident and correct) and `n_corrections`
    mostly 250-330+ out of the trial's step budget -- the controller is
    firing a correction almost every single monitored step -- yet
    displacement still runs out to 7x-33x past threshold anyway. The
    model worked; the correction had nowhere left to go. This matches
    the actuator-saturation hypothesis flagged on 2026-10-25
    ("`decide_action()`'s own formula... shrinks toward zero as the
    grip approaches its ctrlrange max") but never directly confirmed
    until this breakdown.
- **Why this matters:** it gives a concrete explanation for why all four
  independently-tested control-policy changes (squeeze-intensity sweep,
  fixed-step correction, hysteresis, per-finger localization) failed to
  move the 40.0% number. Every one of them only changes the correction
  *formula*. The detection-miss group never calls the formula in the
  first place (gated out below the trigger threshold); the saturation
  group calls it almost continuously with no physical actuator headroom
  left for it to act on. Neither failure mode is reachable by tuning how
  the correction itself is computed -- which is exactly the class of fix
  all four prior attempts were.
- **Not yet done (at the time this entry was first written):** this
  analysis inferred saturation circumstantially (high trigger rate +
  high confidence + failure anyway) from existing summary statistics --
  it did not directly inspect per-step `ctrl` values against `ctrlrange`
  to confirm actuators were actually pinned at their limit. See the
  entry directly below, which closes this gap. The detection-miss
  group's root cause is addressed in the entry after that.

## Actuator saturation -- directly confirmed, no longer circumstantial

- **Date:** 2026-10-30
- **Method:** added an optional `step_callback(step, phase, driven, data,
  disp)` parameter to `closed_loop_sim.py`'s `run_trial()` (default
  `None`, zero behavior change for every other caller -- called once per
  simulation step, right after `mj_step`, with the live `driven` dict of
  `{aid: [lo, hi, target]}` and the actual `data.ctrl[aid]`). New
  `diagnostic_actuator_saturation.py` uses it to record, for every driven
  actuator at every phase-2 (monitoring/pull) step, the fraction of its
  ctrlrange closed: `(ctrl - lo) / (hi - lo)`, where 1.0 means pinned at
  the physical maximum. Tested on 7 trials from the saturation group
  identified in the entry above: `cube` seed 1/2, `ellipsoid` seed 0/2,
  `cylindrical_prism` seed 0/1/2.
- **Result: confirmed directly, for all 7 trials, with no exceptions.**
  By the end of phase 2, every single trial shows mean AND max fraction-
  closed across all driven actuators = **exactly 1.0000** -- every
  actuator is pinned at its exact ctrlrange maximum when the trial ends.
  Across the whole phase-2 window (not just the end), the fraction of
  steps where ALL driven actuators were simultaneously within 1% of
  their max ("pinned") ranged from 30.5% (`cylindrical_prism` seed=1) to
  88.2% (`cube` seed=1, `cylindrical_prism` seed=2), and the mean
  fraction-closed averaged across the whole window was 0.976-0.996 for
  every trial -- i.e. actuators spend the large majority of the
  monitoring phase already very close to their limit, not just at the
  very end:
  ```
  cube               seed=1  pinned 88.2% of phase 2, mean_frac_closed_avg=0.9944, ends at 1.0000/1.0000
  cube               seed=2  pinned 87.6% of phase 2, mean_frac_closed_avg=0.9942, ends at 1.0000/1.0000
  ellipsoid          seed=0  pinned 83.2% of phase 2, mean_frac_closed_avg=0.9921, ends at 1.0000/1.0000
  ellipsoid          seed=2  pinned 62.3% of phase 2, mean_frac_closed_avg=0.9816, ends at 1.0000/1.0000
  cylindrical_prism  seed=0  pinned 84.4% of phase 2, mean_frac_closed_avg=0.9929, ends at 1.0000/1.0000
  cylindrical_prism  seed=1  pinned 30.5% of phase 2, mean_frac_closed_avg=0.9760, ends at 1.0000/1.0000
  cylindrical_prism  seed=2  pinned 88.2% of phase 2, mean_frac_closed_avg=0.9959, ends at 1.0000/1.0000
  ```
- **This is no longer a hypothesis.** The actuator-saturation explanation
  for why all four independently-tested control-policy changes (squeeze
  sweep, fixed-step, hysteresis, per-finger localization) failed to move
  40.0% is now directly, empirically confirmed for the trials tested --
  by the time displacement is running away, the grip has no physical
  room left to close further, regardless of what `decide_action()`'s
  formula computes. Any further fix for this specific failure mode has
  to either (a) start with a tighter/different squeeze so there's
  headroom left in reserve for phase 2 (already tested as a global
  sweep and plateaued, per the squeeze_intensity entry above -- would
  need a smarter per-object or adaptive-formation approach, not a global
  constant), or (b) resist the pull some other way than closing fingers
  further (e.g. a fundamentally different grasp strategy), which is
  exactly the motivation already behind the RL formation+correction
  scope expansion.

## Detection-miss root cause -- strongly correlated with mesh-kind geometry, not fully disentangled from slip severity

- **Date:** 2026-10-30
- **Method:** cross-referenced every trial's `object_kind` (`primitive`/
  `mesh`/`compound`, from `OBJECT_LIBRARY` in `generate_trials.py`)
  against the detection-miss (`max_slip_probability<0.5`) vs. saturation
  (`>=0.9`) classification from the failure-mode-breakdown entry above.
  No new simulation -- pure analysis of `Data/closed_loop_scenes/
  batch_results.jsonl`, already on disk.
- **Result: a strong, nearly bimodal split by object kind.**
  ```
  kind         total  failed  detection_miss  saturation  ambiguous
  primitive    24     12      0               11          1
  mesh         12     11      9               1           1
  compound     9      4       1               3           0
  ```
  **Primitives essentially never produce a detection miss** (0 of 12
  primitive failures) -- when a primitive object fails, the model is
  confident and the controller tries hard anyway (saturation). **Mesh
  objects are the opposite**: 9 of 11 mesh failures (82%) are detection
  misses. The one compound-object detection miss is `l_block` seed=0;
  `l_block` seeds 1/2 both held securely, so `l_block` is not a
  systematically weak-signal object overall -- seed 0 looks like an
  individual borderline case, not a pattern, and is not chased further
  here.
- **A real, carefully-checked confound: detection-miss trials also have
  much lower overshoot** (1.04x-3.17x, i.e. marginal/slow slips) than
  saturation-group trials (mostly 7x-33x, i.e. severe/fast slips) --
  raising the alternative explanation that weak signal might just
  reflect a slower-developing, inherently harder-to-detect physical
  event, independent of mesh vs. primitive geometry.
- **This confound is at least partially disentangled by a direct natural
  comparison: `flat_disk` (primitive) seed=1 vs. `square_pyramid` (mesh)
  seed=1/2.** `flat_disk` seed=1 has an almost identically marginal
  overshoot (1.04x) to `square_pyramid` seed=2 (1.42x) -- yet `flat_disk`
  seed=1 produced `max_slip_probability=0.934` (confident, correctly
  detected) while `square_pyramid` seed=1/2 produced 0.249/0.362 (weak,
  missed), despite seed=1's overshoot (2.57x) being LARGER than
  `flat_disk`'s. If marginal-overshoot-alone explained weak signal,
  `flat_disk` seed=1 should have been just as weak as `square_pyramid`'s
  -- it wasn't. **This favors mesh-kind geometry itself as the stronger
  driver**, not just "these happen to be slow/marginal slips," though it
  does not rule out slip-severity as a contributing secondary factor
  (the two are still correlated in this dataset, just not perfectly).
- **Plausible causal chain (stated as a hypothesis):** the already-
  established mesh-collision contact-point messiness (cone vs. sphere,
  "Object-geometry investigation" entry above -- duplicated/near-
  identical contact points on mesh objects) produces atypical per-step
  tactile readings during the actual slip event, not just at the
  settle-end snapshot originally checked -- readings that don't resemble
  the training distribution's typical "clearly slipping" pattern closely
  enough for the model to respond with high confidence. This would mean
  the SAME mesh-collision artifact already implicated in grip-security
  failure is ALSO degrading the slip-detection model's input quality for
  these objects -- two symptoms of one underlying cause, not two
  separate problems.
- **What this does NOT resolve** this is a correlation
  from existing summary statistics plus one disentangling comparison,
  not a mechanistic confirmation -- it does not inspect the actual raw
  per-step tactile traces for a mesh-object slip event to directly show
  an atypical signature (the kind of thing `diagnostic_visualize_onset.py`
  already does for slip-vs-stable in general, but not yet re-run split by
  object_kind specifically). That would be the natural next step if this
  thread is picked up again, but is not done here -- bounded per scope.

## Independent corroboration: mesh objects are NOT physically harder to hold

- **Date:** 2026-10-30
- **Why:** re-examined `Data/manifest_full_merged.json` -- the full
  20,000-trial OPEN-LOOP dataset (9 generation batches, no corrective
  control, `generate_trials.py`'s scripted squeeze/pull schedule) that
  the slip-detection model was actually trained on, not the 45-trial
  closed-loop batch every prior object-geometry finding was based on.
  17,514 trials had contact. This is a much larger, independent data
  source for the same question: are mesh objects genuinely harder to
  physically hold, or is that specific to the closed-loop setup?
- **At the full dataset's broad condition range (squeeze_intensity
  0.5-1.0, pull_multiple 0.5-3x, both randomized per trial), mesh
  objects are NOT uniquely bad.** Per-object slip rate: cone 0.642,
  hexagonal_prism 0.773, square_pyramid 0.695, triangular_prism 0.706 --
  but primitive `cylindrical_prism` is 0.822 and `ellipsoid` is 0.724,
  both equal to or worse than every mesh object. `n_fingers_contacted`
  shows no mesh-specific deficit either (mesh objects average 3.46-3.62
  fingers touching, squarely inside the same 3.1-3.8 range as
  primitives/compounds).
- **Narrowed to the closed-loop batch's actual regime (squeeze>=0.9,
  pull in [1.2, 1.8x], n=475 trials, 15-39 per object -- small enough
  that individual percentages carry real sampling noise, treat as
  directional):** mesh objects average 57.2% slip vs. 40.4% for
  non-mesh -- a real gap, but far smaller than the near-0%-vs-high-
  success split seen in the actual closed-loop batch. And the ranking
  is not clean: `cylindrical_prism` (primitive) has the HIGHEST slip
  rate of all 15 objects here (96.7%, n=30) -- worse than any mesh
  object -- while `triangular_prism` (mesh) is comparatively low
  (33.3%, n=27). Finger-contact count again shows no mesh-specific
  pattern (mesh objects 4.52-4.78, same range as everything else,
  several primitives at exactly 5.0).
- **What this means, combined with the detection-miss finding directly
  above:** mesh objects do NOT have a clearly, consistently higher
  open-loop physical slip propensity than several primitives -- so
  "genuine shape difficulty" as a blanket explanation for the closed-
  loop batch's near-total mesh failure is not well supported by this
  independent, much larger dataset. What mesh objects DO have, per the
  entry directly above, is a strong, disentangled concentration of
  DETECTION misses (9/11 mesh closed-loop failures vs. 0/12 primitive
  failures). Put together: the likely mechanism is signal quality, not
  physical difficulty -- mesh contact geometry (the duplicated/near-
  identical contact points already found in the earlier contact-
  geometry diagnostic) probably degrades the tactile *reading* without
  the object being any more prone to actually slipping. This sharpens
  rather than contradicts the detection-miss entry -- it rules out the
  competing "mesh objects are just physically harder" explanation using
  a 20,000-trial dataset the earlier finding never had access to.
- **Caveat, stated honestly:** the open-loop dataset and the closed-
  loop batch are different experiments (no corrective control vs. active
  correction; a wide randomized squeeze/pull range vs. one fixed
  condition) -- this is the best available proxy for "is this object
  physically harder," not a direct replication. The narrowed subset's
  small per-object n (15-39) means its specific percentages, especially
  the 0.0% rates for `dumbbell`/`flat_disk`, should be treated as
  directional, not precise.

## Resumed-training segment 1 crash -- likely self-inflicted confound, not confirmed as the same leak

- **Date:** 2026-10-30
- **What happened:** after `--resume-from` was verified clean (80,000 ->
  95,360 steps, reward continuity confirmed), segment 1 of the full
  remaining run (resuming from 95,360 toward a 145,000 target) crashed
  with the same `BrokenPipeError`/`EOFError` symptom in the main process
  at total_timesteps=112,768 (per the orchestrator log) -- a worker died.
- **This does NOT look like the same leak, and the likely cause is
  self-inflicted.** Two concrete differences from the previously-confirmed
  leak pattern: (1) **reset counts were far lower this time** -- the
  dying worker (worker 2) had completed only 14 successful resets
  (workers 0/1/3 were at 15/14/13), nowhere near the ~90-103 range where
  the leak previously manifested; (2) **the failure signature itself is
  different** -- worker 2's diagnostic log shows it got stuck exactly at
  `"lock acquired, compiling model"` at 22:25:48 with NO further log line
  at all (no "model compiled", no "FAILED on attempt N") -- a silent
  hang/death mid-call, not the clean, logged `ValueError: Could not
  allocate memory` exception seen in the confirmed-leak incident.
- **The suspicious coincidence:** at essentially the same wall-clock
  moment (22:25:48), a separate diagnostic script
  (`diagnostic_lower_threshold.py`, run concurrently with training on
  the theory that a single-process CPU diagnostic wouldn't meaningfully
  compete for resources) was launched and itself failed with `OSError:
  [WinError 1114] A dynamic link library (DLL) initialization routine
  failed. Error loading "...torch\lib\nvperf_host.dll"` -- a CUDA/GPU-
  driver-level DLL init failure. That script calls
  `closed_loop_sim.load_model()` with no explicit `device=`, which
  auto-detects and attempts to initialize CUDA -- unlike every training
  worker, which explicitly forces `device="cpu"\` specifically to avoid
  multi-process CUDA-context contention (see `load_model()`'s own
  docstring, which already documents a DIFFERENT real torch/CUDA
  inconsistency found earlier this project). It is plausible that this
  second process's CUDA initialization attempt, running at the same
  moment as 4 other worker processes, disrupted something at the
  GPU-driver or OS-loader level that caused worker 2's (CPU-only,
  CUDA-unrelated) MuJoCo compile call to hang or die -- but this is
  **not confirmed, only strongly correlated by timing.**
- **What this means, stated honestly:** this incident should NOT be
  counted as new evidence about the ~90-100-reset leak's behavior (it
  happened at a much lower reset count, under what looks like a
  different, confounded condition). It also should not be treated as
  disproving the leak either -- it's simply inconclusive on that
  question. The actionable lesson is methodological: **do not run any
  torch/CUDA-touching script concurrently with the multi-worker CPU-only
  training**, even though the training workers themselves never touch
  CUDA -- the mere attempt by an unrelated process to initialize CUDA on
  the same machine may be able to disrupt it. Fixed going forward by
  having `diagnostic_lower_threshold.py` force `device="cpu"` explicitly
  (same as every training worker) rather than auto-detecting, so it's
  actually safe to run concurrently in the future. Segment 1 was
  relaunched solo (no concurrent diagnostic) to get an unconfounded
  read -- see the next entry for that result.

## Resumed-training segment 1, solo retry -- a genuine SIGSEGV, not the confound, and not the ~90-reset pattern either

- **Date:** 2026-10-30
- **What happened:** segment 1 was relaunched with no concurrent process
  running at all (ruling out the suspected confound above). It crashed
  again, but this time with **exit code 139 (SIGSEGV)** -- a real,
  hard native segmentation fault in the training process itself, not a
  catchable Python exception. `faulthandler` was active but did not
  print anything to any worker's diagnostic log before the crash
  (consistent with how SIGSEGV handling can itself be unreliable once
  memory is already corrupted -- its absence here is not strong
  evidence either way).
- **Reset counts at time of death, for all 4 train workers:** 34, 31,
  28, 35 (`grep -c "_build succeeded"` on each `_worker_diag_N.log`).
  **This is far below the ~90-103 range from the original confirmed-leak
  incident** -- training reached total_timesteps=135,296 before dying,
  only ~39,936 steps into this segment.
- **The eval worker (the `DummyVecEnv` holdout-object evaluator, which
  runs IN THE MAIN TRAINING PROCESS, not a separate `SubprocVecEnv`
  subprocess) was stuck exactly at `"lock acquired, compiling model"`
  with zero further log output** -- its diagnostic log for this fresh
  segment shows 0 completed resets, meaning its very first reset
  attempt in this process was the one in progress when the whole
  process segfaulted. **This is the same exact symptom (stuck mid-call
  inside the locked `MjModel.from_xml_string()`, no further output) as
  the confounded crash in the entry above** -- but this time on the
  eval worker specifically, with no concurrent CUDA-touching process to
  blame, and at a much lower reset count.
- **This changes the working hypothesis from the "Root-cause
  investigation" entry.** That entry characterized the failure as a
  cumulative leak that fails reliably after ~90-100 compiles in the
  same process. Two more data points now exist, both failing at far
  lower counts (14 and 28-35) with different symptoms (a silent
  stuck-mid-call hang both times, vs. the original's clean, logged
  `ValueError` after ~91-103 calls). **The more accurate characterization
  now is: `MjModel.from_xml_string()`, even serialized behind
  `build_lock`, has some non-negligible per-call failure probability
  that is NOT simply a function of how many times it's been called
  in-process -- the ~90-103 number from the first incident looks like
  where the dice happened to land that particular run, not a reliable
  safe threshold.** This also means staying under some fixed reset
  count per segment is NOT a guaranteed mitigation for this failure
  mode, even though it's still probably true that failure probability
  increases somehow with call count (not yet disproven, just no longer
  assumed as the sole or dominant factor).
- **What this does NOT change:** the practical workaround (segmenting
  training across fresh processes, resuming from the last good
  checkpoint) still works AS A RECOVERY MECHANISM regardless of why a
  given process dies, since each segment independently checkpoints
  progress. It just means segment length is a risk-reduction knob, not
  a guarantee, and this failure mode can and does recur at a real,
  non-trivial rate even in short, otherwise-healthy segments.
- **Genuinely good news buried in this crash:** `CheckpointCallback`'s
  periodic save captured `Data/rl_logs_production/ppo_slip_ckpt_135360_steps.zip`
  -- confirmed intact and loadable (`PPO.load(...)` ->
  `num_timesteps=135360`) -- meaning this segment's progress was
  effectively NOT lost despite the crash, even though the explicit
  `model.save(args.save_path)` call at the end of `main()` never ran
  (that only fires on clean completion). `Data/ppo_slip_policy_latest.zip`
  itself is still only at 95,360 (the last clean segment boundary) and
  is now stale -- the 135,360 checkpoint is the one to actually resume
  from next, once a decision is made on how to proceed.
- **Stopping here to report rather than retrying a third time on the
  same segment, per standing instructions on unexpected failures.**
  Two different crash types on the same target in a row (a possibly-
  confounded `EOFError`/`BrokenPipeError`, then an unconfounded SIGSEGV)
  is exactly the kind of result that calls for a decision, not another
  blind retry -- open questions worth a real decision before continuing:
  whether to keep using `n-envs=4`, whether to disable/space out the
  eval callback given it's now implicated twice, and whether segment
  length should change given it no longer looks like a reliable safety
  margin against this specific failure mode.

## EvalCallback removed; three clean segments, then one more original-pattern leak data point

- **Date:** 2026-10-30
- **Decision implemented:** `EvalCallback`/`eval_env` removed entirely
  from `train_rl_policy.py` (it was only ever for progress monitoring;
  real evaluation now happens post-training via `evaluate_rl_policy.py`'s
  full 45-trial benchmark). `CheckpointCallback` kept as the sole
  callback. Segments resized to ~40k steps, resuming from the intact
  `ppo_slip_ckpt_135360_steps.zip` (not the stale 95,360 one).
- **Result: three consecutive clean segments** -- 135,360 -> 175,296 ->
  215,232 -> 255,168, all completing with exit code 0, zero crashes.
  `ep_rew_mean` over this stretch: 411 (at 175k) -> 398 (at 208k) -> 363
  (at 247k) -- noisy but not collapsing.
- **Segment 4 (toward the final 300,000) failed** -- but this one is a
  clean, separate, confirmatory data point for the ORIGINAL leak theory,
  not a new eval-related incident (exactly the distinction the decision
  above was meant to let us draw). Worker 2 (a `SubprocVecEnv` train
  worker) failed with the identical signature as the very first
  confirmed incident: `ValueError: Error: engine error: Could not
  allocate memory`, logged cleanly across 5 retry attempts, after only
  **16 successful resets** (workers 0/1/3 were at 20/17/21). This
  further confirms the revised characterization: the original leak-type
  failure is real and still occurring, at yet another low reset count
  (16, vs. 91/103 originally) -- reinforcing that it's a per-call
  failure probability, not a fixed call-count threshold, independent of
  the separate eval-related SIGSEGV issue (which should no longer occur
  at all now that EvalCallback is gone).
- **No progress lost worth noting:** `ppo_slip_policy_latest.zip`
  remains intact at 255,168 (confirmed via `PPO.load`) -- segment 4
  only ran a few thousand steps past that before dying, and no
  intermediate `CheckpointCallback` save fired in between (each
  ~40k-step segment's fresh callback instance mostly doesn't accumulate
  enough internal step-calls to cross its own save_freq threshold
  within such a short segment -- only the explicit final `model.save()`
  on clean completion reliably fires at this segment length). Retrying
  the final ~44,832 steps from 255,168.

## Full 300k-step RL training run -- COMPLETE

- **Date:** 2026-10-30/31 (overnight)
- **Result:** the retried final segment completed cleanly -- exit code
  0, `Saved final policy to ../../Data/ppo_slip_policy_latest.zip
  (300224 total steps)`, independently re-verified via `PPO.load()`
  showing `num_timesteps=300224`. The full run is done, start to finish:
  0 -> 80,000 (first attempt, crashed at n-envs=6 then recovered at
  n-envs=4) -> 95,360 (verified resume) -> 135,360 (checkpoint-recovered
  after a confounded crash, then a genuine SIGSEGV) -> 175,296 -> 215,232
  -> 255,168 -> 300,224, across 8 separate process launches, surviving 4
  distinct failure incidents via checkpoint-resume each time.
- **`ep_rew_mean` over the final segment:** climbed from the ~370s (at
  the point of the SIGSEGV investigation) to 414-473 by the end --
  real, continued improvement through the whole run, not a plateau.
- **Real closed-loop success-rate evaluation (`evaluate_rl_policy.py`,
  the actual 45-trial benchmark) against the final checkpoint: running,
  not yet landed as of this entry.** This is the number that actually
  matters for the 80% target -- `ep_rew_mean` includes grip-force and
  effort terms, not just the binary held/slipped outcome, so it is
  informative about whether training was working but is NOT a stand-in
  for the real success percentage. See the next entry once it completes.

## Re-prioritization pass, 2026-10-31 -- best current call, not a final verdict

- **Why now, with the RL number still pending:** re-read this entire
  file end to end per instructions, rather than keep deferring until
  the eval lands. Explicitly framed as provisional -- refine once the
  real 45-trial RL number is in.
- **What's established, pulled together:** the 40.0% hand-tuned ceiling
  has two confirmed, separate causes -- actuator saturation (21/27
  failures, DIRECTLY confirmed via per-step ctrl-vs-ctrlrange logging,
  structurally unfixable by any correction formula) and detection misses
  (6-10/27, strongly correlated with mesh-kind geometry, and -- per the
  independent 20,000-trial open-loop corroboration -- a signal-quality
  problem, not a physical-difficulty one). RL training (which controls
  grasp FORMATION, the only lever that can address saturation) just
  completed its full 300k-step run. A cheap, RL-independent fix for the
  detection-miss group (lower trigger threshold) is running in parallel.
- **Decision tree for what's highest-value next, by RL outcome:**
  - **If the real RL success rate lands >=~70-80%:** target is
    effectively met or close. Highest-value next work shifts to
    robustness/credibility for the write-up -- multi-seed evaluation
    (the RL policy, like the TCN, has never been checked for run-to-run
    variance), and a per-object breakdown of the RL result specifically
    checking whether it fixed the saturation-group objects (its designed
    mechanism) and, incidentally, the detection-miss objects too (RL's
    policy observes `slip_probability` directly and could in principle
    learn its own implicit, object-adaptive threshold, something
    `decide_action()`'s single global threshold structurally cannot do).
  - **If it lands meaningfully above 40% but below ~70%:** the most
    likely next lever is that 300k steps undertrained a non-trivial
    continuous-control task -- `ep_rew_mean` was still climbing, not
    plateaued, when training stopped, which is circumstantial evidence
    for "more training would likely help further," not proof. Resuming
    training further (the checkpoint-resume mechanism is now proven
    reliable across 4 real incidents) is cheap relative to redesigning
    the reward. Revisiting `GRIP_FORCE_COEF`/`GRIP_FORCE_CAP_MULTIPLE`
    (currently a documented first-pass guess, never tuned against a
    real training curve -- one now exists) is the next-cheapest lever
    after that.
  - **If it regresses below 40%, or is a near-degenerate policy (e.g.
    near-zero variance in actions, or always dropping during
    formation):** that would be a real, concerning finding pointing at
    a flawed reward or environment design, not a training-duration
    problem -- would need the kind of reward-shaping-sanity review this
    session already did once (the `GRIP_FORCE_COEF` magnitude check)
    applied more broadly, before training further.
- **What's worth doing regardless of which scenario lands:** the
  detection-miss lower-threshold test (parallel track, closed-loop-
  domain, not RL-dependent) -- it's a real, separate, additive
  opportunity on the baseline hand-tuned policy either way, and cheap
  enough that there's no reason to gate it on the RL number. Result
  pending in the next entry.
- **Not the highest-value use of remaining time, under this target:**
  further mesh-collision-artifact-vs-genuine-difficulty disentangling
  (already de-prioritized below both RL and the detection-miss fix by
  the independent open-loop corroboration, which answered the practical
  question -- "is it worth trying to fix mesh collision" -- well enough
  to act on, even without a fully mechanistic confirmation), and the
  `rl_env.py` per-episode model-rebuild performance question (already
  investigated and declined as too risky for the likely payoff).

## THE real RL closed-loop number -- 64.4%, with a clean win pattern and one new, concentrated failure mode

- **Date:** 2026-10-31
- **Method:** `evaluate_rl_policy.py` against `Data/ppo_slip_policy_latest.zip`
  (num_timesteps=300224, the completed run), deterministic actions,
  same 45-trial benchmark (15 objects x 3 seeds) as every hand-tuned
  comparison in this file.
- **Overall: 29/45 held securely (64.4%).** Up from the hand-tuned
  policy's 40.0% (18/45) -- a real, large improvement, short of the
  stated 80% target but not close to a wash.
- **Split by train/holdout -- a real, if noisy, generalization result:**
  HOLDOUT objects (cone, cylindrical_prism, plus_cross, sphere -- never
  trained on) **10/12 (83.3%)**, actually ABOVE the 80% target. TRAIN
  objects 19/33 (57.6%), lower than holdout. n=12 for holdout is small
  enough that 83.3% should be treated as encouraging, not conclusive,
  but it is a genuine out-of-distribution result, not a cherry-pick.
- **Per-object, compared directly against the hand-tuned policy's own
  per-object numbers (same benchmark, logged earlier in this file):**
  ```
  object              hand-tuned    RL        change
  cone                0/3           3/3       FIXED
  square_pyramid      1/3           3/3       FIXED
  triangular_prism    0/3           3/3       FIXED
  hexagonal_prism     0/3           2/3       mostly fixed
  plus_cross          0/3           3/3       FIXED
  capsule             2/3           3/3       improved
  cylindrical_prism   0/3           1/3       improved
  sphere              3/3           3/3       unchanged (already good)
  rod                 3/3           3/3       unchanged (already good)
  dumbbell            3/3           3/3       unchanged (already good)
  ellipsoid           1/3           1/3       unchanged
  cube                1/3           0/3       WORSE
  rectangular_prism   0/3           0/3       still bad
  flat_disk           2/3           0/3       WORSE
  l_block             2/3           1/3       WORSE
  ```
  **All four mesh-kind objects, plus `plus_cross`, went from near-total
  failure to near-total success.** This is exactly the mechanism the
  detection-miss investigation predicted: the RL policy observes
  `slip_probability` directly, every step, and can learn its own
  implicit object-adaptive response instead of `decide_action()`'s one
  global 0.3 threshold -- it looks like it learned to trust/react to
  the (weaker, noisier) signal these objects produce in a way the fixed
  threshold never could.
- **A genuinely new, concentrated failure mode: formation-phase drops.**
  11 of the 16 RL failures (69%) have `dropped_during_formation=True`
  -- the object falls before the pull phase even begins, something
  structurally impossible under the hand-tuned policy's fixed scripted
  squeeze (it never drops during approach by construction). Only 5 of
  16 failures are actual pull-phase slips (`cylindrical_prism` seed=0,
  `ellipsoid` seed=0/2, `flat_disk` seed=0, `hexagonal_prism` seed=2).
  **The formation-drop failures concentrate almost entirely on 4
  objects: `cube` (3/3 drops), `rectangular_prism` (3/3 drops),
  `flat_disk` (2/3 drops), `l_block` (2/3 drops)** -- exactly the 4
  objects that got WORSE or stayed bad relative to the hand-tuned
  baseline. Geometrically, these are flat-faced boxes/a flat disk/an
  asymmetric compound shape -- notably NOT sharing the mesh-kind
  property that explained the earlier detection-miss pattern; this
  looks like a different mechanism entirely (grasp-formation strategy
  that doesn't generalize to flat/boxy geometry), not a continuation of
  either previously-characterized failure mode.
- **What this means for next steps (updates the decision tree in the
  re-prioritization entry above -- this lands in the "40-70%" bucket):**
  the saturation and detection-miss failure modes both look
  substantially addressed by RL, as designed. The dominant remaining
  problem is new and different: a formation-phase grasping strategy
  that works well for curved/pointed/thin objects but drops flat-faced
  or boxy ones outright, concentrated on a small, identifiable set of
  objects rather than being spread evenly. This is now the single
  highest-value next target -- more total training steps is a weaker
  bet than before, since this doesn't look like an undertrained-in-
  general problem (most objects are already near-ceiling); it looks
  like a specific gap in what the policy learned to do with a specific
  geometry class.

## Formation-drop mechanism found -- the policy never closes the middle finger, at all, on any object

- **Date:** 2026-10-31
- **Method:** `diagnostic_formation_drop.py`, new -- runs the trained
  policy (deterministic) on the 4 implicated objects and logs
  `self.fraction` (per-finger ctrlrange-closed fraction, 5 values) and
  per-finger touch directly from `SlipGraspEnv`'s own public state after
  every step, sampled through the formation phase.
- **Result: the middle finger's fraction (actuator index 2 --
  `right_m-mcp/pip`) is EXACTLY 0.00 at every single sampled step, in
  every single trace, for all 12 trials across `cube`,
  `rectangular_prism`, `flat_disk`, `l_block`** -- both the ones that
  dropped and the two that didn't (`flat_disk` seed=0, `l_block`
  seed=1). The policy never attempts to close that finger at all on
  these objects.
- **Checked immediately whether this is specific to these 4 objects --
  it is NOT.** Ran the same instrumented trace on 4 objects that hold
  securely every time (`sphere`, `cone`, `capsule`, `dumbbell`, seed=0
  each): **identical result, finger index 2 at exactly 0.00 in every
  case.** This is not a failure specific to flat/boxy geometry -- **the
  trained policy has learned a universal, object-independent 4-finger
  grasp strategy that simply never uses the middle finger at all.** It
  works for curved/pointed/thin objects (4 fingers is apparently enough
  support there) and fails specifically where 4-finger support isn't
  enough -- flat/boxy/asymmetric shapes that need the 5th contact point
  to avoid tipping or falling.
- **A plausible, specific explanation rooted in the existing reward
  design (not a new mechanism -- the same `GRIP_FORCE_COEF`/
  `CONTACT_COEF` terms already in `rl_env.py`):** `_grip_force_sum()`
  sums normal force across ALL 5 fingertips and the reward term
  saturates at `GRIP_FORCE_CAP_MULTIPLE=2.0`x the object's weight --
  once 4 fingers alone clear that cap (plausible for most objects,
  given the touch values logged above regularly show 2-3 fingers
  individually contributing 10-40+ units), there is ZERO additional
  grip-force reward for engaging a 5th finger. `CONTACT_COEF *
  contact_frac` (fraction of fingers touching) DOES still reward going
  from 4/5 to 5/5 (a real, if modest, +0.2/step), but apparently not
  enough to overcome whatever made 5-finger contact less reliable/more
  effort-costly to discover during training (`EFFORT_COEF` penalizes
  the extra action magnitude; a finger that's rarely useful could stay
  a worse local optimum if the exploration never found a configuration
  where using it paid off clearly). This is a plausible, not fully
  proven, account of WHY the current reward balance allowed this --
  consistent with the reward terms as designed, not a new unexplained
  mechanism.
- **Concrete, specific fix this points to (not yet implemented --
  reward-shaping change + a full retrain is a substantial next step,
  flagged for a decision rather than done unilaterally):** strengthen
  the incentive to use ALL 5 fingers specifically, rather than
  rewarding only the aggregate contact fraction/force. Two concrete
  options: (a) raise `CONTACT_COEF` relative to `GRIP_FORCE_COEF`/
  `EFFORT_COEF` so 5/5 contact is worth meaningfully more than 4/5, or
  (b) add a direct per-finger penalty for any finger whose fraction
  stays near 0 for an extended stretch during formation (today's reward
  only ever sees the AGGREGATE `contact_frac`/summed grip force, never
  "finger N specifically has done nothing") -- (b) is more targeted at
  the actual mechanism found here and less likely to have side effects
  on objects that are already working well with 4 fingers.
- **This is a cleanly explained mechanism, not an inconclusive one** --
  unlike the earlier cube/seed=0 MuJoCo-reproducibility anomaly, this
  didn't require stopping short. The remaining open question is
  specifically whether (a) or (b) above (or something else) is the
  right fix, and whether the time cost of a reward change + full
  retrain is worth it at this point in the project, not whether the
  mechanism itself is understood.

## Lower trigger threshold for the hand-tuned policy -- tested, makes things WORSE, does not rescue detection misses

- **Date:** 2026-10-31
- **Setup:** `diagnostic_lower_threshold.py`, same 45-trial benchmark,
  hand-tuned `decide_action()` (Model A, NOT the RL policy -- this
  tests whether the detection-miss group is fixable cheaply on the
  baseline control path specifically), sweeping the trigger threshold
  down from the current 0.3 to 0.2 and 0.15.
- **Result: does not help the detection-miss group, and actively hurts
  everything else.**
  ```
  BASELINE (threshold=0.3): 18/45 (40.0%)  -- detection-miss objects: 3/15 held
  threshold=0.2:            16/45 (35.6%)  -- detection-miss objects: 3/15 held
  threshold=0.15:           14/45 (31.1%)  -- detection-miss objects: 2/15 held
  ```
  The detection-miss-prone objects (the mesh-kind-heavy group) stayed
  flat at 3/15 (threshold=0.2) or got slightly worse (2/15 at
  threshold=0.15) -- lowering the gate does NOT rescue them. Worse,
  overall performance regressed monotonically as the threshold dropped
  (40.0% -> 35.6% -> 31.1%), and the regression is concentrated in the
  "everything else" group (15/30 implied at baseline -> 13/30 -> 12/30)
  -- a more sensitive gate starts firing on noise in trials that didn't
  need correction, without fixing the trials that genuinely have weak
  signal.
- **Why this is informative, not just a negative result:** it confirms
  the detection-miss problem is NOT a simple miscalibrated-threshold
  issue on the hand-tuned control path -- the underlying `max_slip_
  probability` signal for these objects is genuinely weak/noisy (per
  the mesh-collision-driven signal-quality hypothesis above), not just
  sitting slightly below an overly conservative cutoff. A single global
  threshold, however tuned, cannot fix this on the hand-tuned path --
  which sharpens, in direct contrast, why the RL result above is
  significant: RL's policy has access to the SAME underlying
  `slip_probability` signal but learned something qualitatively
  different with it (an object-adaptive response, not a better fixed
  cutoff), and that is specifically what flipped the mesh objects from
  near-total failure to near-total success. The hand-tuned path has no
  cheap fix remaining for this failure mode; RL already solved it.

## Reward fix implemented: MIN_FINGER_FRACTION_COEF, targeting the never-closes-middle-finger finding

- **Date:** 2026-10-31
- **Change:** added `MIN_FINGER_FRACTION_COEF = 0.5` to `rl_env.py`'s
  formation-phase (phase 0/1) reward -- `min(self.fraction)` (the
  WORST-engaged finger's own commanded ctrlrange fraction, not an
  average), added alongside the existing `CONTACT_COEF`/`GRIP_FORCE_COEF`
  terms rather than replacing them. Full reasoning (why a maximin term
  specifically, why it targets commanded fraction rather than measured
  touch, why reweighting the existing aggregate terms was considered and
  rejected as insufficiently targeted) is in `rl_env.py`'s module
  docstring, same documentation discipline as the `GRIP_FORCE_COEF`
  entry before it.
- **Magnitude checked empirically before trusting it** (same practice as
  last time, after `GRIP_FORCE_COEF`'s first pass turned out 5-40x too
  strong): `smoke_test_rl_env.py` under random actions, same 3 seeds as
  always -- episode totals went from 12.66/185.33/22.70 (pre-fix) to
  14.80/203.73/31.90 (post-fix) -- a modest, proportionate 15-40%
  increase, not a dominant jump. Max per-substep contribution
  (0.5 x 1.0 = 0.5) sits between `GRIP_FORCE_COEF`'s (0.3) and
  `CONTACT_COEF`'s (1.0) -- a reasonable first pass, not yet tuned
  against a real training curve with this term active.
- **Next: a cheap, bounded sanity check** -- resume training from the
  completed 300,224-step checkpoint with the new reward active, for a
  short (~20-30 min) segment, checking specifically whether the middle
  finger's fraction starts moving at all off its locked 0.00. See the
  next entry for the result.

## Sanity-check result: middle finger did NOT move -- stopping before a full retrain, per plan

- **Date:** 2026-10-31
- **What happened getting here:** the infrastructure fought back hard --
  5 separate launch attempts before getting usable data. Attempts 1-2
  crashed on the literal FIRST reset of every worker (zero rollouts
  completed), a more severe pattern than anything seen before. Checked
  for orphaned processes per a direct request: found one genuine orphan
  (PID 19188, a `multiprocessing-fork` child whose parent process, PID
  11060, no longer existed -- alive since 2026-10-30 5:59 PM, ~16 hours).
  Could not terminate it myself (`Stop-Process` denied by the sandbox's
  auto-mode classifier -- flagged to the user to clean up manually, not
  worked around). Also found free memory had dropped from 16.3GB to
  7.3GB, consistent with each of the 4 fresh worker processes using
  ~2.17GB (the raw `Get-Process` reading showed corrupted-looking
  negative `WorkingSet` values -- an Int32 display overflow once
  corrected). Attempts 3-5 crashed later, in the already-characterized
  `ValueError: Could not allocate memory` pattern (12-21 resets each) --
  consistent with the known issue, not a new one. Attempt 5 (after
  lowering `CheckpointCallback`'s save frequency so a short segment
  would actually produce a usable checkpoint -- see the
  `--checkpoint-freq` rename entry implicit in this one) finally
  produced real checkpoints at 305,224 / 310,224 / 315,224 steps before
  dying at 317,632.
- **The actual result, checked against `ppo_slip_ckpt_315224_steps.zip`
  (15,000 steps of training with `MIN_FINGER_FRACTION_COEF` active):**
  `diagnostic_formation_drop.py` on `cube`, `rectangular_prism`,
  `sphere`, and `cone` (two that drop, two that don't) all show the
  identical result as before the fix -- **the middle finger's fraction
  is still exactly 0.00 in every case, unchanged.** No movement at all,
  not even a small nonzero drift.
- **Per the agreed branch logic, stopping here -- NOT proceeding to a
  full fresh retrain.** Honest reasoning about why this result doesn't
  necessarily mean the fix is wrong, stated for whoever picks this up
  next: this was a CONTINUATION from an already-converged, 300k-step-
  trained checkpoint with a very deeply entrenched 4-finger strategy --
  PPO's clipped update (`clip_range=0.2`) limits how fast any policy can
  shift per update, and 15,000 steps (~15 policy updates) against 300k
  steps of prior reinforcement may simply not be enough leverage to
  dislodge an already-stable local optimum, even if the new reward term
  would have prevented that optimum from forming in a FRESH run from
  scratch (which is exactly why the plan called for a from-scratch
  retrain as the real test, not a continuation -- this sanity check was
  specifically a cheaper, faster proxy for that, and it's an honestly
  ambiguous proxy, not a clean disproof). No evidence either way on
  whether a fresh run would behave differently -- that question is
  explicitly NOT answered by this result, only deferred.
- **Not done:** the full fresh retrain (correctly withheld, per the
  branch instructions). No conclusion drawn about whether
  `MIN_FINGER_FRACTION_COEF` is the wrong fix -- only that continuing
  from the biased checkpoint doesn't cheaply prove it's the right one
  either. A decision is needed on whether to spend the time on a full
  fresh retrain anyway (the only way to actually test this), try a
  stronger coefficient first as an even cheaper intermediate check, or
  deprioritize this given how much infrastructure friction today has
  cost.

## Escalated MIN_FINGER_FRACTION_COEF 0.5 -> 3.0, user-authorized, proceeding straight to a full retrain regardless of this check's outcome

- **Date:** 2026-10-31
- **Decision:** user authorized, in advance, a 3-step plan with no need
  to check back in between: escalate the coefficient meaningfully,
  cheap-check whether the finger moves, then proceed to a full fresh
  retrain EITHER WAY (not gated on the cheap check's result, since a
  continuation was already shown to be a weak/ambiguous test).
- **Escalation: 0.5 -> 3.0** (6x) -- makes this term's max per-substep
  contribution (3.0) clearly dominant over `CONTACT_COEF`'s (1.0, 3x
  smaller) and `GRIP_FORCE_COEF`'s (0.3, 10x smaller), a deliberate,
  aggressive jump rather than a small nudge, on the theory that if a
  gentle version can't move an entrenched habit, the real test is
  whether a strong one can. Magnitude checked empirically first (same
  practice as always): `smoke_test_rl_env.py`, same 3 seeds, went from
  25.53/295.68/77.94 (at 3.0) vs. 14.80/203.73/31.90 (at 0.5) -- a
  roughly proportionate 1.7-2.4x increase, no runaway blowup, no crash.
  Full reasoning in `rl_env.py`'s module docstring.
- **Next: same cheap continuation check as before** (short, fine-grained
  checkpointing given today's infra friction), then straight to the
  full fresh retrain regardless of what it shows -- see the next entries.

## Strong-coefficient check result -- still 0.00, but a weaker sample (5k steps, not 15k) -- proceeding to full retrain per pre-authorized plan

- **Date:** 2026-10-31
- **Result:** crashed again (same known `EOFError`/leak pattern), but a
  checkpoint was captured at 305,224 steps -- only 5,000 steps of
  training with `MIN_FINGER_FRACTION_COEF=3.0` active (vs. 15,000 steps
  for the 0.5 check, since this run crashed earlier). Checked the same
  4 objects (`cube`, `rectangular_prism`, `sphere`, `cone`): **middle
  finger fraction still exactly 0.00, no movement.**
- **Honestly weaker evidence than the 0.5 check, not stronger** -- 5,000
  steps is only ~5 policy updates, a smaller sample than the already-
  ambiguous 15,000-step/~15-update check. This result should NOT be
  read as "a 6x stronger coefficient also failed, so the fix doesn't
  work" -- it's barely more informative than not checking at all, given
  how little training actually happened before the crash.
- **Not iterating further on cheap checks, per the pre-authorized plan.**
  Proceeding straight to the full fresh retrain (300k steps, fresh PPO,
  `MIN_FINGER_FRACTION_COEF=3.0`, segmented/checkpoint-resume) now,
  regardless of this inconclusive result -- see the next entry.

## Full fresh retrain complete -- the fix worked exactly as designed, and REGRESSED overall performance (64.4% -> 44.4%)

- **Date:** 2026-10-31 (ran unattended overnight)
- **Training completed cleanly:** the self-healing orchestrator
  (`Data/_run_fresh_retrain.sh`) needed 8 process attempts total -- 6
  crashes in the first 45,000 steps (the known intermittent leak,
  including two immediate zero-progress failures), then one single
  continuous run from 45,000 all the way to completion at 301,000 steps
  with zero further crashes. Final policy: `Data/ppo_slip_policy_fresh.zip`.
- **Mechanism check: the fix worked exactly as intended.**
  `diagnostic_formation_drop.py` against the final policy on the 4
  previously-idle-middle-finger objects (`cube`, `rectangular_prism`,
  `flat_disk`, `l_block`) now shows **all 5 fingers reaching fraction
  ~1.00 in every single trace** (spread 0.00-0.13, vs. the fixed
  0.00-for-finger-2/spread-1.00 pattern before) -- the "never closes
  the middle finger" problem is completely gone, confirmed directly,
  not inferred.
- **But the real 45-trial evaluation shows a clear regression: 20/45
  (44.4%), down from the pre-fix fresh run's 29/45 (64.4%).** TRAIN
  objects 13/33 (39.4%, down from 57.6%), HOLDOUT 7/12 (58.3%, down
  from 83.3%). Full per-object comparison:
  ```
  object              pre-fix (64.4%)   post-fix (44.4%)   change
  cube                0/3               0/3                no change (target object, STILL fails)
  rectangular_prism    0/3               0/3                no change (target object, STILL fails)
  l_block              1/3               1/3                no change (target object, still mixed)
  flat_disk            0/3               0/3                no change (was formation-drop, now pull-phase slip -- same count, different mechanism)
  capsule              3/3               0/3                REGRESSED -- was perfect, now always slips
  rod                  3/3               1/3                REGRESSED
  plus_cross           3/3               1/3                REGRESSED
  triangular_prism     3/3               0/3                REGRESSED -- was perfect, now always drops
  sphere               3/3               2/3                slightly worse
  ellipsoid            1/3               3/3                improved
  cone/dumbbell/square_pyramid/hexagonal_prism/cylindrical_prism  unchanged (already good or already mixed)
  ```
- **The honest takeaway: forcing universal 5-finger full closure is not
  a free win.** The 4 objects the fix specifically targeted did NOT
  improve at all -- `cube` and `rectangular_prism` still drop during
  formation every single time, now via a DIFFERENT mechanism (full
  5-finger engagement, very fast, still ends in a drop -- see the
  `diagnostic_formation_drop.py` trace lengths, still as short as
  24-36 macro-steps, barely changed from before the fix) -- meaning
  these two objects have a second, independent problem that idle-
  finger-avoidance was never going to fix on its own (most likely
  something about the SPEED/coordination of closing, or the approach
  trajectory itself, not which fingers engage). Meanwhile, forcing
  full closure on every finger for every object cost real performance
  on objects (`capsule`, `rod`, `plus_cross`, `triangular_prism`) that
  were previously held perfectly well by a more selective grip --
  likely because always slamming to 1.0 fraction on all 5 fingers
  isn't actually the best grip for every shape, and the previous
  policy's "skip finger 2" strategy may have been a genuinely adapted,
  not just lazy, solution for several of these.
- **Net result: this specific fix should NOT be kept over the pre-fix
  policy.** 64.4% (`Data/ppo_slip_policy_latest.zip`, the pre-fix fresh
  run) remains the best verified closed-loop result from this entire
  session. This is a real, reportable negative result -- the
  mechanism-level diagnosis (never closes the middle finger) was
  correct and the fix for that exact mechanism worked, but the
  mechanism wasn't the (or wasn't the only) actual cause of the
  cube/rectangular_prism failures, and fixing it indiscriminately
  everywhere else had a net-negative side effect.
- **Not done, pending a decision:** no further coefficient tuning
  attempted. A decision is needed on whether to: (a) revert to the
  64.4% pre-fix policy as the project's reported result, (b) try a more
  moderate/targeted version of the fix (e.g. a smaller coefficient, or
  only rewarding minimum-finger-engagement for the 4 specific objects
  rather than universally -- though object-conditional reward shaping
  is a bigger design change than anything tried today), or (c)
  investigate cube/rectangular_prism's actual drop mechanism directly
  (the diagnostic infrastructure already exists) before changing the
  reward again, since the fix-worked-but-didn't-help result suggests
  the real cause for those two specific objects is still unidentified.

> **CURRENT REFERENCE RESULT, unambiguously (2026-10-31): 64.4%
> (29/45), policy checkpoint `Data/ppo_slip_policy_latest.zip`
> (verified intact, `num_timesteps=300224`, untouched by every attempt
> below this point).** `Data/ppo_slip_policy_fresh.zip` (44.4%, the
> `MIN_FINGER_FRACTION_COEF=3.0` regression above) is kept on disk for
> comparison only -- it is NOT the result to cite or build on going
> forward unless a future entry explicitly supersedes this note.

## Formation-drop mechanism for cube/rectangular_prism/flat_disk/l_block -- two distinct sub-patterns found, not one clean mechanism

- **Date:** 2026-11-01
- **Method:** `diagnostic_formation_drop_detailed.py`, new -- logs EVERY
  step (not sampled) through formation against the VERIFIED-GOOD 64.4%
  policy (`Data/ppo_slip_policy_latest.zip`, not the regressed one),
  tracking per-finger first-contact step, object speed, and cumulative
  "fall" every step. Compared the 4 implicated objects directly against
  `sphere`/`cone` (successful, same policy) run through the identical
  logging.
- **Ruled out directly: this is NOT a "too late no matter what"
  problem.** In every trace, including all 4 failures, `obj_speed` and
  `fall` are exactly 0 for the first ~50-65 steps -- the object is
  completely static before any finger makes contact. Whatever goes
  wrong happens AFTER contact begins, not before any correction-
  relevant signal could possibly have existed.
- **Ruled out directly: raw closing SPEED (the commanded fraction's own
  ramp-up rate) is not the differentiator.** The fraction trajectory
  (0.08 at step 5 -> 1.0 by step 65-80) is essentially identical across
  every object tested, success or failure -- the policy doesn't close
  perceptibly faster or slower on the objects that fail.
- **A real, measured difference: which/how-many fingers engage, and
  how tightly they arrive together -- but it splits into two different
  sub-patterns, not one:**
  - **Successful grasps (`sphere`, `cone`) use only 3 fingers** (thumb=0,
    index=1, ring=3) -- the middle finger (2) never touches (expected,
    per the earlier finding) AND the pinky (4) never touches either --
    arriving within a tight **10-15 step window** of each other.
  - **`cube` (both seeds) and `flat_disk`: the SAME 3 fingers contact
    early, but the pinky (finger 4) ALSO eventually makes contact, 70-95
    steps LATE** (step 130-150, vs. 55-80 for the other three). After
    that, the grip looks superficially stable (object speed settles to
    ~0.01-0.02, similar magnitude to the successful cases' steady
    state) but exhibits a **slow, steady downward sag** that never
    fully arrests -- `fall` creeps from ~0.03 past the 0.05 threshold
    over 400-500 further steps, including measurable drift BEFORE
    `ramp_steps`/gravity-reenable even happens (so it isn't just
    gravity winning once enabled -- something about the 4-finger
    configuration itself has a small net-downward bias).
  - **`rectangular_prism` (both seeds) and `l_block`: a DIFFERENT
    pattern -- only 3 fingers ever contact (same count as the
    successful cases), but a sudden, violent velocity spike
    (object speed reaching 0.4-1.0, roughly 10-50x the successful
    cases' steady-state speed) appears partway through closing, often
    coinciding with contact partially or fully DROPPING OUT across
    multiple fingers simultaneously (e.g. `l_block` seed=0 at step 85:
    touch goes from 3 fingers reading 10-13 units down to essentially
    zero on all of them in a single logged step) -- consistent with an
    unstable intermediate grip configuration tipping the object before
    full closure, not a slow sag. Once this starts, the object crosses
    the full 5cm drop threshold within 9-25 further steps -- genuinely
    "too late" from THAT point on, even though it wasn't too late
    before the instability triggered.
- **Honest synthesis, not a single clean mechanism:** both sub-patterns
  plausibly trace back to the same root cause (a grip strategy that
  converges 3 fingers in a configuration that works geometrically for
  curved/pointed objects, but doesn't provide adequate support for
  flat-faced/boxy/elongated ones) -- but the FAILURE MODE differs by
  object: `cube`/`flat_disk` get a 4th (late, unhelpful) finger and
  sag slowly; `rectangular_prism`/`l_block` keep the same 3-finger
  count as successful grasps but that configuration is specifically
  unstable for their geometry and fails fast. This is a genuinely
  harder, multi-factor finding than the saturation diagnosis was --
  there is no single lever (like "actuators are pinned") visible here.
- **Not directly checked, flagged honestly rather than skipped:**
  contact LOCATION on the object (edges/corners vs. flat-face-center)
  was not instrumented in this pass (would need `mjData.contact`
  position inspection, same technique as the earlier cone-vs-sphere
  mesh-collision diagnostic, not yet pointed at these 4 objects) --
  plausible given the flat/angular geometry common to all 4 failing
  objects, but not directly measured here, so not claimed as confirmed.
- **What this suggests for a fix, without guessing at one yet:** the
  `rectangular_prism`/`l_block` pattern (sudden multi-finger contact
  loss) looks like it could be addressed by a stability-focused reward
  term (e.g. penalizing sudden touch-count drops during formation,
  distinct from the already-tried min-finger-engagement approach which
  regressed performance) -- but per instructions, no reward change is
  being made on this finding alone. The `cube`/`flat_disk` slow-sag
  pattern looks different again and may need its own explanation before
  any fix is attempted there specifically.

## Maintenance note, 2026-10-31 -- latent 25ch issue in rl_env.py (suggestion only, not changed)

- Static read only (no simulation run). `SlipGraspEnv._observe_step()` calls
  `predict_slip()` as soon as the history holds `WINDOW_SIZE` (300) steps,
  but with `--in-channels 25` the third extra scalar (`grasp_force_at_settle`)
  is only appended at the end of settle (step `ramp_steps + settle_steps`,
  800 by default). Between step 300 and step 800 `predict_slip()` would
  receive 24 columns instead of 25 and fail with a shape mismatch.
  `closed_loop_sim.run_trial()` does not have this problem because it only
  predicts in phase 2. Does not affect any run logged so far (all RL runs
  use the 22-channel Model A). Suggested fix if RL is ever run with
  Model B 25ch: skip prediction (use 0.0) until `_extra_scalars` is complete.

## Rigorous static code audit of `Synthetic data creation/Simulation/`, 2026-11-02

- **Method:** static read only (nothing executed; no MuJoCo/submodule in the sandbox). Read in full: `rl_env.py`, `closed_loop_sim.py`, `train_rl_policy.py`, `evaluate_rl_policy.py`, `batch_closed_loop_trials.py`, plus a targeted cross-check of every `diagnostic_*.py` that monkeypatches `cls.decide_action`. `generate_trials.py` and the remaining diagnostics were not line-by-line audited this pass.
- **FIXED (confirmed bug): `diagnostic_clean_comparison.py` `original_policy` crashed on every call.** `run_trial()` calls `decide_action(predicted, driven, feats=feats)`; `decide_action()`'s docstring claims old 2-arg monkeypatched policies "keep working unchanged", but this script's `original_policy(slip_probability, driven_targets)` takes no `**kwargs` (the other diagnostics were updated when `feats=` was added). Every BASELINE trial in phase 2 would raise `TypeError`, which `run_batch`'s per-trial `except` swallows and logs as a failed trial, so the baseline half of the comparison is silently empty/zero. Fix: added `**kwargs` to the signature (one line, no behavior change otherwise). The logged 13.3% -> 40.0% comparison pre-dates the `feats=` change, so that result is unaffected.
- **FINDING (not changed): `rl_env.py` still has `MIN_FINGER_FRACTION_COEF = 3.0` active, but the documented reference result (64.4%, `ppo_slip_policy_latest.zip`) was trained WITHOUT that term.** results.md (2026-10-31) says the 3.0 fix should not be kept and left revert-vs-retune as a pending decision; the code was never reverted. Any retrain or reproduction of the 64.4% result from the current file will use the 3.0 reward (the 44.4% regression config). Not a scope-creep revert, just an undecided state that is easy to trip over. Suggest the owner set the coefficient to 0.0 (or document it) once decided.
- **FINDING (not changed): `--max-steps` means different things in the two paths.** `closed_loop_sim.run_trial()` treats it as TOTAL steps (ramp+settle+monitor; default 1500 leaves only 700 monitor steps, shorter than the 500-step pull ramp plus any hold), whereas `rl_env` uses it as monitor steps AFTER settle (total 800+1500). The module docstring's "same total step budget as the hand-tuned pipeline" is therefore not true at defaults; the 40.0% baseline used `--max-steps 2500` (1700 monitor steps), so it is roughly but not exactly comparable to the RL eval's 1500.
- **FINDING (not changed): `evaluate_rl_policy.py` and `batch_closed_loop_trials.py` drop crashed trials from the denominator.** A trial that raises is excluded from `n_ok` and the success rate, so errors inflate the reported rate (the FAILED lines are printed, but the "Overall X/Y" line does not flag it).
- **Minor drift (not changed):** `closed_loop_sim.py` module docstring still says `decide_action()` "is a stub that always returns no change ... left for you to write"; it is implemented. `train_rl_policy.py` comment says "105-dim obs"; with 5 fingertips the observation is 5*(4*5+3)=115. `args.squeeze_intensity = 0.95` set in train/evaluate is unused by `rl_env` (harmless). `rl_env.reset()` leaves a scene XML on disk when a build attempt fails and retries. The latent 25ch timing issue logged 2026-10-31 is confirmed still present.

## Rigorous static code audit, follow-up pass, 2026-11-03

- **Method:** static read only (nothing executed). Completed the files the 2026-11-02 pass did not cover: `generate_trials.py` read line by line, plus signature cross-checks of `diagnostic_stiff_contacts.py` (calls `build_trial_xml` / `run_single_trial` / `build_object_assignments` / `_find_body` with the current signatures) and `diagnostic_visualize_onset.py`. No commits touched `Simulation/` since the 2026-11-02 fix (HEAD was that audit commit), so there was no new diff to cross-check for scope creep.
- **Result: no new correctness bugs found; no code changed.** Checked in `generate_trials.py`: gravcomp spawn/re-enable at the ramp->settle boundary, displacement reference latch at the last settle step, clean-vs-noisy signal use for contact detection and `grasp_force_at_settle`, pull ramp, actuator neutral clamping, pad-friction warnings, manifest/npz field consistency, and balancing (seeded, drops empty buckets). All match their docstrings.
- **Minor drift (not changed):** `--n-trials` help text says "split evenly across the 15 objects"; this holds only when `n_trials` is divisible by 15 (otherwise the truncated final block gives some objects one extra trial). `diagnostic_stiff_contacts.stiffen_contacts` silently skips a missing fingertip body (it only reports the patched-geom count).
- The open findings from 2026-11-02 and the 2026-10-31 latent-25ch note remain unresolved and unchanged.
