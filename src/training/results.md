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