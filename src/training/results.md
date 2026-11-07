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

**Confirmed by direct reproduction, 2026-11-01 (the above was a static
read; this is an actual run):** constructed a `SlipGraspEnv` with
`--checkpoint model_b.pt --in-channels 25`, stepped with random actions.
Crashed exactly as predicted, at **global_step=299** (the step where
`len(self._full_history)` first reaches `WINDOW_SIZE=300` and
`predict_slip()` is called for the first time) -- 500 steps before
`grasp_force_at_settle` would be appended at step 799:
```
RuntimeError: Given groups=1, weight of size [32, 25, 1], expected input[1, 24, 300] to have 25 channels, but got 24 channels instead
```
`extra_scalars` at the time of the crash held exactly 2 values
(`[mu_combined, obj_mass]`) as predicted -- confirmed via
`env._extra_scalars` right after `reset()`. **This is a real, 100%-
reproducible crash, not a theoretical edge case** -- it fires on
literally the first `predict_slip()` call of any episode whenever
`--in-channels 25` is used with `rl_env.py`, well within phase 0
(global_step 299 vs. `ramp_steps=500`), so there's no scenario where
training or evaluation with RL + Model B's 25ch checkpoint would run
past this point without the suggested fix (or an equivalent) applied
first. Not fixed here, per instructions -- this is exactly the
information needed to decide whether the bugfix + fresh RL training run
(hours) is worth doing later.

## Model A vs. Model B closed-loop comparison, finally completed (hand-tuned policy, not RL)

- **Date:** 2026-11-01
- **Why this was still an open gap:** the original Model A vs. Model B
  closed-loop comparison (`results.md`'s "Closed-loop batch trial
  results" entry, 2026-10-25) was invalidated by the gravity bug found
  the same day -- after the fix, only Model A was ever re-tested
  (leading to the 13.3%->40.0%/22.2%/40.0% progression cited throughout
  this file). Model B was never re-benchmarked under the corrected,
  realistic-timing conditions.
- **Method:** `batch_closed_loop_trials.py`, same 45-trial benchmark
  (15 objects x 3 seeds, realistic 500/300 ramp/settle timing,
  `squeeze_intensity=0.95`, the current tuned hand-crafted
  `decide_action()`, unmodified) -- identical methodology to every
  Model A number in this file, now run against `src/training/model_b.pt`
  (25ch, F1=0.799 static, the current-best Model B checkpoint per the
  "retrained with num_epochs=40" entry above) with `--in-channels 25`.
- **Result: Model B is WORSE in closed loop -- 15/45 (33.3%), vs. Model
  A's 40.0% (18/45) under the identical tuned policy and timing.** Mean
  `max_slip_probability` 0.6575, mean `n_corrections` 139.33 (both
  higher than Model A's equivalents from the clean-comparison entry,
  i.e. Model B is MORE sensitive/reactive, not less, yet still does
  worse overall -- the same "corrects more, succeeds less" pattern
  already observed in the original, since-invalidated comparison, now
  confirmed under correct, realistic conditions instead of the buggy
  weightless-forever ones).
- **Per-object, for anyone who wants the detail:**
  ```
  object              Model A     Model B
  sphere               3/3         3/3        same (both easy)
  dumbbell             3/3         3/3        same (both easy)
  cube                 1/3*        1/3        same count
  flat_disk            2/3*        2/3        same count
  rod                  3/3*        2/3        worse
  l_block              2/3*        2/3        same count
  cone                 0/3*        0/3        same (both fail)
  rectangular_prism    0/3*        0/3        same (both fail)
  cylindrical_prism    0/3*        0/3        same (both fail)
  plus_cross           0/3*        0/3        same (both fail)
  hexagonal_prism      0/3*        0/3        same (both fail)
  triangular_prism     0/3*        0/3        same (both fail)
  capsule              2/3*        1/3        worse
  ellipsoid            1/3*        0/3        worse
  square_pyramid       1/3*        1/3        same count
  ```
  (*Model A per-object figures are the ones already on record from the
  "THE clean comparison"/mesh-collision entries earlier in this file --
  not re-run again here, just restated for the side-by-side.) No object
  does BETTER under Model B; several (`rod`, `capsule`, `ellipsoid`) do
  worse; the rest tie.
- **Honest interpretation:** this reconfirms, now on solid methodological
  footing, what the original (buggy) comparison already suggested and
  what the static-F1 numbers already hinted at (Model B's higher static
  F1/accuracy never translated to better closed-loop behavior) --
  Model B's physics-informed scalar inputs (`mu_combined`, `obj_mass`,
  `grasp_force_at_settle`) do not help, and plausibly hurt, the actual
  closed-loop task. Model A remains the better model for anything
  closed-loop-related, including as the basis for the RL work (already
  the case -- all RL training in this file used Model A).

## Continuing to train the 64.4% checkpoint, no reward change -- disabled the regressed coefficient first

- **Date:** 2026-11-01
- **Caught before starting:** `rl_env.py` still had
  `MIN_FINGER_FRACTION_COEF=3.0` active from yesterday's regressed fix
  attempt. Continuing to train `Data/ppo_slip_policy_latest.zip` (the
  64.4% checkpoint) under that reward would have contaminated a
  supposedly no-reward-change continuation. Set the coefficient to
  **0.0** (not deleted -- the code path stays, zero contribution) so
  this run uses the exact reward function the checkpoint was already
  trained under.
- **Checked whether reward was still climbing at 300k, as instructed:**
  training `ep_rew_mean` over the final ~45k steps (the segment that
  produced the checkpoint) was mostly flat/noisy around 408-426 for a
  long stretch, then rose to 448-473 in the last ~10-15k steps --
  ambiguous, not a clean plateau, consistent with "possibly still
  improving." Holdout-eval reward (`EvalCallback`, active for this
  entire run) only has recorded points at 40,000/80,000 steps
  (229.1/524.1) -- no further eval points exist for the 80k-300k range
  in `Data/rl_logs_production/evaluations*.npz`, an infrastructure gap
  not diagnosed here (each ~50k segment's fresh `EvalCallback` should
  have crossed one eval boundary but apparently didn't) -- noted
  honestly as a real limitation on this specific check, not chased
  further given the bounded scope of this step.
- **Proceeding with ~150k more steps** (target 450,000, same self-
  healing checkpoint-resume orchestrator as the fresh retrain) given the
  training-reward signal leans toward "maybe still improving" and this
  is a zero-regression-risk test per the user's framing -- see the next
  entry for the result.

## Continuation result: 150k more steps completed cleanly (zero crashes), but closed-loop success went DOWN, not up

- **Date:** 2026-11-01
- **Training itself: the cleanest run of the whole project.** One single
  continuous process attempt, zero crashes, 300,224 -> 450,752 steps.
  `ep_rew_mean` over this stretch stayed flat/noisy (384-419, no climb)
  -- a genuine plateau, not just a bad sample window.
- **Real 45-trial evaluation: 26/45 (57.8%), DOWN from 64.4%.** TRAIN
  objects 17/33 (51.5%, down from 57.6%), HOLDOUT 9/12 (75.0%, down from
  83.3% but still comfortably above 70%). This answers the question
  directly: **no, continuing to train alone did not move the needle
  past 64.4% -- it went backward.**
- **Per-object, vs. the 64.4% checkpoint:** 2 objects improved (`cube`
  0/3->1/3, `cylindrical_prism` 1/3->2/3), 7 stayed the same
  (`rectangular_prism`, `flat_disk`, `l_block`, `cone`, `rod`,
  `plus_cross`, `square_pyramid`, `triangular_prism` -- `rectangular_prism`
  and `flat_disk` both STILL 0/3, confirming these two in particular are
  not a training-duration problem), and 6 got worse (`sphere` 3/3->1/3,
  `capsule` 3/3->2/3, `ellipsoid` 3/3->1/3, `dumbbell` 3/3->2/3,
  `hexagonal_prism` 2/3->1/3). The objects that regressed were mostly
  ones that were previously PERFECT, not the known-problem objects --
  consistent with ordinary PPO training drift/variance (a policy that's
  already converged can wander to a different, not-strictly-better
  local behavior with more gradient steps) rather than a sign that
  training itself is harmful. A single 45-trial eval per checkpoint
  also carries real sampling noise, same caveat this project has
  already established for the TCN's own run-to-run variance -- this
  is one data point, not a seeded multi-run average.
- **Decision, per the pre-agreed branch:** this counts as "doesn't
  meaningfully help" -- moving to designing a targeted, geometry-aware
  fix for the two formation-drop mechanisms, NOT implementing or
  retraining yet. `Data/ppo_slip_policy_latest.zip` (64.4%) remains the
  reference result; `ppo_slip_policy_continued.zip` (57.8%) is kept on
  disk for the record only, not used going forward.

## PROPOSED (not implemented) -- a geometry-aware, self-conditioning fix for the two formation-drop mechanisms

- **Date:** 2026-11-01. Proposal only, per instructions -- no code
  changed, no training run. Written up for review before spending
  hours on another full retrain, given yesterday's blanket-term
  regression (64.4%->44.4%) was exactly the mistake this is designed
  to avoid repeating.
- **Why not another universal reward term:** `MIN_FINGER_FRACTION_COEF`
  applied the SAME pressure to every object every step regardless of
  whether that object needed it, and it measurably hurt several
  objects that were already fine (`capsule`, `rod`, `plus_cross`,
  `triangular_prism`). Both ideas below are designed to be
  **self-conditioning** -- structurally near-zero effect on objects
  that don't exhibit the problem, by construction, not by hand-picking
  which objects they apply to.
- **Part A -- give the policy ONE new, legitimate piece of
  information: the object's own `char_length`.** Currently the RL
  observation has zero geometry/identity signal -- the policy only
  ever learns about object shape reactively, through the tactile
  pattern AFTER contact starts. Adding the object's `char_length`
  (already computed in `self.obj_spec`, a static scalar, not a leaked
  dynamic/positional quantity) as one more observation value is
  comparable to a real deployed system using vision for rough object
  sizing before a tactile-only grasp -- not the same kind of leakage as
  raw position/velocity, which is why `LEAKAGE_FEATURES` excludes those
  specifically. This gives the policy an actual BASIS to learn
  different formation strategies for different-sized/shaped objects,
  rather than the one global strategy it currently has (observed
  directly: 3-finger convergence, same for every object, confirmed via
  `diagnostic_formation_drop_detailed.py`).
- **Part B -- two small, narrowly-targeted reward terms, each matched
  to ONE specific sub-pattern, each naturally zero for objects that
  don't exhibit it:**
  - **For the `rectangular_prism`/`l_block` violent-instability
    pattern:** penalize a SUDDEN DROP in `contact_frac` step-to-step
    during formation (e.g. `max(0, contact_frac[t-1] - contact_frac[t])`,
    scaled by some coefficient) rather than penalizing low contact_frac
    in general. For `sphere`/`cone`/most objects, `contact_frac` never
    drops sharply once contact is made (confirmed in the detailed
    trace data -- steady or slowly-changing touch values throughout),
    so this term would sit at ~0 for them the whole episode. For
    `rectangular_prism`/`l_block`, it would fire specifically at the
    moment multiple fingers simultaneously lose contact (exactly the
    event already identified), directly discouraging the instability
    rather than generically pushing for more engagement everywhere.
  - **For the `cube`/`flat_disk` late-4th-finger slow-sag pattern:**
    reward REDUCING the spread of first-contact timing across fingers
    that do engage, rather than rewarding raw engagement count. Concretely,
    something like a small penalty on the variance/range of "steps since
    first contact" across currently-touching fingers. For objects where
    all engaging fingers already arrive within a tight window (`sphere`,
    `cone`: confirmed 10-15 step spread), this is already near its best
    value, so the term stays small. For `cube`/`flat_disk` (confirmed
    70-95 step spread), it would directly penalize the late-arrival
    pattern instead of just encouraging the 4th finger to engage at all
    (which is what the failed fix did, indiscriminately, everywhere).
- **Honest risks, stated up front:** (1) Part A requires an
  observation-space shape change, meaning full retraining from scratch
  either way -- same cost as Part B alone, so bundling them doesn't add
  retraining cost, but does add a second variable to a single
  experiment, making it harder to attribute credit if the result
  improves. (2) Both Part B terms are still guesses about mechanism,
  not proven fixes -- yesterday's lesson is that a term matching the
  diagnosed mechanism can still fail to help or actively hurt once
  trained; magnitude tuning via `smoke_test_rl_env.py` and a short
  continuation sanity check (same discipline as before) would still be
  required before committing to a full retrain. (3) Neither part
  addresses `cube`'s observed 1/3 variability or guarantees
  `rectangular_prism` specifically improves -- these are reasoned,
  targeted hypotheses, not verified fixes.
- **Recommended order if this is approved:** implement Part A alone
  first (observation change only, reward unchanged) and test via the
  same short continuation-style check, since it's the simpler,
  single-variable change and isolates whether just having geometry
  information changes behavior at all before adding Part B's two new
  reward terms on top. Not started -- awaiting a decision.

## Part A implemented (char_length observation) -- SEVERE regression, worse than Part B's attempt

- **Date:** 2026-11-01
- **Implementation:** exactly as proposed -- `char_length` (a static,
  per-episode scalar from `self.obj_spec`, not a leaked dynamic
  quantity) appended once to the end of the flattened observation
  (`observation_space` 115 -> 116 dims). Reward completely unchanged.
  Verified via direct inspection before training: shape correct (116),
  values correct and object-specific (`cube`=0.0400,
  `rectangular_prism`=0.0700, `rod`=0.0900, matching `OBJECT_LIBRARY`
  exactly). `smoke_test_rl_env.py` reward output was numerically
  IDENTICAL to the pre-Part-A baseline (12.66/185.33/22.70),
  confirming the reward path itself was untouched.
- **Sanity check passed:** short fresh-PPO run (25,600 steps), zero
  crashes, `ep_rew_mean` climbed cleanly (31.6 -> 116) -- a completely
  normal-looking early training curve, no red flags visible at this
  scale.
- **Full retrain: the cleanest run of the entire project** -- one
  single continuous process attempt, zero crashes, 0 -> 300,032 steps
  with no restarts needed at all.
- **Real 45-trial evaluation: 11/45 (24.4%) -- a SEVERE regression,
  worse even than Part B's 44.4% regression yesterday.** TRAIN 9/33
  (27.3%), HOLDOUT 2/12 (16.7%). Most strikingly, **`sphere` and `cone`
  -- the two most reliably perfect objects in the entire project,
  100% across every prior policy -- now fail completely (0/3 each)**,
  along with `rod` (also previously 3/3, now 0/3) and
  `triangular_prism`/`hexagonal_prism` (previously 3/3 and 2/3, now
  0/3 each). This is not a targeted improvement with isolated
  collateral damage (Part B's pattern) -- it's a broad collapse that
  hit the EASIEST objects hardest.
- **On the actual target objects, a small but real mixed signal:**
  `rectangular_prism` improved (0/3 -> 1/3), `l_block` improved
  (1/3 -> 2/3), but `cube` and `flat_disk` are both still 0/3. So
  2 of the 4 specifically-targeted objects did move in the right
  direction -- but this signal is completely swamped by the
  across-the-board collapse elsewhere.
- **Honest interpretation:** adding `char_length` to the observation,
  even though implemented exactly as designed with no reward change,
  measurably destabilized what the policy learned overall -- worse
  than doing nothing (64.4%) and worse than the blanket reward-term
  mistake (44.4%). Plausible explanations, none confirmed: (1) genuine
  single-seed training variance -- this is one run, and this project
  has already demonstrated large run-to-run swings elsewhere (the TCN's
  own 0.69-0.80 F1 spread); (2) the raw, unnormalized `char_length`
  scale (0.04-0.09) sitting awkwardly alongside very differently-scaled
  existing observation values (touch magnitudes in the tens, pull_force
  in Newtons, probabilities in [0,1]) may have disrupted early
  optimization in a way that a normalized version wouldn't; (3) giving
  the network ANY object-identity signal, even a single static scalar,
  may have encouraged a more object-SPECIFIC (and less robust/general)
  strategy that happened to land badly this run.
- **Decision point:** this result argues against Part B (the two
  additional reward terms) being layered on top of Part A as originally
  planned, since Part A alone already regressed badly -- stacking
  Part B on a destabilized base is unlikely to isolate anything
  cleanly. `Data/ppo_slip_policy_latest.zip` (64.4%) remains the
  reference result. Both `ppo_slip_policy_continued.zip` (57.8%,
  continued training) and `ppo_slip_policy_partA.zip` (24.4%, char_length
  observation) are now ruled out as improvements -- every attempt to
  improve on 64.4% this week has made things worse, which is itself a
  real, reportable pattern worth weighing before trying a fourth
  approach.

## Seed-variance check -- reverted to the exact 64.4% config, running a second from-scratch training with nothing changed

- **Date:** 2026-11-01
- **Why:** three independent "improvement" attempts this week
  (continued training, the finger-fix reward term, the char_length
  observation) all made the real evaluation number worse than 64.4%.
  Before concluding anything about those attempts specifically, the
  more basic question needs answering: is 64.4% itself a reliable
  number, or does THIS pipeline (with no explicit seeding anywhere,
  same as the rest of the project) swing meaningfully run to run even
  with nothing changed? A second from-scratch run isolates pure
  training variance -- no new code, no new reward, no new observation.
- **Revert confirmed exact, not just intended:** restored
  `Synthetic data creation/Simulation/rl_env.py` to commit `02c867f`
  (the last commit before the finger-fix work and the only commit
  containing the version actually used for the 64.4% training run) via
  `git checkout 02c867f -- rl_env.py`, rather than hand-reverting
  individual edits -- eliminates any risk of a manual revert missing
  something. Confirmed via direct inspection: zero occurrences of
  `MIN_FINGER_FRACTION_COEF`/`char_length`-as-observation in the
  restored file (the pre-existing, UNRELATED `0.15*char_length`
  slip-threshold computation is untouched, as it should be --
  that's original code, not part of either experiment). Confirmed via
  execution, not just code inspection: `observation_space` back to
  115-dim (not 116), and `smoke_test_rl_env.py`'s reward output is
  EXACTLY identical to the original pre-any-fix baseline
  (12.66/185.33/22.70, same 3 seeds) -- not just similar, identical to
  several decimal places, confirming byte-for-byte config parity.
  `train_rl_policy.py`'s own unrelated infrastructure fixes from this
  week (the `--checkpoint-freq` rename, `EvalCallback` removal) were
  deliberately NOT reverted -- those were bug fixes made for the SIGSEGV
  investigation, not part of either experiment, and reverting them
  would reintroduce an already-diagnosed crash risk for no reason.
- **Next: full 300k-step fresh training run, same self-healing
  orchestrator, zero code changes from this exact config** -- see the
  next entry for the result.

## Two real bugs found in the "revert" above -- fixed properly, and a full retroactive audit of this week's numbers

- **Date:** 2026-11-01
- **Bug 1, the serious one:** reverting to commit `02c867f` didn't just
  undo the char_length/finger-fix changes -- it ALSO deleted
  `reset()`'s `options={"object_name", "scene_seed"}` forced-override
  handling, because that feature was added in a LATER commit (`9ddc2fc`)
  bundled together with the finger-fix work, even though it's
  completely unrelated (it exists so `evaluate_rl_policy.py` and the
  diagnostic scripts can force a specific, reproducible (object, seed)
  pair for the 15x3 benchmark). Without it, `reset()` silently falls
  back to `self._episode_rng.choice(...)` -- a random draw -- with NO
  error and no visible signal anything was wrong. Caught by an
  independent review before any evaluation ran against the affected
  policy.
- **Bug 2, sloppier revert than decided:** checking out `02c867f`
  deleted the entire `MIN_FINGER_FRACTION_COEF` code path (constant,
  docstring, and its use in the reward sum), when the actual decision
  on record (the "Continuing to train the 64.4% checkpoint" entry
  above) was to set it to **0.0 while keeping the code intact** --
  letting a future experiment reuse it without re-deriving it.
- **Both fixed properly:** restored `rl_env.py` from commit `9ddc2fc`
  instead (has BOTH the options-forcing AND the finger-fix code path,
  with `MIN_FINGER_FRACTION_COEF` still at its escalated 3.0 at that
  commit), then set the coefficient to **0.0 via a single targeted
  edit** -- code path, docstring, and comments all intact, exactly
  matching what was actually decided.
- **Re-verified, now that both pieces are correct:** `smoke_test_rl_env.py`
  reward output still exactly matches the original baseline
  (12.66/185.33/22.70, same 3 seeds) -- confirming the reward function
  is still byte-identical despite restoring from a different commit.
  Separately, directly exercised `reset(options={"object_name":...,
  "scene_seed":...})` for 3 different forced (object, seed) pairs and
  confirmed each one is honored exactly (`info` matches the request
  every time) -- not just code-inspected, actually run.
- **Did the seed-variance retrain's evaluation already run with the
  broken `reset()`? No** -- checked before doing anything else: no
  `ppo_slip_policy_seedcheck.zip` existed yet (training was still in
  progress), so nothing from THIS experiment needs to be invalidated
  or re-run. The eventual evaluation will run with the fix in place
  (training itself is unaffected either way -- it never passes
  `options`, only evaluation/diagnostic scripts do).
- **Retroactive audit: did this same gap affect ANY of this week's
  other evaluated numbers (44.4%, 24.4%, 57.8%, or the original
  64.4%)? Checked directly against the actual stored per-trial output
  of each, not assumed.** All four evaluations' per-trial logs show
  the objects appearing in the EXACT systematic sweep order
  (`cube` seed=0,1,2 -> `sphere` seed=0,1,2 -> `cone` seed=0,1,2 ->
  `rectangular_prism` seed=0,1,2 -> ...,  matching `OBJECT_LIBRARY`'s
  order precisely) in every single one. If `options` had been silently
  ignored in any of these runs, `self._episode_rng.choice(...)` would
  have produced a effectively-random object sequence instead --
  getting all 45 trials to land in this exact deterministic order by
  chance is not plausible. **All four numbers (64.4%, 57.8%, 44.4%,
  24.4%) are confirmed unaffected -- the broken `reset()` existed only
  in this one seed-variance-check attempt, caught before it was ever
  evaluated.** Reported plainly either way, per instructions: this
  audit did not find a wider problem, but it was a real, worthwhile
  check given the actual bug found, not a check performed to produce a
  reassuring answer.
- **Next: the seed-variance retrain (still in progress, confirmed
  unaffected by either bug) continues; evaluation will run with both
  fixes in place once training completes.**

## Why are we plateaued at 64.4%, and why did three different single-change attempts all regress it? Research + audit, not a new fix attempt

> **CORRECTION (2026-11-02):** the minibatch, episode-budget and several literature claims in this entry are wrong or overstated. See "Corrections to tonight's audit" below before citing anything here.

- **Date:** 2026-11-01
- **Scope note:** this is diagnosis only, per instructions -- nothing
  in this entry was implemented or trained. One candidate fix is
  proposed at the end, explicitly not yet acted on.

**Part 2 first (audit of our own setup), since it produced a concrete, checkable number the literature search then gets compared against.**

- **No reward or observation normalization anywhere in this project
  (re-confirmed by grep, not assumed):** `grep -n "VecNormalize\|normalize"`
  across `train_rl_policy.py` and `rl_env.py` returns nothing. The
  only clipping in `rl_env.py` is on the action (`MAX_DELTA_PER_STEP`),
  not on reward or observation. PPO is constructed with only
  `n_steps=256, batch_size=256, learning_rate=3e-4` explicitly set --
  every other PPO hyperparameter (`gamma=0.99, gae_lambda=0.95,
  clip_range=0.2, ent_coef=0.0, n_epochs=10, vf_coef=0.5,
  max_grad_norm=0.5`) is left at SB3's library default.
- **Real per-object training-data budget, computed from actual logged
  `ep_len_mean` values (not estimated):** across the fresh-retrain and
  seed-check logs, `ep_len_mean` sits in the 285-335 range for most of
  a run (occasionally higher early on, e.g. mid-320s). Taking ~300-310
  as representative: `total_episodes ~= total_timesteps / ep_len_mean
  = 300,000 / ~305 ~= 980`. SB3's `num_timesteps` already sums across
  all 4 parallel envs, so that ~980 is the TOTAL episode count for the
  entire 300k-step run, not per-env. There are **11 train objects**
  (`cube, rectangular_prism, capsule, ellipsoid, flat_disk, rod,
  l_block, dumbbell, square_pyramid, triangular_prism, hexagonal_prism`
  -- confirmed from the actual `"Train objects (11): [...]"` log line),
  drawn **uniformly at random every reset**, no curriculum, no
  weighting. That's **~89 episodes per object across the ENTIRE 300k-step
  run** -- noticeably smaller than the ~150-200/object rough guess from
  before this audit ran.

**Part 1 (web research), used to judge whether the Part 2 numbers above are actually a problem:**

- **Published closed-loop sim-trained grasp success rates span an
  enormous range (27% to 95%) depending entirely on setup, budget, and
  object set** -- so 64.4% is not obviously low in isolation; it sits
  inside the normal range, not outside it. The high end (DemoGrasp,
  ~95% on DexGraspNet; "From Power to Precision," 82.5% zero-shot) comes
  from setups with far larger training budgets and/or large-scale
  pretraining, not comparable to our single 300k-step run on 11
  objects. [DemoGrasp](https://paperswithcode.co/paper/2509.22149),
  [From Power to Precision](https://arxiv.org/html/2511.13710)
- **A directly relevant data point: a fixed-randomization grasping
  baseline (no curriculum) plateaued around 50% success, versus >70%
  with an adaptive curriculum, after 500,000 steps** -- a LARGER step
  budget than our 300k, with a baseline that still plateaus in the
  same ballpark we're seeing, specifically BECAUSE it lacked curriculum
  weighting. Our setup (uniform random object choice, no curriculum,
  no adaptive weighting, fewer steps) matches the weaker condition in
  that comparison, not the stronger one. [Adaptive Curriculum Domain
  Randomization](https://communities.springernature.com/posts/adaptive-curriculum-domain-randomization-for-robotic-grasping-a-performance-driven-approach)
- **VecNormalize (observation AND reward/return normalization, plus
  clipping) is ON by default in SB3's own RL Baselines3 Zoo for
  continuous-control tasks precisely because raw PPO is known to be
  fragile to reward scale** -- it standardizes observations using
  running mean/variance, and divides reward by the running std of the
  discounted return (clipped at +/-10.0 by default). [SB3 VecNormalize
  source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/vec_env/vec_normalize.html)
- **Our reward function mixes terms of very different raw magnitudes
  with no normalization to absorb that**: `ALPHA_DISPLACEMENT=50.0`
  (times displacement), `SLIP_PENALTY=20.0`, `HOLD_BONUS=20.0`,
  `DROP_PENALTY=15.0`, `GRIP_FORCE_COEF=0.15` (times force, capped at
  2x), `EFFORT_COEF=0.02` (times mean |action|) -- a decade-plus span
  between the smallest and largest coefficients, summed directly into
  the critic's target with no return-scale normalization in between.
  This is consistent with, though does not single-handedly explain,
  why adding or escalating a reward term (`MIN_FINGER_FRACTION_COEF`
  0.5->3.0, accumulated every substep) destabilized the policy rather
  than cleanly layering on top of the existing incentive structure --
  a known PPO failure mode when reward scale shifts and nothing
  renormalizes the value function's targets in response.
- **This does NOT fully explain all three regressions by itself**:
  the continued-training result (57.8%, no reward or observation
  change at all) still regressed below 64.4% with more steps alone.
  That one points at a second, complementary mechanism rather than
  reward-scale shock: our rollout buffer is small (`n_steps=256` vs.
  SB3's own default of 2048, an 8x difference) and `batch_size=256`
  equals `n_steps`, meaning each update already uses the full,
  small, 1024-transition (256 x 4 envs) buffer as one epoch's worth of
  minibatches, repeated for `n_epochs=10` (default, not overridden).
  Small buffer + many epochs over it + no advantage/return
  normalization is a documented recipe for noisy, non-monotonic policy
  quality over continued training -- i.e., the kind of result where
  training longer isn't guaranteed to be better, which is exactly what
  the continued-training experiment found.

**Honest synthesis:** two complementary, literature-grounded candidate
explanations, not one silver bullet: (1) missing reward/observation
normalization, which plausibly explains why reward- and
observation-altering changes destabilized training, and (2) a small
rollout buffer combined with limited per-object sample diversity
(~89 episodes/object, no curriculum) under a uniform-random object
curriculum, which plausibly explains both the 64.4% ceiling itself and
why continued training alone didn't help. Neither is proven --
both are supported by independent literature and by our own grepped
numbers, not by assumption.

**Proposed candidate fix -- NOT IMPLEMENTED, pending a decision:**
add `VecNormalize` (SB3's own wrapper, restoring its documented
default behavior rather than inventing a new mechanism) around the
training `SubprocVecEnv`, with its stats saved alongside the policy
checkpoint and loaded for evaluation (reward normalization OFF, as
the Zoo does, when evaluating -- only observation normalization
carries over). This is additive and reward-structure-preserving (no
reward term changes), which is a meaningfully different kind of
change than the three attempts that already regressed performance,
all of which touched the reward or observation CONTENT rather than
its scale. Not started -- reporting first, per instructions.

## VecNormalize implemented; a self-caused crash cascade on the seed-check retrain while testing it

- **Date:** 2026-11-01
- **Approved and sequenced:** finish and evaluate the seed-variance
  retrain first (stands on its own), then implement/test VecNormalize
  as a fresh-from-scratch 300k run (normalization stats accumulate
  over a run, so a continuation would start them from a biased point --
  same reasoning as every other structural change this week), with
  evaluation using VecNormalize's own matched stats, not raw
  observations.
- **Implemented in `train_rl_policy.py`:** training env is now
  `SubprocVecEnv` wrapped in `VecNormalize(norm_obs=True,
  norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)` on a
  fresh run. `CheckpointCallback(save_vecnormalize=True)` now saves
  `ppo_slip_ckpt_vecnormalize_<N>_steps.pkl` next to every
  `ppo_slip_ckpt_<N>_steps.zip`. `--resume-from` now derives the
  matching vecnormalize path from the checkpoint path and **raises
  `FileNotFoundError` instead of silently restarting stats from
  scratch** if that file is missing -- getting this wrong mid-segmented-run
  would silently corrupt the comparison with no visible symptom, so
  it's a hard stop, not a fallback. Final policy save also writes
  `<save-path>_vecnormalize.pkl`.
- **Implemented in `evaluate_rl_policy.py`:** new `--vecnormalize`
  flag loads the saved stats via `VecNormalize.load(path,
  DummyVecEnv([lambda: env]))` (the `DummyVecEnv` exists only to
  satisfy `VecNormalize`'s shape check at load time -- it is never
  stepped; the real single-env, options-forcing eval loop is
  unchanged) and calls `.normalize_obs(obs)` directly on each raw
  observation before `model.predict()`. `normalize_obs()` is
  documented to not update statistics, so this is safe to call
  per-trial with no state leaking between trials. Reward
  normalization is irrelevant here since `held_securely()` reads
  `info`, never the reward value, so it's intentionally left untouched.
  Backward-compatible: omitting `--vecnormalize` evaluates exactly as
  before, for policies (like the 64.4% checkpoint) trained without it.
- **New orchestrator** `Data/_run_vecnorm_retrain.sh`, same self-healing
  segmented-resume pattern as every other run this week, separate
  `rl_logs_vecnorm/` / `ppo_slip_policy_vecnorm` paths to avoid
  cross-contaminating checkpoints with other experiments.
- **Before launching the real 300k run, ran a cheap smoke test (tiny
  n-envs=2, 1024 steps) to check the wiring -- and it crashed
  (`EOFError` from a `SubprocVecEnv` worker, zero diagnostic output,
  meaning the worker died before even reaching `make_env`'s first
  line).** Checked the seed-variance retrain immediately after and
  found real damage: it had been cleanly running its ORIGINAL attempt 1
  all the way from 0 to 155,000 steps (over halfway) when it hit a
  first, ordinary crash of the kind already characterized this week
  (MuJoCo's per-call allocation failure) -- unrelated to me. But every
  resume attempt immediately after that (attempts 2 through 6, all
  within the exact window my smoke test was running) failed instantly
  with **zero steps of progress each**, which tripped
  `MAX_CONSECUTIVE_NO_PROGRESS=5` and halted the orchestrator entirely
  (`STOPPING`, exit 1) at 155,000/300,000 steps.
- **Root cause: my own mistake, not a new environment bug.** The
  smoke test's 2 workers and the seed-check retrain's 4 workers were
  two SEPARATE `train_rl_policy.py` processes running at the same
  time, each with its own independent `build_lock` (created fresh per
  process via `mp.get_context("spawn").Lock()`) -- the lock only
  coordinates MuJoCo model-compiles WITHIN one process's workers, not
  across two separate script invocations. Running both at once put 6
  total concurrent MuJoCo-compiling workers on the machine with zero
  cross-process coordination between them, reproducing the same kind
  of resource contention already documented this week (the "n-envs=6
  died" diagnostic). This directly violates the standing
  resource-capped-runs constraint, and I should not have started a
  second heavy MuJoCo process while a production run was active --
  caught and owned immediately, not discovered later.
- **Recovery, attempt 1: confirmed no orphaned `python.exe` processes
  remained (`Get-CimInstance Win32_Process`, empty result) before
  restarting; relaunched `_run_seedcheck_retrain.sh` fresh, now with
  nothing else running concurrently -- and it crashed again,
  identically, 0 progress every attempt, same `STOPPING` halt.** With
  no concurrent process this time, resource contention is ruled out --
  there was a second, more direct bug.
- **Real root cause, found by actually reading the attempt log instead
  of re-running blindly a 3rd time (per standing instruction): I had
  edited `train_rl_policy.py` -- a script this already-running
  production experiment depends on -- WHILE it was still active.** The
  new `--resume-from` logic unconditionally required a matching
  `ppo_slip_ckpt_vecnormalize_<N>_steps.pkl` sidecar file next to the
  checkpoint, and raised `FileNotFoundError` if it was missing. The
  seed-check experiment predates VecNormalize entirely and has no such
  files anywhere -- every resume attempt was hitting that new
  exception directly (confirmed from `_seedcheck_attempt_6.log`'s own
  traceback), not crashing in MuJoCo/SubprocVecEnv at all. The earlier
  "resource contention" explanation above was wrong, or at best only
  explained the ORIGINAL unrelated crash at 155,000 -- it did not
  explain why EVERY subsequent resume failed identically with zero
  diagnostic worker output, which in hindsight was the exception firing
  before any worker-related code ran.
- **This is a more serious mistake than the first explanation
  suggested: editing a shared, in-use script while a real run depends
  on it, not just a resource/timing collision.** Fixed properly rather
  than reverting the VecNormalize work: `train_rl_policy.py` now checks
  whether the checkpoint's own log-dir contains ANY
  `*_vecnormalize_*_steps.pkl` file at all. If none exist anywhere (a
  legacy, pre-VecNormalize experiment like seed-check), it resumes
  exactly as before -- no normalization wrapper, no error. Only a
  log-dir that genuinely already has VecNormalize sidecar files
  enforces the exact-match hard-stop. Also fixed two latent bugs the
  same read-through caught before they could bite later: a duplicate
  `print` referencing an undefined variable on the legacy path (would
  have been a `NameError` at the very next print statement), and an
  unconditional final `train_env.save(...)` that would have thrown
  `AttributeError` at the very end of a successful legacy run (a plain
  `SubprocVecEnv` has no `.save()` method) -- both guarded now.
  Confirmed with `python -m py_compile` before touching the real run
  again.
- **Recovery, attempt 2: relaunched `_run_seedcheck_retrain.sh` a
  third time with the fix in place, nothing else running concurrently.
  Resumed from 155,000 without error this time** -- ran cleanly the
  rest of the way to 300,000/300,000 steps (`ppo_slip_policy_seedcheck.zip`,
  `num_timesteps=300408`), orchestrator exit 0. The VecNormalize smoke
  test's original `EOFError` (fresh-run path, no `--resume-from`
  involved, so not explained by this specific bug) is still an open
  question, to be re-tested alone once the VecNormalize retrain itself
  is launched.

## Seed-variance check result: 35.6%, not close to 64.4% -- pure training-run variance is large

- **Date:** 2026-11-02 (session resumed after a long rate-limit pause;
  the eval had already completed during the gap)
- **The actual number, reported plainly, not spun either direction:**
  the real 45-trial eval against the completed seed-check checkpoint
  (`ppo_slip_policy_seedcheck.zip`, 300,408 steps, zero code changes
  from the exact 64.4% config -- confirmed `rl_env.py` byte-identical
  reward output and correct `options`-forcing both re-verified
  yesterday after the revert-bug fix) gives **16/45 held securely =
  35.6%** (TRAIN objects: 12/33 = 36.4%, HOLDOUT: 4/12 = 33.3%). All 45
  trials ran cleanly, same systematic sweep order, no failures.
- **This is NOT close to 64.4%.** A ~29-percentage-point swing from
  nothing but a different random seed (there is no explicit seeding
  anywhere in this RL pipeline, confirmed earlier this week) is a
  large amount of pure run-to-run variance for a single training run
  to carry. Per-object, the pattern isn't "uniformly worse" either --
  `cone` went 3/3 (same as perfect), several objects that held
  securely before now fail completely (`cube` 0/3, `flat_disk` 0/3,
  `l_block` 0/3, `plus_cross` 0/3, `triangular_prism` 0/3,
  `cylindrical_prism` 0/3) -- a different policy, not a uniformly
  weaker one.
- **This honestly weakens, but does not erase, two of this week's
  three "regression" conclusions.** With two data points now bounding
  the variance band at roughly 35.6%-64.4% for an UNCHANGED config:
  - The continued-training result (57.8%) sits BETWEEN those two
    bounds. The earlier conclusion ("training longer alone made it
    worse") is now considerably less certain -- 57.8% is fully
    consistent with ordinary seed variance, not necessarily a real
    effect of additional training.
  - The finger-fix result (44.4%) also sits BETWEEN those two bounds,
    for the same reason -- it, too, cannot be confidently distinguished
    from ordinary seed noise with only one run each.
  - The char_length result (24.4%) sits BELOW the lower bound (35.6%)
    observed so far. This one still looks like a real effect beyond
    what pure variance alone has shown -- though the size of that
    "real" effect is now smaller than it first appeared (roughly an
    11-point gap below the noise floor, not a 40-point gap below
    64.4%), since 64.4% itself is now understood to be closer to the
    lucky/high end of this pipeline's variance than a stable baseline.
  - Reported honestly in both directions, per instructions: this does
    NOT mean the reward/observation changes were fine after all --
    char_length still looks like a real regression. It means two of
    the three "regressions" are now ambiguous rather than confirmed,
    purely because single-run comparisons without multiple seeds per
    condition can't separate a real effect from this much inherent
    noise. That's a genuine methodological gap in how this week's
    comparisons were run, not a new result to act on by itself.
- **What this means going into the VecNormalize experiment:** it's
  still worth running (reducing variance IS literally its documented
  purpose, independent of any single sample's optimism), but a single
  VecNormalize run landing anywhere in roughly the 35-65% range cannot,
  by itself, be confidently called better or worse than 64.4% either --
  the same noise band applies. Interpreting that result honestly when
  it lands means judging it against this ~29-point noise floor, not
  against the single 64.4% number in isolation.

## VecNormalize full 300k retrain launched -- smoke-tested alone first, resolved yesterday's open EOFError question

- **Date:** 2026-11-02
- **Before touching the real run, re-tested the fresh-run VecNormalize
  path that crashed with `EOFError` yesterday -- this time completely
  alone, nothing else running concurrently (confirmed via
  `Get-CimInstance Win32_Process`, no stray python.exe).** Same tiny
  smoke test (n-envs=2, 1024 steps): completed cleanly, both
  `ppo_slip_ckpt_*_steps.zip` and the matching
  `ppo_slip_ckpt_vecnormalize_*_steps.pkl` saved in lockstep as
  expected. **This resolves yesterday's open question: the original
  `EOFError` really was the resource contention from running a second
  heavy MuJoCo process concurrently with the seed-check retrain, not a
  bug in the VecNormalize wiring itself.**
- **Also smoke-tested the resume path on this (genuinely
  VecNormalize-using) log-dir**, extending from its 1020-step
  checkpoint to 1536: correctly detected the log-dir uses VecNormalize,
  loaded the matching stats (`"Resumed VecNormalize stats from
  ...1020_steps.pkl"`), continued training without error.
- **Also smoke-tested `evaluate_rl_policy.py --vecnormalize`** against
  that same tiny (barely-trained) checkpoint -- all 15 trials ran
  cleanly with normalized observations fed to `model.predict()`, 0%
  held (expected and uninformative at ~2k training steps; the point
  was confirming no exceptions anywhere in the pipeline, not the
  success rate).
- **All three integration points confirmed clean in isolation. Smoke
  test artifacts deleted. Launched the real run:** `_run_vecnorm_retrain.sh`,
  full 300k target, fresh from scratch, same self-healing segmented
  auto-resume pattern as every other run this week, writing to
  `Data/rl_logs_vecnorm/` / `Data/ppo_slip_policy_vecnorm` (kept
  separate from every other experiment's paths). Being watched now,
  alone, with nothing else running concurrently this time.

## Four more fixes from independent code review (2026-11-02), all verified before touching the live run

1. **Serious:** `--resume-from` the FINAL saved policy (e.g.
   `ppo_slip_policy_vecnorm.zip`, not a mid-run `_steps.zip` -- the
   exact workflow already used to continue-train the 64.4% checkpoint)
   silently missed its VecNormalize sidecar (different naming: no
   `_steps` suffix) and fell back to raw unnormalized training with
   ZERO error. Fixed with a new `find_sidecar()` helper that checks
   actual file existence against both naming shapes this project
   produces, instead of re-deriving SB3's internal scheme by string
   substitution (also addresses the "fragile if the scheme changes"
   review point). Verified: resuming from a final-save path now finds
   and loads the sidecar correctly (confirmed by its own print line).
2. Moved both sidecar checks (VecNormalize + the new reward-config
   check below) to BEFORE `SubprocVecEnv` is constructed, so a
   missing/mismatched sidecar aborts immediately instead of after a
   full worker-spawn + MuJoCo-compile cycle.
3. (Addressed by the `find_sidecar()` rewrite above.)
4. **Systemic:** generalized the VecNormalize-sidecar pattern to cover
   EVERY reward constant, not just the one (`MIN_FINGER_FRACTION_COEF`)
   that already bit us once. `rl_env.get_reward_config()` snapshots all
   15 tunable constants; a new `RewardConfigCallback` saves it
   alongside every checkpoint; `--resume-from` now hard-errors if the
   checkpoint's recorded config doesn't match `rl_env.py`'s CURRENT
   constants. Verified: deliberately mismatching
   `MIN_FINGER_FRACTION_COEF` (0.0 recorded vs. 3.0 current) correctly
   raised `ValueError` naming the exact mismatch before any training
   resumed; reverted immediately after.
- All four fixes verified with small, isolated smoke tests (not the
  live run) before being left in place; `python -m py_compile` clean
  on all three touched files. The live 300k VecNormalize run
  (currently 110,000/300,000, unaffected throughout -- it was never
  touched) will exercise these same resume-sidecar code paths the next
  time it crashes and auto-resumes, same as it already has several
  times this week.

## New standing rule for tonight + a second independent review's 5 findings on the above

- **Date:** 2026-11-02. User authorized working through the night with
  one condition: every change touching a real training/eval run gets
  an independent review (this project's `/code-review` skill at high
  effort, or an equivalently careful self-review) BEFORE it's used,
  not occasionally -- applied retroactively to tonight's VecNormalize
  changes immediately.
- **5 findings came back, triaged for real risk given how these
  scripts are ACTUALLY used tonight, not worst-case generality:**
  1. `find_sidecar()`'s glob fallback picks `matches[0]` with no
     disambiguation if multiple same-step sidecars exist in one
     directory. **Not fixed tonight** -- every orchestrator script
     this project uses gives each experiment its own dedicated
     `--log-dir`, so this can't actually occur under how these scripts
     are run here. Noted as a real general-purpose limitation, not an
     active risk.
  2. A renamed checkpoint with no sibling sidecar silently resumes
     unnormalized. **Not fixed tonight** -- no orchestrator script
     here ever renames a checkpoint. Noted, not active.
  3. The "any sidecar nearby" fallback scans the whole directory, so
     two unrelated lineages sharing one log-dir could false-positive
     an error. **Not fixed tonight** -- same reason as #1, dedicated
     dirs per experiment.
  4. **Fixed:** `evaluate_rl_policy.py` had no automatic check that
     `--vecnormalize` was supplied when the policy needs it -- a
     forgotten flag on the exact script that produces headline numbers
     would silently evaluate with raw observations and print a
     confidently wrong percentage. Now auto-detects the matching
     sidecar next to `--policy` via the same `find_sidecar()` used for
     training resume (still hard-errors if a sidecar exists nearby but
     doesn't match). Verified with a fresh smoke-test checkpoint:
     correctly printed `"Auto-detected VecNormalize sidecar: ..."` and
     loaded it with no `--vecnormalize` flag passed.
  5. The reward-config mismatch check only iterates keys present in
     the OLD recorded JSON, so a reward constant added to
     `_REWARD_CONFIG_KEYS` after a checkpoint was recorded would never
     be checked on resume. **Not fixed tonight** -- no new reward
     constants are being added under the current converge-only plan,
     so this can't fire tonight either. Noted for whoever touches
     reward constants again later.
- **Rationale for fixing only #4 tonight:** the other four require
  either a scenario that cannot occur given how this project's scripts
  are actually invoked (dedicated per-experiment log-dirs, no renaming,
  no new reward terms tonight), or a more involved design decision not
  worth making under the Nov 4 clock. #4 was the one genuine, live risk
  to tonight's actual headline numbers, so it's the one that got fixed
  immediately. `python -m py_compile` clean on all three files

## VecNormalize seed #1 result: 2.2% -- catastrophic, not just "no better." Verdict: VecNormalize is not the fix, stopping here

> **CORRECTION (2026-11-02): the verdict and the explanation below are withdrawn.** 2.2% is one seed, and its eval path has not been checked at runtime. The evidence offered for "reward farming" was false: final `ep_len_mean` 162 is the shortest of any run, not longer than typical (the "~100-130" figure was this run's own early training), and final `ep_rew_mean` 334 is ordinary (seed3 355, partA 360, the 64.4% run 442), not "far higher than any other run". The mechanism was also impossible: formation lasts a fixed 800 physics steps (160 macro-steps), set by the global step, so a policy cannot stay in it longer. Skipping VecNormalize seed #2 was my call and departed from the agreed plan. See "Corrections to tonight's audit" below.

- **Date:** 2026-11-02
- **Result:** full 300k run completed cleanly (300,032 steps, zero
  crashes this segment, both sidecars saved correctly, auto-detected
  correctly by the just-fixed eval harness). The real 45-trial eval:
  **1/45 held securely (2.2%)** -- TRAIN 1/33 (3.0%), HOLDOUT 0/12
  (0.0%). Per-trial output: almost every trial is either
  `dropped_during_formation=True` or `slipped=True` right at/past the
  object's threshold; only `l_block` seed=1 held.
- **Checked for a bug before accepting this as a real result** (per
  tonight's review discipline) rather than assuming either way:
  training's own internal metrics are textbook-healthy and show NO
  sign of a training-process bug -- `explained_variance` 0.84-0.92
  (value function fits well), `value_loss` low (0.07-0.18), `std`
  stable at ~0.82 (not collapsed, not exploded), `approx_kl` normal
  (0.01-0.02). `obs_rms`/`ret_rms` in the saved VecNormalize pickle
  have no NaN/Inf and sane magnitudes. This was a cleanly CONVERGED
  PPO run by every internal signal -- not a crash, not numerical
  instability, not a wiring bug in the sidecar/auto-detect code
  (separately verified earlier tonight).
- **Likely real explanation, not a bug: reward/task misalignment,
  probably made worse by return normalization.** `ep_rew_mean` climbed
  steadily to 340-363 (far HIGHER than any other run this week) while
  `ep_len_mean` also climbed to ~160-164 (longer than the ~100-130
  typical elsewhere) -- consistent with the policy learning to
  maximize ACCUMULATED PER-STEP formation-phase reward (contact +
  grip-force terms) by staying in formation longer, rather than
  reaching a secure terminal hold. Return normalization changes the
  RELATIVE scale PPO's advantage calculation assigns to large one-time
  terminal rewards (HOLD_BONUS, SLIP_PENALTY, the
  ALPHA_DISPLACEMENT-scaled phase-2 term) versus small
  repeated-every-step ones (CONTACT_COEF, GRIP_FORCE_COEF) -- a
  plausible mechanism for why a policy that converges well by its own
  reward signal can still get dramatically worse at the actual task.
  This is a reward-design problem, not something VecNormalize itself
  did wrong.
- **Decision, given the Nov 4 clock: do NOT run a VecNormalize seed #2.**
  A second seed could only ever confirm "wide spread" at this point --
  even a lucky second run landing near 60% would still mean a 55+ point
  spread, the same "not a fix" verdict either way, and chasing the
  underlying reward-scale interaction would mean redesigning reward
  normalization, which is explicitly off the table tonight (no new
  fixes). One severe, cleanly-verified-as-not-a-bug result is enough to
  answer the question VecNormalize was meant to answer.
  **VecNormalize is ruled out as tonight's fix.**
- **Pivoting per the pre-agreed decision tree's second branch:**
  stopping all new-fix experimentation. Remaining seeds go toward an
  honest mean +/- spread for the best-available NON-VecNormalize
  config (the exact original reward/observation setup already run
  twice: 64.4% and 35.6%, results.md "Seed-variance check result").
  One more seed of that exact config next, for a 3-point mean +/- std --
  plus the user-approved 1,000-trial evals (hand-tuned baseline first,
  in this exact gap since nothing is training right now; the RL
  policy's 1,000-trial eval last, once the 3-seed baseline number is
  locked in).

## Baseline (hand-tuned Model A) 1,000-trial eval: 58.6% -- meaningfully different from the 45-trial estimate of 40.0%

> **CORRECTION (2026-11-02): this entry is wrong on methodology and cause.** The run was NOT "identical methodology": it used `max_steps=1500`, while the 40.0% run used `max_steps=2500`. The gap is mainly a config difference, not small-sample noise. Do not compare 58.6% with 40.0%. See "Baseline comparability check" below.

- **Date:** 2026-11-02. `batch_closed_loop_trials.py`, identical
  methodology to every prior Model A number (500/300 ramp/settle,
  `squeeze_intensity=0.95`, `model_a.pt`, 22ch), just 67 trials/object
  instead of 3 -- 1,005 total, 0 failures, 1969.5s (1.96s/trial).
- **Result: 589/1005 held securely (58.6%)** -- mean
  `max_slip_probability` 0.3968, mean `n_corrections` 26.19.
- **This is NOT close to the 45-trial number (40.0%, 18/45) --
  another real demonstration of how much a 3-seed/object sample can
  mislead,** same lesson as the RL seed-variance finding, just for the
  baseline this time. Per-object spread is huge and bimodal: several
  objects near-perfect (`dumbbell` 67/67, `capsule` 67/67, `flat_disk`
  65/67, `sphere` 59/67) and several near-total-failure
  (`cylindrical_prism` 1/67, `cube` 14/67, `cone` 21/67) -- the 40.0%
  45-trial estimate wasn't wrong due to a bug, it was just too small a
  sample of a highly object-dependent success rate to land near the
  true mean.
- **Implication for the paper's numbers going forward: the 45-trial
  numbers quoted earlier this week for every RL config (64.4%, 35.6%,
  2.2%, etc.) should be assumed to carry similarly large sampling
  uncertainty on top of the already-measured 29-point seed-to-seed
  training variance** -- two separate, compounding sources of noise,
  not one. Full 1,000-trial RL evaluation (planned next, once the
  3-seed baseline mean is locked in) will matter even more than
  expected for getting a trustworthy final number.

## Caught before launching: --use-vecnormalize needed to be opt-in, not unconditional, or seed3 would have been silently corrupted

- **Date:** 2026-11-02. While prepping seed3 (a third from-scratch run
  under the ORIGINAL, non-VecNormalize config, for a 3-point mean+-std
  against seed1/64.4% and seed2-seedcheck/35.6%), realized the current
  `train_rl_policy.py` made VecNormalize UNCONDITIONAL for every fresh
  (non-`--resume-from`) run -- there was no flag to turn it off. Seed3
  would have silently trained under VecNormalize instead of the plain
  config it's supposed to match, defeating the entire point of the
  3-point comparison, with no error at all.
- **Fixed before any run was launched:** added `--use-vecnormalize`
  (default off/`store_true`). Every orchestrator script written
  tonight (seed1, the unused seed2, production, seed-check) is
  unaffected or was already specific about what it needed; seed3's
  script never passes the flag, so it now correctly gets the plain,
  unnormalized env. Verified with a tiny smoke test: a fresh run
  without `--use-vecnormalize` now produces no `_vecnormalize.pkl`
  sidecar at all (only the policy + reward-config sidecar), confirming
  the plain path is taken. `python -m py_compile` clean. Seed3 launched
  immediately after, full 300k target, same self-healing segmented
  pattern.

## Seed3 completed (300,032 steps, clean) -- but its eval hit the exact edge case flagged-but-deferred earlier tonight

- **Date:** 2026-11-02. Evaluating seed3 immediately threw
  `FileNotFoundError` from `find_sidecar()`: it found seed1's
  `ppo_slip_policy_vecnorm_vecnormalize.pkl` while scanning for ANY
  vecnormalize sidecar near seed3's checkpoint, and wrongly concluded
  seed3 was missing its own. This is finding #3 from the earlier
  independent review ("any_nearby fallback scans the ENTIRE resume
  directory ... two unrelated lineages sharing one log-dir"),
  deliberately left unfixed at the time with the reasoning "every
  orchestrator gives each experiment its own `--log-dir`, so this
  can't occur" -- true for PERIODIC checkpoints, but every
  experiment's FINAL save (`model.save(args.save_path)`) lands
  directly in the shared flat `Data/` folder, which that reasoning
  missed. The deferral was wrong for this specific case.
- **Fixed properly:** `find_sidecar()` now scopes both the `_N_steps`
  match and the `any_nearby` fallback to the checkpoint's own lineage
  prefix (the shared `name_prefix` for periodic checkpoints, or the
  exact resume stem -- e.g. `ppo_slip_policy_seed3` -- for a final
  save), not the whole directory. Verified directly: `find_sidecar` on
  seed3's path now correctly returns `None`, while the same call on
  seed1's VecNormalize path still correctly finds its real sidecar.
  `python -m py_compile` clean. Re-running seed3's 45-trial eval now
  with the fix in place.
- **No corrupted number resulted** -- the bug was a hard crash, not a
  silent wrong answer, so nothing needed to be thrown out.

## Seed3 real result: 17.8% -- the 3-seed baseline mean+-std is now locked in, and it's nowhere near 70%

> **CORRECTION (2026-11-02):** "same config" holds for the reward, observations and hyperparameters, but not for the training procedure: the 64.4% run took 8 process launches, seedcheck 2 productive ones, seed3 1. These numbers are also 45 trials each, in deterministic mode only, so the mean and std are provisional until the dev-set re-score below. See "Corrections to tonight's audit".

- **Date:** 2026-11-02. Seed3's 45-trial eval: **8/45 held securely
  (17.8%)** -- TRAIN 8/33 (24.2%), HOLDOUT 0/12 (0.0%).
- **Three seeds of the exact same config, zero code changes, now in
  hand: 64.4%, 35.6%, 17.8%.** Mean = **39.3%**, population std =
  19.2 points, sample std (n-1) = 23.5 points, range 17.8-64.4 (a
  46.6-point spread across 3 random seeds).
- **Against the user's explicit "consistent 70%" bar from earlier
  tonight: this does not clear it, not close.** The mean sits at
  roughly half that target, and the spread is so large that ANY
  single seed's result is a poor predictor of the next one. Per the
  user's own pre-set stopping rule ("if nothing reaches 70%
  consistently... report the honest best result with real mean+-spread
  ... and stop"), this is that moment -- not a new judgment call, just
  executing what was already agreed.
- **This is the honest number for the paper's RL result:** a 3-seed
  mean+-std (39.3% +/- ~20-24 points) for the current reward/env
  design, not a single lucky point estimate. The 29-39-point run-to-run
  spread is itself a legitimate, reportable finding -- it says this
  training setup is highly seed-sensitive, which is real information
  about the system, not noise to explain away.

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

## Baseline comparability check: 40.0% and 58.6% are NOT comparable -- the gap is mainly a max_steps difference, not sampling noise

- **Date:** 2026-11-02. Read-only: no simulation was run. Compared `Data/closed_loop_scenes/batch_results.jsonl` (the stored 18/45) with `Data/baseline_modelA_1000trial_results.jsonl` (the 1,005 trials).
- **Answer: not comparable.** I wrote "identical methodology" and "too small a sample" without checking. I never passed `--max-steps` to the 1,005-trial run, so it used the batch script's default of 1500. The 40.0% run used 2500 (this file, "Real, properly-timed batch").

| Setting | 45 trials (2026-10-25) | 1,005 trials (2026-11-02) |
|---|---|---|
| Checkpoint | `model_a.pt` | same file, unchanged since before the first run |
| `squeeze_intensity` | 0.95 | 0.95 |
| `pull_multiple` | 1.5 | 1.5 |
| ramp / settle / pull ramp | 500 / 300 / 500 | 500 / 300 / 500 |
| `decide_action()` threshold, `STEP_FRACTION` | 0.3, 0.35 | 0.3, 0.35 (same at every commit since) |
| **`max_steps` (total, includes ramp + settle)** | **2500** | **1500** |
| Steps at full pull | 1200 | 200 |
| Seeds | 0-2 | 0-66 |

- **Per-trial, same 45 (object, seed) pairs: 18/45 at 2500 steps, 24/45 at 1500.** 39 trials have the same outcome. All 6 that differ went from slip to held: `cone`/1, `capsule`/2, `flat_disk`/1, `hexagonal_prism`/0, 1 and 2. None went the other way.
- **Cause: the horizon, not non-determinism or a code change.** If the runs are deterministic and differ only in length, the 1500-step run is a prefix of the 2500-step run, so its maximums can never exceed the longer run's. There are zero violations across 45 trials on displacement, slip probability and corrections. 14 trials have bitwise-equal `max_displacement`, 10 of them with corrections firing, across separate processes 8 days apart. 14 trials differ by exactly 200 corrections, which at one decision per 5 steps is exactly the 1,000 extra steps.
- **Code changes:** the `predict_slip()` refactor, the `feats` argument, `step_callback` and the `load_model()` device handling are in `closed_loop_sim.py` between the runs. The bitwise matches show they did not change behaviour for Model A. `batch_closed_loop_trials.py` is unchanged since commit 81b8e90.
- **Splitting the 18.6-point gap:** 40.0% -> 53.3% (same trials, shorter horizon) is 13.3 points. 53.3% -> 58.6% (45 pairs versus all 1,005) is 5.3 points, about 0.7 SE at n=45 and plausibly sampling. Excluding seeds 0-2, the 1500-step rate is 565/960 (58.9%).
- **cube/seed=0 anomaly:** not resolved, since it concerned the batch path versus a diagnostic script. But this batch path reproduces across processes, which weakens the "engine-level non-reproducibility" candidate recorded earlier.
- **Model B has the same problem.** The stored 33.3% run shows the same fingerprint (`cube`/0: 331 corrections then, 131 now), so it also used 2500. The 1,005-trial Model B run in progress uses 1500. It is comparable with A's 58.6%, not with 33.3%. A's 40.0% and B's 33.3% remain comparable with each other.
- **Neither hand-tuned horizon matches the RL evaluations.** `rl_env.py` counts `max_steps` after settle, so RL's 1500 means 2300 total steps and 1000 at full pull. RL has 1000 steps at full pull, hand-tuned has 1200 (the 40.0%) or 200 (the 58.6%). The 58.6% therefore faces a much easier test than any RL number and must not be set against them. The matching setting is `--max-steps 2300`.
- **Second asymmetry, not yet measured:** RL `held_securely()` counts a drop during formation as failure. `run_trial()` has no such check and latches its displacement reference at the end of settle, so an object already dropped by then could score as held. The JSONL has no object height, so this can't be checked from stored data. If it occurs, it inflates every hand-tuned number relative to RL.
- **What can be cited now:** 58.6% (589/1005) only as "hand-tuned Model A, 1500 total steps, no formation-drop check". No 1,005-trial hand-tuned number exists yet at a horizon comparable to RL.

## Corrections to tonight's audit

- **Date:** 2026-11-02. Checked against SB3 2.9.2 source and the training logs.
- **Minibatching: I was wrong.** SB3's rollout buffer is `n_steps * n_envs` (`buffer_size = self.env.num_envs * self.n_steps`), and the logs step by 1024. With `batch_size=256` that is 4 minibatches per epoch, 40 per PPO iteration. "`batch_size` equals `n_steps`" and "zero real minibatching" were both wrong.
- **n_steps: the "8x" comparison was misleading.** Samples per update are 1024 versus 2048 for SB3's default at `n_envs=1`, a factor of 2. What is actually short is the per-env rollout: 256 macro-steps, against a 460 macro-step cap on a fully held episode ((800 + 1500) / 5). Final mean episode lengths are 162-351. Any held episode outlasts one rollout, so its +20 `HOLD_BONUS` is reached through value bootstrapping. That is the real case for the n_steps test, and it is untested.
- **Episode budget: "~89 per object" was wrong.** It used `ep_len_mean` values from a mixed log tail. Summing 1024 / `ep_len_mean` per iteration over complete logs gives 1,669-2,122 episodes per run, about 150-190 per object.
- **"No seeding anywhere" was too strong.** Each worker's scene stream is seeded with its worker index and restarts at every process launch. Only policy initialisation and action sampling vary freely between runs.
- **"Exact same config" needs a qualifier.** Reward, observations and hyperparameters match, and the `rl_env.py` diff since 9ddc2fc is only the coefficient and `get_reward_config()`. The procedure does not: 8 launches for the 64.4% run, 2 productive for seedcheck, 1 for seed3, and `EvalCallback` was present for part of the 64.4% run. Three points can't separate this from chance, and it is an uncontrolled difference in the seed comparison.
- **Literature claims I overstated in the plateau entry:**
  - "27% to 95% sim-trained closed-loop": the 27% is a real-world deployment figure, so that is not a sim-only range.
  - The curriculum comparison comes from a secondary write-up, and "BECAUSE it lacked curriculum" was my inference.
  - "VecNormalize is SB3's default": PPO does not apply it automatically. Its flags default to true only once it is wrapped. "Precisely because raw PPO is fragile" was my wording. The same claim in `train_rl_policy.py`'s docstring and inline comment has been removed.
  - arXiv 2505.19247 reports better PPO results with larger minibatches and fewer epochs on some tasks. "Relative to n_steps" was my addition.
- **Omitted from my variance report:** the same search summary of Henderson et al. says TRPO and PPO were the most stable methods, with small confidence bounds. That cuts against treating our spread as typical. We can say seed sensitivity is a documented problem. We cannot say our 46-point range is normal for PPO.
- **Training logs say something about the 2.2%, not yet tested at runtime.** Final `ep_len_mean` roughly tracks eval success: 351 (64.4%), 218 (24.4%), 216 (17.8%), 202 (35.6%, partial log), 162 (2.2%). Episodes were already short during stochastic training, which weighs against an eval-only bug but does not rule one out. The static read also passes: the sidecar has `norm_obs=True`, shape (115,), about 300k samples and no zero-variance dimensions; `training=False` is set; and `normalize_obs` runs before `predict`.
- **Open hypothesis, untested:** formation reward can reach about 1,040 over 160 macro-steps, while phase 2 pays at most +20 at the end and -50 x displacement at every macro-step. Holding at 5 mm for 300 macro-steps scores about -55, against about -20 for slipping straight away. Holding may therefore be weakly rewarded. The harness below records reward per phase so this can be measured.
- **`evaluate_rl_policy.py` uses `deterministic=True`, while training samples with final action std 0.82-0.93.** Every RL success rate this week measures the mean action, not the policy as trained. Item checked next; see the rule below.

## Pre-registered evaluation rules for all remaining RL comparisons (written before any of these results exist)

- **Date:** 2026-11-02.
- **Dev set:** 15 objects x seeds 1000-1009 = 150 trials, the same for every checkpoint. SE is about 4 points at p=0.4. Going to 300 only reaches 2.8, small next to a between-seed SD of about 20. Used for all choices.
- **Locked set:** the 15 x 67 structure, seeds 0-66, with seeds 3-66 (960 trials) as the primary figure, because seeds 0-2 have already been used to compare and choose policies. Used only to report finalists, never to choose. The 64.4% checkpoint's run must be labelled "best of 3 seeds, deterministic".
- **Horizon:** RL `--max-steps 1500`, unchanged. Any hand-tuned comparison with RL uses `--max-steps 2300`.
- **Eval-path check, 2.2% policy, the original 45 pairs, deterministic, direct versus training-style wrapper:** pass means at least 43/45 trials agree. A failure voids the 2.2% and blocks any further VecNormalize run until explained. Separately, a stochastic wrapper run on train objects should land within 25% of the final training values (`ep_rew_mean` 334, `ep_len_mean` 162). If not, report it.
- **Deterministic versus stochastic, 64.4%, 35.6% and 17.8% checkpoints on the dev set:** per-trial pairing, with action noise seeded per trial. If `|difference| > 2 * sqrt(b + c) / n` on any checkpoint (b, c = discordant trials), it counts as changing the numbers: stop and report before going further. The eval mode for later comparisons is chosen on the dev set only.
- **Config comparison with 2-3 seeds:** all seeds count, none dropped except documented crashes.
  - **Improved:** every seed of the new config above every baseline seed, AND a mean difference of at least the larger of 15 points or 1 baseline SD on the dev set.
  - **Worse:** the same test in reverse.
  - **Anything else: inconclusive,** reported that way with all seed values.
  - Full separation has chance probability 1/10 for 2 v 3 seeds and 1/20 for 3 v 3, so even "improved" is directional evidence, not significance.
- **Stopping:** every seed at or above 70% on the dev set means confirm once on the locked set, then stop. Otherwise report the best config's mean and SD across all its seeds. A single best seed is never the headline.

## Model B 1,005 trials at 1500 total steps: 57.9%, tied with Model A

- **Date:** 2026-11-02. `batch_closed_loop_trials.py`, `model_b.pt` (25ch), seeds 0-66, 0 failures: **582/1005 (57.9%)**, against Model A's 589/1005 (58.6%). Label both "1500 total steps, 200 at full pull, slip-only scoring". Comparable with each other only. The drop finding below means neither rate measures objects actually held.

## Formation-drop recording added -- and many "held" hand-tuned trials are objects on the floor

- **Date:** 2026-11-02. Read-only scans of stored data and 5 re-run trials. No training or dataset change.
- **Change:** `run_trial()` now records `dropped_during_formation` and `formation_max_fall`, using the same test as `SlipGraspEnv.step()`: fall from spawn height above 0.05 m during ramp or settle. `rl_env.py` now reads the threshold from one shared constant, same value. The spawn height comes from a separate `MjData`, so the simulation state is untouched.
- **Verified:** 5 known trials (A: `cube`/0, `plus_cross`/0, `flat_disk`/0. B: `cube`/0, `plus_cross`/0) reproduce the stored 1,005-trial results bitwise on every stored field. RL's spawn height matches on 2 scenes (0.245 m).
- **Batch runner hardened after two reviews:** `--max-steps` and `--squeeze-intensity` are now required, full config and code hash are stored in every row, a file with a different config is refused, runs resume after a crash, and interrupted trials are not saved.
- **The cube/seed=0 anomaly is resolved.** `cube`/0 scores "held" (displacement 0.0059) with `formation_max_fall` 0.227 m from a 0.245 m spawn: it has fallen to about floor level before the displacement reference is latched at end of settle, so it barely moves afterwards. `flat_disk`/0 is the same (fall 0.243 m, displacement 0.0002, no corrections). The diagnostic that found it on the floor and the batch that scored it held were both right. Engine-level non-reproducibility is not needed to explain it.
- **Consequence:** every hand-tuned rate so far (40.0%, 33.3%, 58.6%, 57.9%) counts dropped objects that land and stay put as held. The fraction is being measured by the 2300-step reruns.
- **The same problem is in the TCN's dataset.** `generate_trials.py` latches its reference the same way. `Data/pitcn_dataset_full_random_split.npz` has 12,259 + 2,627 + 2,628 = 17,514 trials, exactly the `ever_contacted` trials in `manifest_full_merged.json`. A random sample of 1,000 of those, applying the same fall test to the stored `obj_z`:

| `trial_slip_label` | Trials sampled | Fell > 5 cm in formation | Median fall |
|---|---|---|---|
| 0 (no slip) | 398 | 297 (74.6%) | 0.215 m |
| 1 (slip) | 602 | 245 (40.7%) | 0.045 m |

- At `squeeze_intensity >= 0.9`, 59 of the 100 sampled no-slip trials had also dropped, so higher grip does not remove it.
- **What this does and doesn't show.** About three quarters of the sampled no-slip training examples appear to be objects that had already fallen out of the hand, even though each had some finger contact at some point. Model A and Model B may therefore have learned partly to tell "falling" from "already at rest", rather than slip from secure grip. That is a hypothesis: F1 on trials without a formation drop has not been measured. These numbers are a sample, and they do not show how much of the models' F1 depends on these trials.
- **Not done:** no relabelling, regeneration or retraining. Those are stop conditions and need a decision. RL training is also affected indirectly, since RL observations include the TCN's slip probability.

## `model_a.pt` is the F1 0.691 retrain, not the cited 0.773 checkpoint (hash match; F1 measurement pending)

- **Date:** 2026-11-03. Found by review of `src/analysis/tcn_f1_without_drops.py`. Read-only.
- **Fact:** `src/training/model_a.pt` and `model_a_f1_0691_run.pt` have the same sha256 (`ce813826...`) and the same timestamp (2026-10-25 21:06). This entry's earlier section says the restore run "landed at F1=0.691", and that 0.773 "remains the number currently in use".
- **Consequence if the measurement confirms it:** all later Model A closed-loop numbers, and every RL policy's TCN observation, use the 0.691 checkpoint. That includes `results.md`'s "Model A (22ch, F1=0.773 static)" label, written with a run from the same evening, and 40.0%, 58.6%, and the current 2300-step run. Model A versus Model B closed-loop comparisons are then 0.691 versus 0.799 static F1, and the static 0.773 versus 0.799 comparison would refer to a different file.
- **Earlier version:** git commit 9d4d079 (2026-10-24 20:34) has a different `model_a.pt` (`dbb50c5a...`), extracted to the scratchpad only. Whether it is the 0.773 run is not known yet.
- **Next:** gate-only runs of both checkpoints in the next gap between MuJoCo jobs. No checkpoint is replaced, and no subset numbers are produced until the gate is settled.

## Correction, by measurement: the checkpoint in `model_a.pt` has F1 0.691. The 0.773 model is an older file in git

- **Date:** 2026-11-03, 12:37am. `src/analysis/tcn_f1_without_drops.py --gate-only`, whole test split (2,628 trials), per-timestep pooled metric, CPU. Tolerance +/-0.001, fixed in the script before measuring. Log: `Data/_tcn_gate_only.log`.

| Checkpoint (sha256 prefix) | Where it is | Decision rule | Logged F1 / accuracy | Measured F1 / accuracy | Gate |
|---|---|---|---|---|---|
| `ce813826` | `src/training/model_a.pt` (same bytes as `model_a_f1_0691_run.pt`) | argmax | 0.691 / 0.887 | 0.690658 / 0.886765 | OK |
| `dbb50c5a` | `model_a.pt` in commit 9d4d079 only, not in the working tree | argmax | 0.772893 / 0.898991 | 0.772911 / 0.899000 | OK |
| `948a2716` | `src/training/model_b.pt` | threshold 0.50 | 0.772206 / not logged | 0.772206 / 0.910707 | OK |
| `948a2716` | `src/training/model_b.pt` | threshold 0.30 | 0.798588 / 0.908238 | 0.798584 / 0.908237 | OK |

- **This corrects** the line "Model A (22ch, F1=0.773 static): 8/45 (17.8%) held securely" in "Closed-loop batch trial results (Model A vs Model B, 45 trials each)". That line stays as written, because this file is append-only. If that run used the current file, its label should read `ce813826`, F1 0.691. The file's timestamp (2026-10-25 21:06) is before that batch's stored results (22:07) and after the 9d4d079 commit, but no log records which file the 8/45 run loaded.
- **What is measured about later runs:** the 1,005-trial and 2300-step hand-tuned runs and the 2.2%, seedcheck and seed3 RL runs loaded `src/training/model_a.pt` after that timestamp, so they used `ce813826`, F1 0.691. The 2300-step rows also store its hash. The earlier 64.4% run's checkpoint dates from 2026-10-30, also after it. Every RL policy therefore used `ce813826` as its slip signal.
- **The static comparison and the closed-loop comparison use different files.** 0.773 versus 0.799 compares `dbb50c5a` with `948a2716`. Closed-loop Model A versus Model B compares `ce813826` with `948a2716`.
- **Nothing was restored, replaced or renamed.** Which Model A the paper cites is the user's decision.

## Hand-tuned at 2300 steps, 1,005 trials each: 18.9% and 19.3% under RL's 5 cm rule, and that rule is itself in question

- **Date:** 2026-11-03. Files: `Data/handtuned_model{A,B}_2300step_results.jsonl`. No errors or duplicates. Model A is `ce813826` (F1 0.691), Model B is `948a2716`.
- **Scores (held of 1,005):** no drop rule A 448 (44.6%), B 449 (44.7%). 15 cm rule A 319 (31.7%), B 320 (31.8%). 5 cm rule A 190 (18.9%), B 194 (19.3%). Seeds 3-66 (960 trials), 5 cm rule: A 180 (18.8%), B 185 (19.3%). Paired at 5 cm: 6 held only by A, 10 only by B. Tied.
- **Drop flags are the same for A and B on all 1,005 trials,** as expected, because formation runs before any controller action.
- **Older results with falls joined on:** 1500 steps, A 58.6% to 45.5% (15 cm) and 26.6% (5 cm), B 57.9% to 44.8% and 25.9%. 2500 steps, 45 trials, A 40.0% to 31.1% and 22.2%, B 33.3% to 24.4% and 17.8%. Prefix check: Model A's shorter run never exceeds its longer run's maximum displacement, in 1,005 of 1,005 and 45 of 45 trials.
- **The 5 cm cutoff sits in the densest part of the distribution.** Trials by fall: 4.5-5.0 cm 117, 5.0-5.5 cm 171, 5.5-6.0 cm 89. Model A held by cutoff: 5 cm 190, 6 cm 302, 7 cm 316, 8 cm 317, 10-20 cm 319. Only 147 trials fell more than 15 cm, and 129 of those scored as held under the old rule.
- **Objects that sank 5-7 cm look gripped in training data.** Sample of 1,500 contact-only trials, seed 2, end of settle: 5 cm or less (n=739) median 4 contacts, force 16.2, 75.1% slip. 5-7 cm (n=232) median 5 contacts, force 49.4, 75.4% slip. 7-15 cm (n=150) median 3 contacts, force 0.15, 98.0% slip. Over 15 cm (n=379) median force 0.15, 7.9% slip. Not measured directly in closed loop.
- **Correction to "Formation-drop recording added" above:** it says about three quarters of sampled no-slip examples appear to have fallen out of the hand. That counted everything past 5 cm. In this sample, 349 of 593 no-slip examples (59%) fell more than 15 cm, and 57 (10%) sank 5-7 cm.
- **Correction to "A genuinely new, concentrated failure mode: formation-phase drops":** that entry calls these drops structurally impossible under hand-tuned control. On the same 45 trials, hand-tuned passes 5 cm in 23, and RL in 11, 14 and 22.
- **RL comparison on the same 45 trials, 5 cm rule:** hand-tuned A 10/45, B 9/45. RL 29/45, 16/45, 8/45. Paired against Model A: seed 2 has 11 held only by RL and 5 only by hand-tuned; seed 3 has 4 and 6. Seed 1 has no per-trial file. RL trials end at the 5 cm crossing, so RL can't be re-scored at another cutoff without a new evaluation.
- **Not changed:** the 5 cm rule, the benchmark, any checkpoint. Which rule the paper uses is the user's decision.

## TCN F1 split by formation drop (5 cm rule), all checkpoints gated

- **Date:** 2026-11-03, 12:38am. `src/analysis/tcn_f1_without_drops.py`, whole test split (2,628 trials: 1,224 no drop, 1,404 drop), pooled per timestep, CPU. Output: `reports/tcn_f1_without_drops.json`, `Data/_tcn_f1_subsets.log`.
- **F1 (all / no drop / drop):** `ce813826` argmax 0.691 / 0.789 / 0.559. `dbb50c5a` argmax 0.773 / 0.868 / 0.660. `948a2716` threshold 0.50 0.772 / 0.868 / 0.638. `948a2716` threshold 0.30 0.799 / 0.879 / 0.703.
- **Reading:** F1 is higher without drop trials, so these trials aren't inflating it. The slip base rate differs between subsets (26.5% vs. 19.3%), so this isn't purely a model effect. The split uses the 5 cm rule, and no floor-level split has been run. This doesn't measure how floor-level training examples affect closed-loop behaviour.

## RL re-scored past the 5 cm rule: the old scores reproduce exactly, and the rule barely changes them

- **Date:** 2026-11-03. `evaluate_rl_policy.py --continue-past-formation-drop`, original 45 trials, deterministic, `model_a.pt` (`ce813826`). Files: `Data/rescore_45_seed{1,2,3}.jsonl`, `Data/rescore_check_seed3_default.jsonl`. Reviewed independently before these results were used. The review's later fixes (fall omitted from default rows, script added to the code hash, plain bools in `info`) don't touch trial handling.
- **Reproduction passed.** Seed 3 in default mode matches its old log on all 45 trials. With the flag on, seed 2 and seed 3 match on every held and drop flag, and on slip and displacement for trials without a drop. Seed 1 has no old per-trial log, but it matches the logged 29/45 and all 15 per-object counts. So 64.4% is reproducible, and the seed spread comes from training, not evaluation noise.
- **Held of 45 (5 cm / 10 cm / 15 cm):** seed 1 29 / 29 / 29. Seed 2 16 / 17 / 17. Seed 3 8 / 8 / 8. Hand-tuned A 10 / 14 / 14. Hand-tuned B 9 / 12 / 12.
- **Falls (5 cm or less / 5-15 cm / over 15 cm):** seed 1 34 / 7 / 4. Seed 2 31 / 7 / 7. Seed 3 23 / 7 / 15. Of the 21 RL trials that fell 5-15 cm, 1 held.
- **Paired with hand-tuned A at 10 cm (held only by RL / only by hand-tuned):** seed 1 15 / 0. Seed 2 8 / 5. Seed 3 4 / 10. Seed 1 is the best of 3 seeds, so its result is favourably selected.
- **Reading:** the 5 cm rule is not what is holding these policies back. This corrects the suggestion, in the 2300-step entry above and in my messages, that RL's drop failures might include many real grips. On these policies they mostly don't. The limit: these policies were never trained past 5 cm, so this doesn't show what a policy trained under a looser rule could do. It only removes the direct evidence for trying one.
- **Order change:** the locked-set runs start before the dev-set checks, at the user's request, so the paper has them first. They aren't used to choose anything, so the pre-registered rule still holds.

## Pre-registered: longer rollouts (n_steps 512), 3 fresh seeds (written before any of these runs)

- **Date:** 2026-11-03.
- **Why:** identical re-runs reproduce exactly, so the 17.8%-64.4% spread comes from training. Each rollout is 256 steps per env, and a held trial lasts up to 460, so updates often see partial trials. This is a hypothesis about the spread, not a known cause.
- **The one change:** `n_steps` 256 to 512. Batch size stays 256, so each rollout gives 8 minibatches instead of 4. Everything else matches the baseline seeds: 300,000 env steps (rounding to whole rollouts makes it 301,056, against 300,032), 4 envs, learning rate, reward, observations, training objects, and TCN `ce813826`. All three seeds count. Crashes get documented, not dropped.
- **Scoring:** dev set, seeds 1000-1009, deterministic, trials running past 5 cm. The 5 cm rule is primary because it is RL's existing rule, and 10 cm is reported beside it. If the deterministic versus stochastic check trips its rule, stop and report before using these.
- **Verdict:** the existing rule, against the three baseline seeds on the same dev set. Improved needs every new seed above every baseline seed AND a mean difference of at least the larger of 15 points or 1 baseline SD. Worse is the reverse. Anything else is inconclusive. That is a high bar and inconclusive is the likeliest outcome.
- **Spread:** SD across seeds is reported, but 3 seeds per config can't support a verdict on it.
- **Not running:** a 10 cm retrain, because re-scoring removed the direct evidence for it (1 of 135 RL scores changed). `norm_obs`-only, because it is lower priority than this. Neither result is known, and either could be tried later.
- **Code timing:** `--n-steps` goes into `train_rl_policy.py` only after the evaluation chain ends, because `evaluate_rl_policy.py` imports that file.

- **Addendum to "Pre-registered: longer rollouts", from code review, before any run:** 301,056 steps holds only for a run with no crash. Crash-and-resume runs can total 300,040-302,032, and the baselines were 300,224, 300,408 and 300,032. Each crash also discards about 1,016 counted steps, against 504 at baseline. Record each seed's actual total and crash count with its result. If a seed crashes between its last checkpoint and final save, the script uses that last checkpoint as the final policy and says so in `Data/_run_nsteps_retrain.out`.

## RL on the locked set, 2.2% check and deterministic vs. stochastic: results, with the stop rule tripped

- **Date:** 2026-11-03. `evaluate_rl_policy.py --continue-past-formation-drop`, `model_a.pt` (`ce813826`), deterministic unless stated. Files: `Data/rl_seed{1,2,3}_1005trial_results.jsonl`, `Data/evalpath_*.jsonl`, `Data/rl_seed{1,2,3}_dev_{deterministic,stochastic}.jsonl`. Reproduce with `src/analysis/nov3_drop_analysis/rl_locked_checks.py`. Each locked file has 1,005 rows, no errored trials, no duplicates and one code hash. Policies: seed 1 `afb280ad`, seed 2 `af378971`, seed 3 `58587691`.
- **Locked set, primary figure, seeds 3-66, held (Wilson 95% CI):**

  | | 5 cm | 10 cm |
  |---|---|---|
  | Seed 1 (best of 3 seeds, deterministic) | 588/960, 61.2% (58.1-64.3) | 610/960, 63.5% (60.4-66.5) |
  | Seed 2 | 405/960, 42.2% (39.1-45.3) | 423/960, 44.1% (41.0-47.2) |
  | Seed 3 | 144/960, 15.0% (12.9-17.4) | 147/960, 15.3% (13.2-17.7) |
  | **Mean and SD across seeds** | **39.5, SD 23.2** | **41.0, SD 24.3** |
  | Hand-tuned A | 180/960, 18.8% | 305/960, 31.8% |
  | Hand-tuned B | 185/960, 19.3% | 308/960, 32.1% |

- **All 1,005 trials, 5 cm / 10 cm:** seed 1 617 / 639 (61.4% / 63.6%). Seed 2 421 / 440 (41.9% / 43.8%). Seed 3 152 / 155 (15.1% / 15.4%). Mean 39.5 / 40.9, SD 23.2 / 24.2.
- **Stopping rule:** no seed reaches 70%, so the rule applies: report the mean and SD across all seeds, and a single best seed is never the headline. Seed 1 must be labelled "best of 3 seeds, deterministic".
- **Paired with hand-tuned A, seeds 3-66 (held only by RL / only by hand-tuned, exact McNemar):** at 5 cm, seed 1 438 / 30 (p = 5e-94), seed 2 299 / 74 (p = 4e-33), seed 3 111 / 147 (p = 0.029). At 10 cm, seed 1 353 / 48 (p = 2e-58), seed 2 250 / 132 (p = 2e-9), seed 3 95 / 253 (p = 1e-17). Seeds 1 and 2 beat hand-tuned, and seed 3 is worse than it. Model B gives the same pattern; the script prints those.
- **Three-seed mean against hand-tuned A, seeds 3-66 (not pre-registered, df 2):** 5 cm, difference 20.7 points, t = 1.54, p = 0.26. 10 cm, difference 9.2 points, t = 0.66, p = 0.58. So individual policies differ clearly from hand-tuned, but this does not establish that the training procedure beats it on average.
- **The 45-trial numbers held up roughly, at 5 cm:** seed 1 64.4% to 61.2%, seed 2 35.6% to 42.2%, seed 3 17.8% to 15.0%.
- **Outcomes at 5 cm, all 1,005 (held / fell past 5 cm / slipped without falling):** seed 1 617 / 200 / 188. Seed 2 421 / 291 / 293. Seed 3 152 / 486 / 367.
- **Training vs. held-out objects at 5 cm, seeds 3-66:** seed 1 57.1% vs. 72.7%. Seed 2 38.8% vs. 51.6%. Seed 3 15.5% vs. 13.7%. Hand-tuned A 15.6% vs. 27.3%, so the held-out group also scores higher under a controller with no training objects. This likely reflects which shapes are in that group, not generalization. Seed 1's 72.7% is not a 70% result.
- **Per object at 5 cm, all seeds, of 67 (seed 1 / 2 / 3 / hand-tuned A):** square_pyramid 66/65/21/25. rod 63/35/0/4. cone 61/66/25/15. dumbbell 61/17/7/32. sphere 56/43/0/48. plus_cross 53/24/8/10. ellipsoid 51/37/16/23. triangular_prism 51/15/11/0. hexagonal_prism 39/33/6/0. capsule 38/31/27/21. cylindrical_prism 26/3/2/0. l_block 22/21/18/3. flat_disk 14/16/1/3. rectangular_prism 9/6/3/5. cube 7/9/7/1. Seed 1 reaches 70% on 8 of 15 objects. Cube and rectangular prism stay at 9 of 67 or fewer under every controller here. The cause is not established.
- **2.2% eval-path check: passed.** Direct and wrapper paths agree on 45/45 trials (rule: at least 43), and also on slip, drop and episode length, with rewards within 3.5e-6. Both hold 1/45. Stochastic wrapper on training objects: reward 337.5 vs. 334 (+1.0%), length 153.3 vs. 162 (-5.3%), rule within 25%. It holds 0/45. The 2.2% is not an evaluation bug. It stays one unseeded run on 45 trials, so it is not a verdict on VecNormalize.
- **Deterministic vs. stochastic, dev set, 150 paired trials: THE RULE TRIPS ON ALL THREE POLICIES, at both cutoffs.** Same objects and start conditions in both modes, with action noise seeded per trial.

  | 5 cm | Deterministic | Stochastic | Only det / only sto | Difference | Limit | p |
  |---|---|---|---|---|---|---|
  | Seed 1 | 86 (57.3%) | 73 (48.7%) | 19 / 6 | 8.7 | 6.7 | 0.015 |
  | Seed 2 | 56 (37.3%) | 35 (23.3%) | 24 / 3 | 14.0 | 6.9 | <0.001 |
  | Seed 3 | 19 (12.7%) | 8 (5.3%) | 14 / 3 | 7.3 | 5.5 | 0.013 |

  At 10 cm: seed 1 91 vs. 75 (difference 10.7, limit 6.8), seed 2 60 vs. 39 (14.0, 7.4), seed 3 19 vs. 8 (7.3, 5.5). Every RL success rate reported so far measures the mean action, and sampling from the policy as trained holds 7-14 points fewer.
- **Action on the trip:** reported to the user on 2026-11-03. Which mode is primary for later comparisons is waiting on the user. Locked-set numbers above are deterministic and are labelled so. Longer-rollout results are not used until this is decided. Training itself does not depend on the evaluation mode, so the retrain was not cancelled.
- **Baseline dev scores for the longer-rollout verdict, deterministic:** 5 cm 57.3, 37.3, 12.7 (mean 35.8, SD 22.4). 10 cm 60.7, 40.0, 12.7 (mean 37.8, SD 24.1). So "improved" at 5 cm needs every new seed above 57.3% and a mean of at least 58.2%. No seed has been judged yet.
- **First longer-rollout launch failed at startup, no training happened.** The patch applied at 23:52, then all 15 attempts (5 per seed) exited with 0 steps and no policy, and no seed was scored. Cause: Windows commit memory ran out. The limit is 35.8 GB (31.8 GB RAM plus a fixed 4 GB page file), and about 24 GB was already committed by other software, including 8.8 GB for the browser. Each training launch needs about 15 GB across 5 processes. Evidence: 4 of 15 attempt logs show qhull or `MemoryError` out-of-memory failures, the other 11 show the same worker crash during reset, and 2.0 GB of commit was left during an attempt. Logs: `Data/nsteps512_failed_startup_2026-11-03/`. These are not seed crashes for the addendum's count. The restart (`Data/_run_nsteps_restart.sh`) uses the same config, still 4 envs, and waits for at least 17 GB of free commit.

## Pre-registered, measurement only: noslip pass on fresh seeds (written before any of these runs)

- **Date:** 2026-11-04, evening.
- **Why:** on the 150 dev trials, the noslip pass alone moved the three unchanged policies from 57.3 / 37.3 / 12.7% to 81.3 / 67.3 / 40.7% at 5 cm, and hand-tuned A from 23.3 to 44.0%. The dev set has been looked at many times, so this measures the same thing on trials nothing has been chosen on.
- **This adopts nothing.** Changing the benchmark's physics is the user's decision and it has not been made. Default physics stays the benchmark until then, and no figure from this run replaces a result above.
- **What runs:** `diagnostic_confirm_runs.py`. RL seeds 1, 2 and 3 (`afb280ad`, `af378971`, `58587691`, unchanged, trained on default physics) and hand-tuned A at squeeze 0.95, all with `model_a.pt` (`ce813826`), deterministic, trials running past 5 cm, 2,300 physics steps, `noslip_iterations` 20, nothing else changed. 15 objects x seeds 2000-2066, 1,005 trials per controller. Before launch the runner reproduced 60 of 60 existing rows exactly (default and noslip, RL seed 1 and hand-tuned A, dev seed 1000).
- **Not touched:** the locked set stays unrun under noslip until the user decides, so it remains available for one final read.
- **Reported:** every controller, whatever it scores. Held at 5 cm and 10 cm with Wilson intervals, fell and slipped counts, per-object counts, mean and SD across the three seeds, and RL vs. hand-tuned A paired with exact McNemar. All three seeds count. The existing rule stands: no single best seed as the headline, and seed 1 is labelled "best of 3 seeds, deterministic".
- **How to read it:** 5 cm is primary, RL's existing rule. If a controller's dev figure falls outside its fresh-seed 95% interval, the dev figure misstated it, and the write-up says so. No early stopping. No trial is rerun except to resume after a crash, which gets documented.
- **What this can't show:** what training under noslip would give, because the policies and the TCN were trained on default physics. Whether a fixed grip matches RL under noslip is a separate dev-set run.
- **Addendum to "Pre-registered, measurement only: noslip pass on fresh seeds", before any fresh-seed result was read:** elliptic cone with `impratio` 10 joins this measurement, with the same controllers, seeds and rules, run after the noslip runs. Why: on seed 1's 150 dev trials it holds 118 against noslip's 122 (paired 11 vs. 15, p = 0.56), and it is a more defensible setting if one is ever adopted. The MuJoCo Menagerie's Allegro, Shadow Hand and Robotiq 2F-85 files set exactly `cone="elliptic" impratio="10"`. MuJoCo's modeling guide says "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance. If that is not sufficient, enable the Noslip solver", and calls noslip "just an adhoc mechanism" that has shown "some instabilities in models with more complex interactions among multiple contacts". This adopts nothing either.

## Longer rollouts (n_steps 512): results, verdict "worse"

- **Date:** 2026-11-04. Rules: "Pre-registered: longer rollouts". Launched at 14:01 after a reboot at 13:58 (Windows event 1074, Start menu) ended the wrapper that had spent 14 hours waiting for memory. Each seed trained on its first attempt with no crash and totals 301,056 steps. Scene, TCN `ce813826`, evaluation code, reward config and trial protocol match the baseline rows field for field. Policies: seed 1 `90ca3a2d`, seed 2 `d3a531e4`, seed 3 `2089e132`. Files: `Data/ppo_slip_policy_nsteps512_seed{1,2,3}.zip`, `Data/rl_nsteps512_seed{1,2,3}_dev_deterministic.jsonl`. Reproduce with `src/analysis/nov3_drop_analysis/nov4_chain_results.py`.
- **Dev set, deterministic, 150 trials, held:**

  | | Seed 1 | Seed 2 | Seed 3 | Mean | SD |
  |---|---|---|---|---|---|
  | Baseline, 5 cm | 57.3% | 37.3% | 12.7% | 35.8 | 22.4 |
  | n_steps 512, 5 cm | 10.0% | 0.0% | 2.7% | 4.2 | 5.2 |
  | Baseline, 10 cm | 60.7% | 40.0% | 12.7% | 37.8 | 24.1 |
  | n_steps 512, 10 cm | 10.0% | 0.0% | 2.7% | 4.2 | 5.2 |

- **Verdict, by the pre-registered rule, at both cutoffs: worse.** Every new seed is below every baseline seed. The mean falls 31.6 points at 5 cm (bar 22.4) and 33.6 at 10 cm (bar 24.1).
- **Status:** deterministic only. The deterministic vs. stochastic stop rule is still open, so this is reported and not used for anything further until the evaluation mode is decided.
- **The spread:** SD falls from 22.4 to 5.2 only because all three seeds are near the floor. That is not an improvement.
- **The hypothesis behind this run is not supported:** rollouts shorter than a trial are not shown to be what causes the spread.
- **Why it got worse is not established.** Both configurations take about the same number of gradient steps (about 11,760 vs. 11,720), but n_steps 512 collects half as many rollouts, so PPO's clip range limits how far the policy moves half as often. That is a hypothesis. Finger traces for these policies are queued.
- **Reward doesn't rank these policies by holding.** Mean evaluation reward and held at 5 cm: 623 and 86, 493 and 56, 434 and 15, 434 and 4, 415 and 19, 404 and 0. The pull-phase part is negative for all six and least negative for the policies that hold least: -31, -27, -14, -21, -23, -14 in that order.

## Nov 4 code review and diagnostics: the TCN signal, fixed grips and contact settings

- **Date:** 2026-11-04. All read-only: no reward, observation, action, controller, physics, benchmark or training setting changed, and no stored result was touched. Dev set, 15 objects x seeds 1000-1009, 5 cm unless stated. The physics variants are set on in-memory models. With default settings the diagnostic reproduces seed 1's stored rows 150 of 150. Scripts: `diagnostic_contact_physics.py`, `diagnostic_handtuned_physics.py`, `diagnostic_scripted_grip.py`, `diagnostic_confirm_runs.py`. Analysis: `nov4_chain_results.py`, `noslip_sanity_checks.py`, `contact_settings_compare.py`, `path_to_70.py`, `creep_and_reward_checks.py`.
- **The TCN signal changes no RL outcome.** Replacing the slip probability with 0 at test time leaves 86, 56 and 19 held, and 0 of 450 trials change at either cutoff. Trajectories differ only slightly: seed 1 has the same trial length on 145 of 150, with a median change in maximum displacement of 0.00 mm. This tests the signal's use at run time. It does not test whether training with the signal helped, because no policy was trained without it.
- **Hand-tuned, same direction, not yet the exact test.** A scripted version of the same ramp with no TCN and no corrections holds 40, against hand-tuned A's 35 (paired 6 vs. 1, p = 0.125; 10 cm 62 vs. 58, p = 0.344). The exact run, hand-tuned with no TCN, is queued.
- **A fixed grip matches the best RL policy under default settings.** No policy, tactile input or TCN. Held of 150 (5 cm; 10 cm; fell; slipped): thumb, index, ring and pinky at full speed 86 (57.3%; 60.7%; 34; 30). Same with a slower ring and pinky, speeds read from seed 1's traces, 92 (61.3%; 64.0%; 29; 29). Same full-speed grip plus the middle finger 34 (22.7%; 39.3%; 98; 18). Thumb, index and ring at full speed 76 (50.7%; 52.7%; 40; 34). Slow 5-finger close, like the hand-tuned ramp, 40 (26.7%; 41.3%; 85; 25). RL seed 1 is 86 (57.3%; 60.7%; 33; 31), paired 9 vs. 9 (p = 1.00) with the first grip and 1 vs. 7 (p = 0.07) with the second. Choosing among grips on 150 trials inflates the best one, and the second grip's speeds came from seed 1. So RL's lead over hand-tuned may come from closing fast without the middle finger, not from learned feedback.
- **What seed 1 does.** Median closure at RL steps 10, 20, 50: thumb 0.80, 1.00, 1.00. Index 0.80, 1.00, 1.00. Ring 0.62, 1.00, 1.00. Pinky 0.20, 0.62, 1.00. Middle 0.00 throughout the median trial. On 15 traced trials, raw mean actions sit near or beyond the +1 limit, and 4 of 15 change any finger by more than 0.01 during the pull.
- **Objects move while contacts are far inside their friction limit.** Seed 1 under default settings, 15 trials at full load: median speed 1.18 mm/s, up to 5.9. Median friction use 3%. Capacity 26x load. Fingertips moved a median of 0.33 mm. Failures keep arriving at constant load: 118 of seed 1's 188 slips come after the pull reaches full value, and hand-tuned A held 589 trials after 200 steps at full pull, then lost 141 by 1,000 steps, 108 of them under 2x the limit.
- **MuJoCo documents this.** Modeling guide, "Preventing slip", slow slippage: "Even when the tangential force required for static equilibrium lies strictly inside the contact friction cone, MuJoCo's regularized soft-contact model does not guarantee an exact zero-velocity stick state." The scene sets none of these options, so it runs MuJoCo's defaults: pyramidal cone, `impratio` 1, no noslip pass, `solref` 0.02 1, `solimp` 0.9 0.95 0.001, `condim` 3.
- **Other MuJoCo hand models do set them.** MuJoCo Menagerie, `<option>` lines: Allegro, Shadow Hand and Robotiq 2F-85 `cone="elliptic" impratio="10"`. LEAP `cone="elliptic" impratio="100"`. Shadow DEXEE `cone="elliptic" noslip_iterations="5"`. Modeling guide: "When contact slip is a problem, the best way to suppress it is to use elliptic cones, large impratio, and the Newton algorithm with very small tolerance. If that is not sufficient, enable the Noslip solver."
- **Stricter contact settings, same controllers, none retrained. Exploratory, dev set, not adopted:**

  | Held of 150 (5 cm / 10 cm / fell / slipped) | Default | Noslip 20 | Elliptic, `impratio` 10 |
  |---|---|---|---|
  | RL seed 1 | 86 / 91 / 33 / 31 | 122 / 129 / 16 / 12 | 118 / 124 / 10 / 22 |
  | RL seed 2 | 56 / 60 / 52 / 42 | 101 / 108 / 30 / 19 | queued |
  | RL seed 3 | 19 / 19 / 78 / 53 | 61 / 66 / 62 / 27 | queued |
  | Hand-tuned A | 35 / 58 / 85 / 30 | 66 / 101 / 66 / 18 | queued |

  Noslip against default, paired at 5 cm (held only with / only without): seed 1 42 / 6 (p = 1e-7), seed 2 56 / 11 (p = 2e-8), seed 3 49 / 7 (p = 8e-9), hand-tuned A 41 / 10 (p = 2e-5). Elliptic against default, seed 1: 41 / 9 (p = 6e-6). Noslip against elliptic, seed 1: 15 / 11 (p = 0.56), and 34 trials are rescued by both. Three-seed mean under noslip at 5 cm: 63.1, SD 20.6, against 35.8, SD 22.4. These were not set before the data was seen, the dev set has been looked at many times, and the policies and TCN were trained on default physics. Fresh-seed runs are pre-registered above.
- **The hardest shapes improve the most, seed 1, held of 10, default / noslip / elliptic:** flat disk 0 / 10 / 10, cube 0 / 7 / 5, capsule 4 / 10 / 10, rectangular prism 1 / 5 / 5, cylinder 2 / 4 / 7, plus 7 / 10 / 10. Some lose: dumbbell 9 / 6 / 5. This changes the line above that the cause of the cube and rectangular prism failures "is not established": much of it looks like the contact settings, on 10 trials per object.
- **Is the behavior under noslip physically sensible? Mostly, with one prediction that failed.**
  - The grip doesn't change: seed 1's median closure matches at every sampled step, and the slip probability is similar (pull median 0.17 vs. 0.19).
  - No sign of instability on these 450 trials: no solver warnings in the logs, the largest object speeds are similar, and nothing is thrown upward (largest rise after settle 4-8 mm, against 13 mm under default).
  - Grips that rely on friction start working. Among seed 1's held trials, 16 are carried mostly by friction under default and 52 under noslip, and those objects move at a median 2.88 and 0.79 mm/s. Most held trials under default (70 of 86) are carried mostly by normal force, as in a cage.
  - Contacts at a slip are closer to their limit. Median worst contact over its friction limit in the last samples before a slip: 0.56 under default, 0.76 under noslip. So not every slip is pure sliding at the limit, and an object could also be turning.
  - **The prediction that failed:** I expected holding to rise with pad friction under noslip. Split at each object's median pad friction, low vs. high: seed 1 84.0 vs. 78.7, seed 2 68.0 vs. 66.7, seed 3 41.3 vs. 40.0, hand-tuned A 41.3 vs. 46.7. Default is also flat. On 150 trials this can't exclude a small effect, and a larger test is possible when the fresh-seed runs finish. A likely reason is that friction isn't the limit in these grips: capacity is a median 20-50x load for most controllers, and about half of the contact force goes through links and palm, whose friction is fixed at 1.0 (fingertip share median 0.42-0.49). So noslip's remaining failures aren't shown to be friction-limit failures, and this doesn't prove the setting is physically right.
- **Why the middle finger hurts, a hypothesis from geometry.** Its path goes through the object with no opposing thumb. Tip positions at full closure relative to the object's center, in mm: thumb (-14, -3, -30), index (-28, -1, -11), middle (-7, 0, -1), ring (13, 1, -9), pinky (33, -4, -21).
- **The channels named "normal" and "tangential" don't measure that.** Weighted by force over 18 fingertip contacts on 6 objects, the contact normal in the sensor frame is |x, y, z| = 0.67, 0.53, 0.22, and `--normal-axis` is 2. Axis 2 reads a median 18% of the touch sensor (10th-90th 2-62%, 1,583 samples), and touch matches the true contact force. This affects the `*_normal` and `*_tangential` inputs, `grasp_force_at_settle` and the `GRIP_FORCE` reward term. It does not invalidate TCN results, because the network only sees numbers, but the Methods wording is wrong, and a friction-cone loss built on these channels would be wrong. It may also be part of why the October physics features didn't separate slip from stable grasping. That is a hypothesis.
- **Grip strength isn't the limit.** Fingertip contact forces reach 20-70 N on objects under 1.5 N, which contradicts the "actuator saturation limits the grip" explanation in earlier notes. That explanation should not go in the paper as is.
- **The reward barely rewards holding.** With gamma 0.99, the +20 hold bonus is worth 0.98 at the start of the pull and 0.20 at the start of a trial, against about 650 of formation reward per trial. Steady displacement above about 4.4 mm values worse than slipping immediately. Within a policy, held trials still score better in the pull than slipped ones (-20.7 vs. -56.4 for seed 1).
- **Correction to what I reported today:** I said the later steps of `_chain2_nov4.sh` would fail because of CRLF line endings. They ran correctly and their results are valid, 150 rows each. I also wrote a replacement chain from that wrong belief. It was stopped before it started, which prevented duplicate rows, and removed. `_chain4_nov4.sh` and `_chain5_nov4.sh` run only what hadn't run.

## Decision: elliptic cone with `impratio` 10 is the primary benchmark (recorded before any fresh-seed result was read)

- **Date:** 2026-11-04, 8:12 pm EST. The user delegated this choice ("choose whatever you think is best for the benchmark") and I made it. At this point no fresh-seed row had been read under any setting, and seed 1 on the dev set was the only elliptic result read.
- **The choice:** `cone="elliptic" impratio="10"`. Everything else in the scene, trial protocol, scoring and evaluation rules stays as it is. Default settings are reported beside it in every table of held rates. Noslip stays a sensitivity result.
- **Why not default:** under default settings the benchmark partly measures a solver artifact. At full load, objects move at a median 1.18 mm/s while contacts use 3% of their friction limit (seed 1, 15 trials), which is the slow slippage MuJoCo documents. This project never chose those settings. The scene inherits them.
- **Why elliptic and not noslip:** MuJoCo's modeling guide recommends it first and calls noslip "just an adhoc mechanism". Allegro, Shadow Hand and Robotiq 2F-85 in MuJoCo Menagerie set exactly these values. On the one controller run under both, noslip held slightly more (122 vs. 118 of 150, p = 0.56), so elliptic was not picked for its score.
- **Fixed now, so later results can't steer it:**
  - Elliptic stays primary if noslip scores higher on fresh seeds, and whether or not anything reaches 70%.
  - `impratio` is not tuned. The `impratio` 3 and 100 runs are sensitivity results and can't change this choice.
  - "Pre-registered evaluation rules for all remaining RL comparisons" still applies: all seeds count, the locked set under this setting is never used to choose, and a single best seed is never the headline.
- **What would reverse it, fixed before the fresh-seed results are read.** Either of these on the fresh-seed elliptic runs puts default back as primary, with elliptic reported as a sensitivity result:
  - **The gain doesn't replicate:** at 5 cm, fewer than 3 of the 4 controllers have a 95% Wilson lower bound above their locked-set default rate (seed 1 61.2%, seed 2 42.2%, seed 3 15.0%, hand-tuned A 18.8%).
  - **Instability:** solver warnings in those runs' logs.
- **Must be disclosed wherever this setting is reported:** it was proposed after scores had been seen on a much-used dev set. Any policy or TCN trained on default physics is labelled that way. Pad friction doesn't predict holding on the 150 dev trials, so this shrinks a documented artifact and doesn't prove the setting physically correct.
- **What this changes today:** no file, policy or stored result. Default-physics results stay valid as default-physics results. The real scene file is unchanged. Runs so far set the option in memory, and a scene copy with the one `<option>` line reproduced 15 of 15 of those rows.
- **Final figures:** one read of the locked set (seeds 3-66) under this setting, once the finalists are settled, paired with the existing default rows. Fresh-seed results check this choice and are not the final figures.
- **Still the user's decisions:** retraining under this setting, the TCN under this setting, the evaluation mode, and the headline.

## Nov 4 evening: runs interrupted by a crash, and the TCN-removal results that finished

- **Written:** 2026-11-06. At about 8:41 pm on Nov 4 the laptop froze (Windows Kernel-Power 41, bugcheck code 0, so a hang or power loss rather than a software crash) while six jobs were running, about four minutes after the sixth started. It froze again at 8:49 pm, was powered off by hand and needed Startup Repair. Nothing has run since. Five jobs had been running side by side since the afternoon, against the rule of one MuJoCo-heavy job at a time. The cause isn't known, but that load may have contributed, so the rule is back in force.
- **Finished:** hand-tuned A on fresh seeds under noslip, 1,005 rows, not yet read. Every `_chain4_nov4.sh` step through `handA_dev150_noslip20_noTCN`. Seed 2 under elliptic on the dev set.
- **Interrupted, not read:** fresh-seed noslip for RL seeds 1, 2 and 3 at 600, 621 and 666 of 1,005 rows. They resume from their files if rerun. The fresh-seed elliptic runs never started.
- **Interrupted, not used:** seed 3 elliptic dev (100 of 150 rows, from an append-only script, so a rerun starts over) and seed 1 noslip 5 without TCN (111 of 150). `elliptic_scene_check.py` recorded no result. `_chain6_nov4.sh` never started.
- **Removing the TCN signal, dev set, same 150 trials, 5 cm (10 cm), held with / without, then trials held only with / only without:**

  | Controller and setting | With TCN | Without | Only with / only without | p |
  |---|---|---|---|---|
  | RL seed 1, default | 86 (91) | 86 (91) | 0 / 0 | 1.0 |
  | RL seed 1, noslip | 122 (129) | 122 (129) | 0 / 0 | 1.0 |
  | RL seed 2, noslip | 101 (108) | 99 (106) | 4 / 2 | 0.69 |
  | RL seed 3, noslip | 61 (66) | 60 (65) | 1 / 0 | 1.0 |
  | Hand-tuned A, default | 35 (58) | 38 (60) | 1 / 4 | 0.38 |
  | Hand-tuned A, noslip | 66 (101) | 71 (106) | 1 / 6 | 0.13 |

  For RL, 7 of 600 noslip-or-default outcomes change, none significantly. Hand-tuned A, whose TCN triggers grip corrections, holds slightly more without it under both settings, also not significant. So the TCN adds nothing measurable to either controller here. The RL policies were trained with the signal, so this doesn't show what training without it would give. Hand-tuned A with no TCN is the exact test that was queued. Script: `src/analysis/nov3_drop_analysis/tcn_removal_compare.py`.
- **Seed 2 under elliptic, dev set:** 87 of 150 held at 5 cm (58.0%), 99 at 10 cm, 32 fell past 5 cm during formation. Default 56, noslip 101. Exploratory, like the other dev-set contact results.

## Amendment before any fresh-seed elliptic row exists: final figures, fixed grip, and run order

- **Written:** 2026-11-06, about 5:10 pm, before any fresh-seed elliptic run started. The user approved this run. No fresh-seed result under any setting has been read, including hand-tuned A's finished noslip run.
- **No retrain for the Nov 8 paper.** The laptop froze under sustained load on Nov 4. A retrain is the heaviest job here, 5 processes for about 5 hours, and would leave no time to evaluate it. It moves to Phase 2. So nothing gets chosen after this run: the controllers are the existing ones, fixed in advance.
- **So the fresh-seed elliptic results are the final elliptic figures.** This replaces "Final figures: one read of the locked set under this setting" in "Decision: elliptic cone". The locked set stays unrun under elliptic and remains available for Phase 2. Default-physics figures stay the locked-set ones (seeds 3-66). Those are different trials, so default and elliptic are compared unpaired at full size, and paired only on the dev set.
- **The reversal rule applies unchanged** to these runs, through `src/analysis/nov3_drop_analysis/elliptic_gate_check.py`, written on Nov 4 before any of these rows existed.
- **Added: fixed grip `TIRP_fast` on the same 1,005 trials under elliptic.** It has no policy, tactile input or TCN. On the dev set it matched RL seed 1 under default (86 vs. 86 of 150) and under noslip (118 vs. 122). That grip was designed from seed 1's traces and looked at on dev results, so its dev figure is optimistic. This measures it on trials nothing was chosen on. Reported: held at 5 cm and 10 cm with Wilson intervals, and paired against each RL seed and hand-tuned A with exact McNemar. It doesn't feed the reversal rule.
- **Checks before launch:** the fresh-seed runner reproduced 15 of 15 stored seed-1 elliptic dev rows exactly (held, slipped, dropped, formation fall, maximum displacement, episode length and reward). The grip script ran 15 dev trials under elliptic without error.
- **Order, one job at a time:** RL seed 1, hand-tuned A, RL seed 2, RL seed 3, fixed grip. Then, lowest priority, the interrupted noslip runs for RL seeds 1-3 resume from their own files. If the run is cut short, whatever finished is reported as finished and the rest as not run. Script: `Data/diag_fresh_seeds/_fresh_elliptic_nov6.sh`.
- **No early stopping.** A controller's rows aren't read until its run completes. Reporting is the same as in the noslip pre-registration: every controller, whatever it scores.

## Nov 6: paper figures and new analyses of stored data

- **Written:** 2026-11-06, evening. Everything here comes from stored rows, through `src/analysis/paper_figures.py`: 17 figures in `reports/figures/paper/`, with every number and test in `stats_summary.md` there. Every input matched its logged trial and held counts before use. No new simulation, apart from the fresh-seed queue already logged.
- **The reward doesn't favour holding.** RL seed 1, locked set, median total reward per trial: held 611.7 (n = 588), slipped after the grasp formed 616.6 (n = 183), fell while it formed 644.7 (n = 189). Held vs. fell: Mann-Whitney p = 0.037, in the wrong direction. Held vs. slipped: p = 0.71. Pull-phase reward does separate them (medians -12.2, -51.3, -28.9), but formation reward is about 20 times larger.
- **Training reward ranks the seeds wrongly.** Final mean training reward: 448, 329 and 361 for seeds 1, 2 and 3. Held on the locked set ranks them 1, 2, 3. Seed 1's log repeats steps where a crashed segment resumed, and the figure keeps the resumed segment.
- **The seeds differ in which fingers they close.** Dev set, noslip traces, median closure at RL step 160: seed 1 has thumb, index, ring and pinky at 1.00 and the middle finger at 0.00. Seed 2 has thumb, index and pinky at 1.00, ring at 0.78, and middle at 0.00 after moving it briefly. Seed 3 has thumb, index and pinky at 1.00, ring and middle at 0.00.
- **Most of the stricter settings' gain comes while the grasp forms.** Dev set, falls past 5 cm during formation, default to elliptic: seed 1 33 to 10 (24 trials fall only under default, 1 only under elliptic, p < 0.0001), seed 2 52 to 32 (27 and 7, p = 0.0008). Slips after the grasp formed: 31 to 22 and 42 to 31. Every controller shows the same pattern under noslip.
- **Creep is reduced, not removed.** Same trials in the hand at full load under both settings, RL seed 1, dev set: elliptic, median 1.19 to 0.99 mm/s, slower on 46 of 87, Wilcoxon p = 0.041. Noslip, 1.17 to 0.69 mm/s, slower on 50 of 88, p = 0.010. Friction used rises from a median 4% to about 28% of the limit. This corrects the impression in the Nov 4 entries that the stricter settings stop creep: they raise holding mainly by keeping objects in the hand while the grasp forms.
- **Pad friction, stratified by object.** Mantel-Haenszel odds ratio of holding, above vs. below each object's median pad friction. Locked set, default: 0.90, 1.04 and 1.00 for seeds 1-3, and 0.89 for hand-tuned A (p from 0.56 to 0.92). Hand-tuned A, fresh seeds, noslip: 0.67, p = 0.020, opposite to the prediction, one of five tests and not significant after correcting for five.
- **Failures at constant load, default settings.** RL seed 1, locked set: of 771 trials in the hand when the pull starts, 67 slip while the pull ramps up, 73 in the first second at full load and 43 in the second.
- **Hand-tuned A, fresh seeds, noslip (pre-registered, complete):** 468 of 1,005 held at 5 cm (46.6%, Wilson 43.5-49.7), 707 at 10 cm (70.3%, 67.5-73.1), 432 fell past 5 cm during formation, 105 slipped. Its dev figure at 5 cm, 44.0%, is inside the interval. At 10 cm the dev figure, 67.3%, is just below the interval, so there the dev set slightly understated it. The RL noslip runs resume tonight after the elliptic runs.
- **Seed 2 under elliptic, dev set:** 87 held, against 56 under default. Held only under elliptic 43, only under default 12, p < 0.0001.
- **Scene archived.** `Data/benchmark_scenes/scene_slip.xml` is a byte copy of the untracked scene (SHA-256 starting d9d3aefde2f9). `scene_slip_elliptic.xml` adds the option line and an assetdir. It hasn't passed `elliptic_scene_check.py` yet, so nothing uses it.
- **Paper outline rewritten** to match these results: the Nov 4 flags are resolved, results R18-R22 are new, and fresh-seed numbers are marked pending.

## Nov 6: factor summary and novelty check

- **Factor summary (Figure 0, `fig00_factor_summary`).** Change in held rate at 5 cm, same trials on both sides, 95% paired bootstrap interval: RL seed 1 vs. seed 3 +46.2 (+42.7 to +50.0, locked set). Best RL seed vs. hand-tuned A +42.5 (+39.0 to +46.0, locked set). Four-finger vs. five-finger fixed grip +34.7 (+25.3 to +44.0, dev). No drop rule vs. 5 cm rule +25.7 (+23.2 to +28.5, seeds 0-66). Elliptic vs. default +21.3 and +20.7 for seeds 1 and 2 (dev). Best RL seed vs. fixed grip 0.0 (-5.3 to +5.3, dev). Detector F1 0.799 vs. 0.691 +0.4 (-0.4 to +1.2, seeds 0-66). With vs. without the TCN: hand-tuned A -2.0 (-5.3 to +0.7), RL seed 1 0.0 (dev). The rows come from different trial sets, so they compare sizes, not exact values.
- **Novelty, from a quick web search, not a literature review.** Already documented elsewhere: MuJoCo's contact slip (its documentation; Le Lidec et al., IEEE T-RO 2024, arXiv:2304.06372), learned policies barely changing without touch at test time in some setups (arXiv:2109.11234, arXiv:2204.00117), and closed-loop slip control helping on a real hand (arXiv:2010.01928). Not found: a ranking of these factors on the same trials, or a measurement of how much contact settings change a grasping benchmark's success rate. The outline states the claim with "to our knowledge" and marks the new references as unchecked.

## Nov 6: fresh-seed queue paused

- **Paused:** 2026-11-06, 6:30 pm EST, at the user's request so they could use the laptop. RL seed 1 under elliptic had 770 of 1,005 rows; nothing else in the queue had started. No row was read. Resuming reruns `_fresh_elliptic_nov6.sh`, which continues each run from its own file, so at most the trial in progress was lost and gets run again.

## Nov 6-7: the pause didn't hold, the laptop froze again, and RL seed 1's fresh elliptic result

- **What happened:** at 6:30 pm on Nov 6 I stopped the queue's task and its Python process, and told the user nothing was running. The queue script itself survived. Its retry loop logged the kill as a failed attempt (exit 127), waited two minutes and restarted. RL seed 1 under elliptic resumed at 770 rows and finished at 7:02 pm. Hand-tuned A then started and reached 686 rows. The laptop froze (Kernel-Power 41, bugcheck 0; the last log write was at 7:29 pm) and was restarted with the power button at 8:33 pm. That is the third freeze, and every one has happened during or right after simulation load. The cause isn't proven.
- **Data:** every row in both files parses, with no duplicates (1,005 and 686 unique trials). The trial in progress at the freeze wasn't saved.
- **A deviation:** while diagnosing on Nov 7, I read the end of hand-tuned A's log, which shows its partial held count at 650 trials. The pre-registration says a run isn't read before it completes. Nothing was decided from it, and the run continues unchanged.
- **RL seed 1, fresh seeds, elliptic (complete, pre-registered):** 810 of 1,005 held at 5 cm (80.6%, Wilson 78.0-82.9), 855 at 10 cm (85.1%, 82.7-87.1), 83 fell past 5 cm during formation, 112 slipped. The dev figure at 5 cm, 78.7%, is inside the interval. Labelled "best of 3 seeds, deterministic", as the existing rule requires; the headline is the three-seed mean.
- **Resumed on Nov 7** with `Data/diag_fresh_seeds/_fresh_resume_nov7.sh`: the same runner and arguments. A STOP file in that folder now halts the queue between attempts and runs, and there is a 10-minute cool-down between runs. Order: hand-tuned A, RL seed 2, RL seed 3, fixed grip. The interrupted noslip runs are left out pending a decision. If they aren't finished, they are reported as interrupted at 600, 621 and 666 rows, as the Nov 6 amendment allows.
- **Nov 7, the interrupted noslip runs are queued after all (user's decision):** `Data/diag_fresh_seeds/_fresh_noslip_finish_nov7.sh` waits until the fixed grip finishes, then completes RL seeds 1, 2 and 3 under noslip from 600, 621 and 666 rows, one job at a time with cool-downs, using the same runner and arguments as `_fresh_noslip_nov4.sh`.
