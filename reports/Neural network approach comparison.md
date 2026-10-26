# Neural network approach comparison — PI-TCN-SlipNet vs. published slip-detection literature

Scope: compares the current architecture (`src/models/tcn.py`, `src/models/PI_TCN.py`, trained by
`src/training/train_pitcn.py`) and the current closed-loop policy
(`Synthetic data creation/Simulation/closed_loop_sim.py`) against real published work on
tactile/force-based slip detection and time-series contact classification. Every item below says
what it is, where it comes from, and exactly how it would plug into this codebase. Items are
ranked by how directly they address this project's two actual open problems: (a) run-to-run F1
variance (0.69–0.80 on identical config) and (b) weak closed-loop hold-success (~20%) despite
reasonable static F1.

---

## TL;DR ranking (most actionable first)

| # | Finding | Effort | Addresses |
|---|---|---|---|
| 1 | No random seeding anywhere in the training code | Trivial (10 min) | (a) variance — makes it measurable/reproducible |
| 2 | `decide_action()` reacts to a single noisy instantaneous probability, no smoothing/hysteresis | Small (1–2 hrs) | (b) closed-loop — directly matches a known failure mode in the control literature |
| 3 | Zero normalization layers in a 7-block dilated residual stack | Small–medium (half day) | (a) variance — this is the one deviation from the original TCN paper's recipe that specifically targets training stability |
| 4 | Report mean±std over 5–10 seeds instead of single-run F1 | Small (half day of compute) | (a) variance — reframes the "0.69–0.80" number correctly instead of chasing a single lucky run |
| 5 | Uniform per-finger correction regardless of which fingertip is actually slipping | Small–medium (1 day) | (b) closed-loop — this project already has per-finger features but throws away *which* finger is slipping at the control stage |
| 6 | Gradient clipping | Trivial (1 line) | (a) variance, standard in the original TCN training recipe, currently absent |
| 7 | The "pull_force dominates, 9 hand-engineered friction features added nothing" finding has real precedent — but the precedent also explains why it's a dead end to keep pushing on hand-engineered scalar features | Analysis only, no code | Explains (why static F1 plateaus) — redirects effort |
| 8 | Ensembling across seeds at inference time | Medium (train 3–5 models, average logits) | (a) variance, (b) smoother closed-loop probability signal "for free" |
| 9 | Multi-task auxiliary head (e.g., predict grasp-force-at-settle or friction coefficient as a regression side-target) | Medium | Marginal — literature precedent is mixed/adjacent, not a slam dunk |
| 10 | Attention/GNN fusion across the 5 fingertips | Large (redesign) | Not well-precedented for *this exact* problem in the literature either — see below |
| 11 | Switch to a dense/vibration-sensitive tactile modality | Out of scope (sim limitation, not a code fix) | Addresses *why* raw tactile is flat, but isn't fixable in software |

---

## 1. Architecture families actually used for this task in the literature

### 1.1 TCN (same family as this project) — closest real comparison

**Grover et al., "Under Pressure: Learning to Detect Slip with Barometric Tactile Sensors and a
Temporal Convolutional Neural Network"** (arXiv:2103.13460 / IROS 2021/2022). This is the most
directly comparable published system: a TCN classifying binary static-vs-slip from low-cost
barometric fingertip sensors on a 3-finger gripper.

Concrete differences from this project's `TCN` (`src/models/tcn.py`):
- They build on the original **Bai et al. 2018 generic TCN** (dilated causal convs + residual
  blocks — architecturally almost identical to this project's `TCNBlock`), but then explicitly
  **replace weight normalization with layer normalization, and ReLU with SELU**, and add **20%
  dropout on every layer** (including the two FC layers at the head). This project currently has
  *none* of normalization, SELU, or dropout — dropout was tried and regressed performance per the
  project's own notes, but normalization was apparently never tried in isolation. That's the
  single biggest concrete architectural gap between this project and the one published system
  solving almost the identical problem with the same backbone family. See §2.1 for why this is
  worth testing specifically for the variance problem, not just as a generic "best practice."
- Input is a 1-second window (100 steps at 100 Hz) from a 2×3 barometer array per fingertip —
  i.e., this is **slip detection** (recognizing slip that is already underway from a recent
  window) not **slip prediction** (anticipating slip before it starts). Ground truth comes from
  measured fingertip velocity, not from a commanded/known variable. This matters for comparison
  to this project's "pull_force dominates" finding — see §3.
- They report F1 91.4% on their held-out test split but **91.3%→35–100% per-object range** in
  real-world lift trials (Table II), and **74.5–94.0%** across motion conditions (Table III). They
  do **not** report any random-seed variance — all numbers are single runs, same gap this project
  has. This is useful context: the published baseline for this exact architecture family doesn't
  actually solve the seed-variance problem either; it just doesn't surface it.
- Class imbalance handling: **random undersampling of the majority class every epoch**, not
  inverse-frequency loss weighting. This project's class-weighted CrossEntropy + label smoothing
  is arguably more principled (doesn't throw away data), so this isn't a recommended change, just
  a documented alternative.
- Data augmentation: x-flip / y-flip / 180° rotation (each p=0.25) + small Gaussian noise on the
  raw barometer input, every epoch. See §2.4.

**Gao et al., "Visuo-Tactile-Based Slip Detection Using A Multi-Scale Temporal Convolution
Network"** (arXiv:2302.13564, 2023). Uses a **multi-scale TCN** — multiple parallel dilation/
receptive-field branches fused together, rather than this project's single linear stack of
7 increasing dilations — explicitly to capture slip signatures at different timescales
simultaneously (fast onset vs. slow drift). Reports 96.96% accuracy, beating an LSTM visuo-tactile
baseline and an earlier ConvLSTM result (~82.6%). Relevant mainly as evidence that **multi-scale
fusion beats a single-scale TCN when there's little pre-slip signal to extract and the model needs
to hedge across timescales** — but this is a vision+tactile fusion paper, not a pure-tactile one,
so the headline number isn't apples-to-apples. Multi-scale TCN is a real, moderate-effort
architectural option (parallel branches at e.g. dilations {1,2,4} / {8,16,32} / {64,128} summed or
concatenated before the output head) but not a high-priority item given this project's actual
bottleneck looks like it's data/signal-limited, not capacity-limited (see §3).

### 1.2 LSTM / ConvLSTM / CNN-RNN hybrids

**Zapata-Impata et al., "Learning Spatio Temporal Tactile Features with a ConvLSTM for the
Direction Of Slip Detection"** (Sensors 2019, PMC6387284). ConvLSTM over ~50ms windows,
7-way slip-direction classification from one tactile sensor. **James & Lepora, "Slip detection for
grasp stabilisation with a multi-fingered tactile robot hand"** (arXiv:2010.01928) uses classical
SVMs on the Bristol TacTip ("Tactile Model O") across 11 objects — a reminder that the
classical-ML baseline for this exact multi-fingered-hand slip task is not deep learning at all,
and is still a fair comparison point if ever asked "is a TCN even necessary here."

**MIT CSAIL GelSight work**: VGG-CNN feature extractor feeding an LSTM over video frames from an
optical (image-based) tactile sensor. The important structural difference from this project is the
**sensor modality**, not the recurrent-vs-convolutional choice — see §3.

Takeaway for this project: LSTM/ConvLSTM is not shown anywhere to beat TCN on this task when the
input is force/pressure time series (vs. tactile images) — Gao et al. explicitly beat an LSTM
baseline with their TCN, and Grover et al. chose TCN over LSTM specifically for the long effective
memory via dilation without recurrence's sequential-step cost. This project's choice of TCN over
LSTM is well supported by the literature; switching recurrent is not a recommended change.

### 1.3 Graph neural networks for multi-contact fusion

**TacGNN** (Zhang et al., arXiv:2304.00736, in-hand manipulation on an Allegro Hand) and
**Funabashi et al.** (IEEE RA-L 2022, arXiv:2205.04169, GCN over 1152 distributed tactile
measurements) both build a graph over *individual taxels within a dense tactile array*, with edges
between spatially neighboring sensing elements, specifically because a CNN needs a rectangular
grid and a multi-fingered hand's taxel layout isn't one.

This is **not directly applicable to this project** as-is: this project's "5 fingertips" are 4
scalar channels each (touch, normal, tangential, torque_mag) — there is no spatial taxel grid
within a fingertip to build a graph over. A GNN here would only make sense at the *inter-finger*
level (5 nodes, one per fingertip, edges = kinematic/contact adjacency), which is a much smaller
and less-precedented use of GNNs than what these papers actually do. This is a real architecture
family for the task, but it solves a different sub-problem (spatial taxel fusion within one
sensor) than this project has (coarse per-fingertip scalars, cross-finger fusion). Treat as **not
a good match**, despite being a real technique in the space.

### 1.4 Spiking neural networks

**Lu et al., event-based SCNN on the NeuroTac sensor** (arXiv:2509.09546, IEEE RA-L 2026) — 94.33%
3-class accuracy (no-slip / incipient / gross), and reports detecting incipient slip 360–1840ms
*before* gross slip under gravity-induced slip. **Cordella, Zollo et al.** — SNN for prosthetic
hands modeling SA-I/FA-I mechanoreceptor populations, framed around power efficiency (simulated
162x reduction on Loihi), not accuracy. SNNs here are tied to **event-based/neuromorphic sensor
hardware** (NeuroTac, event cameras) that this project's MuJoCo simulation does not and cannot
produce — MuJoCo sensors output dense per-step scalars, not asynchronous spike events. **Not
applicable** unless the sensing pipeline itself changes to event-based simulation, which is a
different project. Flagged here only because it was explicitly asked for; not recommended.

---

## 2. Specific techniques these papers use that this project's model doesn't

### 2.1 Normalization — the most concrete, most testable gap

This project has **zero normalization layers anywhere** (`TCNBlock.forward` in `src/models/tcn.py`
is conv → residual add → ReLU, nothing else) and also removed dropout after it regressed
performance. The **original TCN paper** this whole family descends from (Bai, Kolter, Koltun,
"An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling,"
2018) uses **weight normalization** on every conv layer as a *load-bearing* part of their reported
stability, not an optional extra. Grover et al.'s slip-specific TCN swapped that for **layer
normalization**. Both published TCNs for sequence tasks keep *some* normalization; this project
has deliberately removed the closest analog (dropout) but never tried the other one.

Why this specifically targets the **variance problem**: a 7-layer residual stack with dilations up
to 64 and no normalization means the scale of activations flowing through the residual adds is
entirely a function of random initialization and early-training dynamics, with nothing per-layer
to re-normalize it. Different random inits can land in different effective-depth/gradient-scale
regimes, which is a textbook mechanism for exactly the kind of "same config, 0.69 vs 0.80 F1"
instability described. This is a **half-day, cheap, directly testable experiment**: add
`nn.LayerNorm` (applied over the channel dimension at each timestep, inside `TCNBlock`, after the
causal conv and before the residual add) or weight-normalize the `CausalConv1d`'s underlying
`nn.Conv1d` via `torch.nn.utils.parametrizations.weight_norm`. Avoid `BatchNorm1d` here — it
computes statistics across the batch dimension per-channel, which is fine for causality (it
doesn't look across time) but behaves badly with this project's padded variable-length sequences
(padding timesteps would pollute the batch statistics unless masked, which `nn.LayerNorm` sidesteps
entirely since it normalizes per-timestep-per-sample). This is the single highest-value "small,
testable change" in this whole report for problem (a).

### 2.2 Attention/fusion across fingertips beyond concatenation

Searched specifically for cross-finger attention / tactile-transformer fusion for slip detection
and **did not find a published example that does this**. The closest related work — **Ayral et
al., "Reactive Slip Control in Multifingered Grasping"** (arXiv:2602.16127, 2026) — fuses
piezoelectric slip cues with piezoresistive contact mapping via a fixed hybrid pipeline, not
learned attention, and explicitly calls out that "the force increase in typical reactive slip
control is often chosen arbitrarily, with the same force applied by each finger" as a known
limitation of the field — which is also exactly what this project's `decide_action()` does (see
§4.2). This is a real, named gap in the literature, not just in this project. **Verdict: don't
treat "add cross-finger attention" as a well-precedented fix borrowed from a paper that solved
this** — it would be a genuinely novel contribution here, not a known technique being ported in.
If pursued, the much cheaper version is not an attention layer but simply **routing per-finger
slip probability to the control policy instead of one global scalar** — see §4.2, which is
low-effort and backed by the same literature as a known omission worth fixing.

### 2.3 Multi-task learning (auxiliary regression heads)

No paper found that jointly trains slip classification with displacement/force prediction as a
true shared-backbone multi-task objective for this exact problem. Closest matches: a GelSight
force-field-entropy paper (arXiv:2303.00935, ~95.6% accuracy) uses tactile-derived force-field
estimates as an *input feature* to a separate classifier, not as a joint training target. A 2022
IEEE RA-L paper trains **grasp-stability-before-lift and slip-after-lift together** in one
multi-output framework and reports the joint model is more reliable than either single-phase
model. That's the one solid precedent, but it's joint classification across two *phases*, not
classification + regression on a physical quantity.

Applied to this project: a plausible version would be an auxiliary head predicting
`grasp_force_at_settle` or the friction coefficient (already available as training labels, already
used as PI-TCN *inputs* in the 25-channel variant) as a regression side-loss, on the theory that
forcing internal representations to be physically grounded could stabilize training. Given the
project's own finding that **9 different hand-derived friction-ratio features already failed to
add discriminative power**, there's a real chance this just adds a second loss term that correlates
near-perfectly with the already-available pull_force input and adds optimization noise rather than
signal. This is a **medium-effort, uncertain-payoff experiment** — rank it below the normalization
and seeding fixes, worth trying only after those are exhausted.

### 2.4 Data augmentation specific to tactile time series

Two concrete, precedented recipes:
- **Grover et al.** (barometric TCN, directly comparable): per-epoch random spatial flips/rotation
  of the sensor array (not directly transferable — this project's fingertip channels aren't a
  spatial grid) + **small additive Gaussian noise on the raw input**, which *is* directly
  transferable: add noise to `touch`/`normal`/`tangential`/`torque_mag` channels during training
  only.
- **A 2022 IEEE Sensors Journal study** on a 3-finger-gripper tactile time-series dataset tested
  jitter, scaling, magnitude warp, time warping, and cropping across MLP/LSTM/CNN/ConvLSTM, and
  found the combination of **scaling + time-warping** gave the best, statistically significant
  gain on a deep CNN.

Caveat specific to this project: both of these augmentation recipes assume the raw signal carries
usable information that noise/warping helps the model generalize over. This project's own finding
is that the raw tactile channels are **already flat/uninformative**, and the dominant signal is a
noiseless, deterministically-known `pull_force`. Augmenting with noise on the *tactile* channels
is low-risk and cheap to try, but augmenting or warping `pull_force` itself would need care — it's
not sensor noise, it's a commanded control variable, and corrupting it synthetically could just
teach the model to ignore the one channel that currently works. **Recommendation: if tried, add
small Gaussian noise only to the 20 tactile/force channels, leave `pull_force` and `phase`
untouched**, and treat this as a low-priority, low-effort experiment rather than a fix for either
of the two headline problems.

### 2.5 Uncertainty / calibration

**Joyce et al.** (arXiv:2506.20045) and the variational-belief grasping paper (arXiv:2604.25897)
both train models to flag when their own prediction is unreliable, and route to safer behavior
instead of acting on an overconfident point estimate. This is conceptually the most relevant
*category* of technique to problem (b): `predict_slip()` in `closed_loop_sim.py` already exposes a
softmax probability (not a hard class), which is good, but `decide_action()` throws away
everything except the most recent single-timestep value — no variance/confidence estimate, no
temporal consistency check. A cheap, literature-adjacent version of "uncertainty-aware control"
here is not a new calibration head, it's **using the probability trajectory already sitting in
memory** (`feature_history`/the rolling window) to require, e.g., N consecutive above-threshold
calls before acting, which is a debouncing/hysteresis technique, not formal uncertainty
quantification, but addresses the same underlying problem (don't act on one noisy sample) far more
cheaply. See §4.2.

### 2.6 Ensembling

**A 2022 IEEE RA-L multi-phase paper** explicitly ensembles six different neural network
architectures for stability+slip prediction and reports it "is a viable and practical choice to
boost the overall performance." This is a real, cheap technique directly applicable to problem
(a): train 3–5 models from different random seeds (once seeding is added, per item #1), average
their softmax outputs at inference (in `predict_slip()`, loop over `_model` list and mean the
probs before returning). This doesn't fix *why* training is unstable, but it directly fixes the
*symptom* that matters operationally — a single run landing at F1 0.69 instead of 0.80 — by never
shipping a single unlucky run. This is a **medium-effort (3–5x training time, trivial inference
change), high-confidence win** for both making deployed performance consistent and smoothing the
probability signal fed to the closed-loop controller.

---

## 3. Precedent for "raw tactile signal is flat, external commanded force dominates"

This is a real, explainable pattern, and the literature points to three distinct reasons it shows
up here but not in the papers reporting strong raw-tactile pre-slip signal:

**(a) Sensor bandwidth/resolution gap.** The papers that *do* report a genuine pre-slip tactile
signal are specifically exploiting **high-frequency micro-vibration** at the contact interface,
not force/torque magnitude:
- **Teshigawara et al.** (Journal of Robotics and Mechatronics, 2012) found slip precursors as
  **kHz-range frequency components** in a dedicated vibration sensor's output, appearing just
  before gross slip.
- The **Bielefeld 16×16 tactile array** runs at up to **1.8 kHz** specifically because incipient
  slip shows up as micro-vibration between the surfaces, and a network is trained on the spectral
  signature of that vibration — not on force magnitude at all.
- **Komeno & Matsubara** (arXiv:2402.11879) go further and *actively inject* a white-noise
  vibration into the fingertip and read the change in propagated spectrum, explicitly because "the
  tiny signals from incipient slip are quickly buried by environmental noise" in a passive reading.

This project's per-fingertip features are **4 scalar magnitudes per timestep** (touch, normal
force, tangential force, torque magnitude) — there is no vibration/spectral channel at all, and no
indication the simulation timestep/sampling rate is anywhere near kHz-scale vibration-sensing
resolution. Structurally, this project's sensors cannot represent the signal that the papers
reporting strong raw-tactile precursors are actually using. This is not a modeling failure; it's a
sensing-modality gap, and it is **out of scope for a software-only fix** — it would require adding
a synthetic high-frequency vibration/micro-slip signal to the MuJoCo contact model and sensor
export pipeline, which is a data-generation change, not a network-architecture change.

**(b) Deterministic simulation vs. noisy real hardware.** `Howe & Cutkosky` (DTIC ADA255725, an
older acoustic-emission slip study) report that "detection of incipient slip was generally
unsuccessful" even though a transition phase between full contact and gross slip could be
verified to exist — i.e., published precedent exists for the signal being *present in principle but
not practically extractable* from the available sensing, even on real hardware. In this project's
case the situation is arguably worse for feature engineering, not better: MuJoCo's contact solver
and `pull_force` are **noiseless and deterministic**, and `pull_force` is a known, directly-
commanded ramp whose relationship to the Coulomb slip condition (`pull_force` vs. `friction ×
normal_force`) is close to an exact physical threshold crossing in this simulated setting. That
would explain *both* of this project's own findings at once: `pull_force` dominates (it's nearly a
sufficient statistic for the deterministic slip condition already), **and** 9 hand-derived
Coulomb-ratio features all failed to add anything beyond it (there's no residual signal left to
extract once `pull_force`, `grasp_force_at_settle`, and `mu_combined` are already inputs — those
three numbers are close to fully determining the noiseless simulated slip event). This is this
report's own synthesis, not a direct claim from any single cited paper, but it is consistent with
and explains both the Grover et al. "detection not prediction" framing (§1.1) and the Howe/Cutkosky
negative result above.

**(c) Labeling/task design differences.** Grover et al.'s task is explicitly **slip detection**
(classify the last 1 second of already-occurring motion) rather than **slip prediction**
(anticipate before onset) — and they still only reach 91% F1 with substantial per-condition spread
(74.5–94%), using a sensor modality designed around spatial pressure redistribution. If this
project's per-timestep label is effectively asking the model to *anticipate* slip ahead of the
actual displacement threshold being crossed, that is a strictly harder task than what most cited
baselines solve, which is additional (though less certain) context for why static F1 plateaus
around 0.77–0.80 rather than the 90%+ seen in some tactile-detection papers — those papers' sensors
and task framing are measuring something closer to "has slip already started," while physics-
informed friction-ratio features were presumably aimed at prediction.

**Practical implication:** stop investing further effort in new hand-derived scalar friction/
force-ratio features — the literature and this project's own ablations both point to this being a
genuinely saturated direction given the sensing modality and deterministic simulator, not a
modeling or feature-engineering shortfall. If more signal is wanted, it has to come from either (i)
a genuinely different/denser simulated sensor channel (vibration, spatial taxel array, contact
patch imaging) — a data-generation project, not a model change — or (ii) accepting that in this
deterministic sandbox, `pull_force`-driven prediction near the Coulomb threshold *is* close to the
ceiling, and redirecting effort to the closed-loop control side, which independently has its own,
separately fixable problems (§4.2).

---

## 4. Concrete fixes for this project's two actual problems

### 4.1 Run-to-run training variance (F1 0.69–0.80 on identical config)

Verified directly in the code: there is **no `torch.manual_seed`, `np.random.seed`, or
`random.seed` call anywhere under `src/`** — `src/Utils/seed.py` exists but is an empty file, never
populated or imported. `src/training/train_pitcn.py` constructs the model, optimizer, and
`DataLoader(shuffle=True)` with zero seeding. This alone guarantees different weight
initialization and different batch order every run, which is a sufficient (if not necessarily
complete) explanation for the reported spread.

Ranked fixes:
1. **Seed everything** (`torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`,
   `random.seed`, plus `torch.backends.cudnn.deterministic = True` /
   `torch.backends.cudnn.benchmark = False`) in `train_pitcn.py`, using `src/Utils/seed.py` (fill
   in the empty stub) as the single source of truth. **This makes a given run reproducible, but
   does not by itself reduce the spread across different seeds** — it converts "nondeterministic
   mystery" into "measurable seed sensitivity," which is the correct first step per both Jordan's
   "Calibrated Chaos" (arXiv:2304.01910) and Tan et al.'s empirical study (ASE 2020, showing even
   *fixed-seed* repeated runs of the same CNN still spread ~2.9 accuracy points due to GPU-level
   nondeterminism) — i.e., full bit-for-bit reproducibility is not fully achievable on GPU even
   with seeding, but it tightens the distribution substantially and is close to free to add.
2. **Add normalization inside `TCNBlock`** (§2.1) — the one architectural lever with direct
   precedent in both the original TCN paper and the one published slip-specific TCN, aimed
   squarely at reducing activation-scale sensitivity to initialization.
3. **Add gradient clipping** (`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=...)`
   before `optimizer.step()`) — a one-line addition, standard in the original TCN training recipe,
   currently absent from `train_pitcn.py`, and a classic stabilizer for deep residual conv stacks.
4. **Stop reporting single-run F1 as "the" number.** Per Jordan (arXiv:2304.01910) and the Purdue
   ASE 2020 study, the methodologically correct comparison across configs is mean ± std over 5–10
   seeded runs, not best-of-one. Given this project is already seeing 0.69–0.80, that range *is*
   the actual result; report it as such rather than treating 0.80 as "the" model and 0.69 as noise
   to discard.
5. **Ensemble across seeds at inference** (§2.6) — once multiple seeded checkpoints exist from
   step 1/4 anyway, averaging their softmax outputs in `predict_slip()` is nearly free and directly
   improves the *deployed* consistency, independent of whether the underlying training-variance
   cause is ever fully diagnosed.

### 4.2 Weak closed-loop performance (~20% success) despite reasonable static F1

Read directly from `Synthetic data creation/Simulation/closed_loop_sim.py`. Three concrete,
literature-backed issues in the control policy itself, independent of model quality:

1. **No smoothing/hysteresis on the slip signal.** `decide_action()` (line ~143) acts on
   `slip_probability`, a single instantaneous value returned by `predict_slip()` from only the
   *last* timestep of the current window, with a flat `if slip_probability <= 0.3: return {}`
   gate. The *same* 0.3 threshold is used to turn correction on and off — a classic bang-bang
   setup with no deadband, even though `max_slip_probability`/`last_slip_probability` history is
   already being tracked in `run_trial()`. A patent on gripping-force suppression control
   (US 8644986) explicitly documents the standard fix in this exact domain: **gripping-force
   *increase* should react quickly, but force *release*/relaxation should happen more slowly**,
   and the raw slip signal should be **smoothed with a moving average or low-pass filter** before
   it drives control, specifically to avoid chatter from noisy instantaneous readings. This
   project's `feature_history` buffer already contains the last 300 steps — a trailing moving
   average of `predict_slip()` outputs (or even just requiring 2–3 consecutive above-threshold
   calls, since inference already only runs every 5 steps) is a **small, cheap, directly
   precedented fix**, and is the single most actionable item for problem (b).
2. **No per-finger localization in the control response.** `decide_action()` applies the *same*
   `step = STEP_FRACTION * slip_probability * (hi - target)` correction to every driven actuator
   identically, derived from one global scalar probability. **Ayral et al.** (arXiv:2602.16127,
   2026) explicitly flag this exact pattern — "the force increase in typical reactive slip control
   is often chosen arbitrarily, with the same force applied by each finger" — as a known
   limitation their null-space QP approach is built to fix. This project already computes
   per-fingertip features (`touch`, `normal`, `tangential`, `torque_mag` × 5 fingers) and feeds
   them into the model, so the information needed to distinguish "which finger is slipping" exists
   in the pipeline; it's discarded at the point the single scalar softmax output is produced.
   A **medium-effort, concretely scoped fix**: either (i) run the existing model with per-finger
   masking/attribution to approximate which finger's channels are driving the slip prediction at a
   given step, or (ii) the more principled version, train the model (or a small per-finger head)
   to output 5 independent slip probabilities, one per fingertip, and have `decide_action()`
   correct each actuator's grip based on its own finger's probability rather than one shared
   value. This does not require full cross-finger attention (§2.2, which has no real precedent to
   lean on) — just not collapsing the 5-finger signal to one scalar before it reaches the
   controller.
3. **Control-loop latency vs. model receptive field.** `predict_slip()` is only queried every 5
   simulation steps (`step % 5 == 0`), and reacts with a position-target nudge scaled by
   `STEP_FRACTION=0.35` of remaining actuator range — worth sanity-checking against the ~35-40ms
   sensing-to-command latency and sub-50ms stabilization window reported as achievable in Ayral et
   al.'s hybrid approach, to see whether this project's 5-step/`STEP_FRACTION` combination is
   actually reacting fast enough relative to how quickly simulated displacement crosses the
   slip threshold once pull_force ramps past the Coulomb limit. This is a tuning/diagnostic check,
   not a code change, but worth doing before concluding the *model* is the bottleneck in the 20%
   success rate — a well-calibrated model fed through a laggy, unsmoothed, un-localized controller
   can plausibly produce exactly this kind of static-F1-good/closed-loop-bad gap on its own.

---

## 5. What's a big undertaking vs. a small testable change — explicit honesty check

**Small, cheap, do-these-first (hours to a day each):**
- Seed everything (§4.1.1)
- Gradient clipping (§4.1.3)
- Smoothing/hysteresis in `decide_action()` (§4.2.1)
- Gaussian noise augmentation on tactile channels only (§2.4)

**Medium effort, worth trying after the above (a day to a few days):**
- Add LayerNorm (or weight-norm) inside `TCNBlock` and re-run the seed-variance comparison (§2.1,
  §4.1.2)
- Multi-seed ensembling at inference (§2.6, §4.1.5)
- Per-finger slip probability instead of one global scalar feeding `decide_action()` (§4.2.2)
- Report mean±std over seeded runs as the real metric (§4.1.4)

**Large undertakings, low confidence of payoff given this project's specific findings — deprioritize:**
- Cross-finger attention/transformer fusion (§2.2) — not even established in the literature for
  this exact problem; would be novel work, not a known fix being ported in.
- GNN fusion (§1.3) — solves a different problem (spatial taxel fusion) than this project has
  (coarse per-finger scalars).
- Multi-task auxiliary regression heads (§2.3) — mixed precedent, and this project's own ablations
  (9 failed hand-derived features) suggest there's little residual signal left for a second loss
  term to exploit.
- Switching to a genuinely vibration-sensitive or spatial-taxel sensor model (§3a) — would likely
  be the only way to get real pre-slip tactile signal per the cited precedent, but it's a
  data-generation/simulation change, not a network change, and is explicitly out of scope here.
- Multi-scale TCN branches (§1.1) — plausible but unlikely to help given the bottleneck looks
  signal-limited (§3), not capacity-limited.

---

## Sources

- Bai, Kolter, Koltun. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling." 2018. (Origin of the TCN architecture family; weight-norm + dropout as part of the original recipe.)
- Grover et al. "Under Pressure: Learning to Detect Slip with Barometric Tactile Sensors and a Temporal Convolutional Neural Network." [arXiv:2103.13460](https://ar5iv.labs.arxiv.org/html/2103.13460)
- Gao et al. "Visuo-Tactile-Based Slip Detection Using A Multi-Scale Temporal Convolution Network." [arXiv:2302.13564](https://arxiv.org/pdf/2302.13564)
- Zapata-Impata et al. "Learning Spatio Temporal Tactile Features with a ConvLSTM for the Direction Of Slip Detection." Sensors 2019. [PMC6387284](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6387284/)
- "Combined tactile/visual GelSight+camera slip detection." [arXiv:1802.10153](https://alphaxiv.org/abs/1802.10153)
- MIT CSAIL. "Slip Detection with Deep Neural Network by Using GelSight Touch Sensor." [csail.mit.edu](https://www.csail.mit.edu/research/slip-detection-deep-neural-network-using-gelsight-touch-sensor)
- James & Lepora. "Slip detection for grasp stabilisation with a multi-fingered tactile robot hand." [arXiv:2010.01928](https://arxiv.org/pdf/2010.01928)
- Ayral et al. "Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and Internal-Force Optimization." [arXiv:2602.16127](https://arxiv.org/html/2602.16127v1)
- Lu et al. "A Neuromorphic Incipient Slip Detection System using Papillae Morphology." [arXiv:2509.09546](https://arxiv.org/pdf/2509.09546)
- Zhang et al. "TacGNN: Learning Tactile-based In-hand Manipulation with a Blind Robot using Hierarchical Graph Neural Network." [arXiv:2304.00736](https://ar5iv.labs.arxiv.org/html/2304.00736)
- Funabashi et al. "Multi-Fingered In-Hand Manipulation with Various Object Properties Using Graph Convolutional Networks and Distributed Tactile Sensors." IEEE RA-L 2022. [arXiv:2205.04169](https://arxiv.org/pdf/2205.04169)
- "Learning to Detect Slip through Tactile Estimation of the Contact Force Field and its Entropy." [arXiv:2303.00935](https://arxiv.org/pdf/2303.00935)
- IEEE RA-L 2022 multi-phase grasp-stability + slip-prediction ensemble paper (six-architecture ensemble). [unpaywall: 10.1109/LRA.2022.3151260](https://unpaywall.org/10.1109%2FLRA.2022.3151260)
- Joyce et al. "Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception." [arXiv:2506.20045](https://arxiv.org/html/2506.20045v1)
- "Variational Neural Belief Parameterizations for Robust Dexterous Grasping under Multimodal Uncertainty." [arXiv:2604.25897](https://arxiv.org/pdf/2604.25897)
- 2022 IEEE Sensors Journal tactile time-series data-augmentation study (jitter/scaling/time-warp/crop across MLP/LSTM/CNN/ConvLSTM).
- Teshigawara et al. "Highly Sensitive Sensor for Detection of Initial Slip and Its Application in a Multi-Fingered Robot Hand." Journal of Robotics and Mechatronics, 2012. [fujipress.jp](https://www.fujipress.jp/jrm/rb/robot002400020298/)
- Komeno & Matsubara. "Incipient Slip Detection by Vibration Injection into Soft Sensor." [arXiv:2402.11879](https://ar5iv.labs.arxiv.org/html/2402.11879)
- Howe & Cutkosky. Acoustic emission slip-transition study. [DTIC ADA255725](https://apps.dtic.mil/sti/pdfs/ADA255725.pdf)
- Gripping-force suppression control patent (asymmetric increase/release rate, slip-signal smoothing). [US 8644986](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8644986)
- Jordan. "Calibrated Chaos: Variance Between Runs of Neural Network Training is Harmless and Inevitable." ICLR 2024. [arXiv:2304.01910](https://ar5iv.labs.arxiv.org/html/2304.01910)
- Tan et al. "Studying the Effect of Randomness on Neural Network Training." ASE 2020. [cs.purdue.edu](https://www.cs.purdue.edu/homes/lintan/publications/variance-ase20.pdf)
- "Learning to be Reproducible: Custom Loss Design for Robust Neural Networks." [arXiv:2601.00578](https://arxiv.org/pdf/2601.00578)
- Calandra et al. "The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?" CoRL 2017. [arXiv:1710.05512](https://arxiv.org/pdf/1710.05512)
