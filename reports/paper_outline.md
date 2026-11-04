# Paper outline: slip detection and closed-loop grasp control in simulation

**Status:** working draft, last updated Nov 4, 2026, evening. R9, R12, R13 and R14 are filled in, and R15-R17 are new. **Points marked `[FLAG Nov 4]` are contradicted or changed by the Nov 4 review and need rewriting before use. I flagged them and did not rewrite them, because the framing and the headline are your call** (see `reports/path_to_70_review_2026-11-04.md`). The seeded F1 items wait on your runs.

**How to read this file**
- Every point is something to say in that section, in the order it should appear. Rewrite it in your own words.
- `[PENDING: ...]` means the run hasn't finished. Don't estimate these numbers.
- `[YOUR CALL: ...]` means a decision that is yours.
- `[FLAG Nov 4: ...]` means a Nov 4 finding changes this point. The evidence is in `results.md`, "Nov 4 code review and diagnostics".
- `[n]` points to the numbered reference list at the end.
- Numbers come from `src/training/results.md`, or from `paper_stats.py` and `rl_locked_checks.py` in `src/analysis/nov3_drop_analysis/` (run them from the repo root to reproduce them). Result files are named in each section.

**How far the references were checked:** every reference was matched to Crossref, arXiv or the publisher by its exact identifier, except where its entry says otherwise, so they are all real papers with the right authors and venues. I read abstracts, or search summaries of them, for some, marked **[A]**, and only checked publication details for the rest, marked **[M]**. I did not read any paper in full. Read each paper before relying on it, especially **[M]** ones. Where a point below says what an [M] paper found, that comes from memory, so confirm it first.

---

## 0. The core of the paper

**Central question:** How much do a slip detector's static score and a benchmark's definition of "success" tell you about real closed-loop grasp performance?

**Why this framing:** the strongest results are about measurement and rigor, not a high success rate. It is also honest about where the project stands. Don't present this as "we reached X%".

**The four findings to carry the paper**
1. How "held" is defined changes the result a lot. The same 1,005 trials score 44.6%, 31.7% or 18.9%.
2. In this setup, a large static F1 gap (0.691 vs. 0.799) produced no closed-loop difference under the hand-tuned controller. `[FLAG Nov 4: stronger evidence now exists. Removing the TCN signal changes 0 of 450 RL outcomes on the dev set (R15). The "actuator headroom" explanation does not hold up, though (R6).]`
3. A learned controller can beat hand-tuned control, but training doesn't reliably produce one. Three repeats of the same setup held 61.2%, 42.2% and 15.0% of 960 trials at 5 cm, against 18.8% for hand-tuned, and the third is worse than hand-tuned. `[FLAG Nov 4: a fixed script with no policy or TCN matches seed 1 on the dev set, 86 vs. 86 of 150 (R16). So don't attribute this lead to learned feedback. It may be the grip.]`
4. Failures fall into two kinds that depend on object shape: objects falling before the pull, and objects slipping during it. A few shapes almost never hold. `[FLAG Nov 4: much of the "few shapes almost never hold" part looks like contact settings (R17).]`
5. `[FLAG Nov 4, YOUR CALL: a possible new finding. Contact solver settings alone moved every controller by 20-30 points on the dev set, more than any detector or controller difference here (R17). Not adopted, and fresh-seed runs are pending.]`

**Don't claim**
- That this is physics-informed. The current Model B has no physics loss (see 5.5).
- That any of this is novel, unless your literature check supports it. Offline metrics failing to predict closed-loop results is already known in driving research [37, 38], so the possible contribution is showing it for tactile slip detection and grasping, with these specifics.
- Anything about real hardware. This is simulation only.
- That RL training beats hand-tuned control on average. Two of three policies do, one doesn't, and the three-repeat mean isn't established as better (R9). 
- `[FLAG Nov 4]` That a stricter contact setting is physically correct. It shrinks a documented solver artifact, and pad friction still doesn't predict holding (R17).
- `[FLAG Nov 4]` That the slip signal improves RL. It changes no outcome at test time (R15).

**Numbers not to use**

| Number | Why not |
|---|---|
| 58.6% / 57.9% as success rates | Short test (200 steps at full pull) and no drop rule. Use only to show the scoring effect |
| 40.0% / 33.3% as headline results | 45 trials, no drop rule |
| 64.4% as "the RL result" | Best of 3 training repeats on 45 trials. Must be labelled "best of 3 seeds, deterministic" |
| F1 0.773 beside closed-loop Model A results | Those runs used the F1 0.691 file |
| "Model B is worse" or "Model B wins" | Tied in closed loop. The 0.799 also relied on 40 epochs and a tuned threshold |
| 61.2% as "the RL result" | Best of 3 repeats. The pre-registered rule says a single best seed is never the headline. Label it "best of 3 seeds, deterministic" |
| Seed 1's 72.7% on held-out objects as reaching 70% | Those 4 shapes are easier for every controller here, and it's a subset (R9) |
| VecNormalize's 2.2% as a verdict on VecNormalize | The number is real (the eval-path check passed), but it is one unseeded run on 45 trials (R12) |
| "89 episodes per object", "published range 27-95%", "pure randomness" | Corrected in `results.md`, "Corrections to tonight's audit" |
| 81.3% or 78.7% as a result | Seed 1 is best of 3, on the much-used dev set, under settings chosen after seeing scores and not adopted (R17). Use only what the fresh-seed runs give, and only if you adopt a setting |
| 61.3% for the slower fixed grip as an unbiased figure | Best of several grips on 150 trials, with speeds read from seed 1 (R16) |

**`[YOUR CALL]` before writing**
- **Which contact settings are the benchmark (R17).** This comes first, because it changes every number below. Options and my recommendation are in `reports/path_to_70_review_2026-11-04.md`.
- Target venue and format (length, sections, citation style). References below are numbered and can be reformatted.
- Which Model A to cite. My recommendation: label closed-loop results as F1 0.691 and report static F1 as a range (see R2).
- Whether to report 5 cm, 10 cm or both. Plan so far: both, with 5 cm primary for RL because it was the existing rule.
- RL evaluation mode, because the pre-registered check tripped (R13). My recommendation: deterministic as primary, since every locked-set run used it and it's the common way to evaluate PPO, with the stochastic numbers reported beside it.
- RL headline number. The pre-registered rule makes it the three-repeat mean and SD (39.5%, SD 23.2, at 5 cm), with 61.2% shown as "best of 3 seeds, deterministic". Changing that now would be changing a rule after seeing results, so I recommend keeping it.

---

## 1. Title options

1. Does a Better Slip Detector Mean a Better Grasp? Static Metrics, Success Definitions and Closed-Loop Control on a Simulated Five-Fingered Hand
2. Measuring What Matters in Slip-Aware Grasping: A 1,005-Trial Simulation Study of Hand-Tuned and Learned Control
3. When Grasp Success Depends on the Definition: Evaluating TCN Slip Detection and PPO Control in MuJoCo

---

## 2. Abstract (one sentence or two per point)

1. **Problem:** stable grasping needs detecting and responding to slip, and detectors are usually judged by static classification scores.
2. **Gap:** static scores may not predict closed-loop performance, and success criteria can hide failures.
3. **Method:** a simulated five-fingered ORCA hand in MuJoCo, 15 object shapes, a TCN slip detector from 20,000 simulated trials, compared under a hand-tuned controller and PPO.
4. **Evaluation:** 1,005 paired trials per condition, evaluation rules written in advance, and scoring under several drop definitions.
5. **Result 1:** hand-tuned success on identical trials was 44.6%, 31.7% or 18.9%, depending on whether objects falling before the pull counted as failures.
6. **Result 2:** two detector checkpoints with F1 0.691 and 0.799 tied in closed loop (paired test p = 0.45 at 5 cm).
7. **Result 3:** across three PPO training repeats, held rates at 5 cm were 61.2%, 42.2% and 15.0% (mean 39.5%, SD 23.2), against 18.8% for hand-tuned control on the same 960 trials. Learned control can be much better, but training didn't reliably produce it. `[YOUR CALL: confirm the evaluation mode, R13]` `[FLAG Nov 4: "learned control can be much better" needs rewriting, see finding 3.]`
8. **Conclusion:** grasp benchmarks should verify the object is still held, and detectors should be evaluated in closed loop.

---

## 3. Introduction

**Paragraph 1: Why grasping is hard**
- Robots need to hold objects they haven't seen, under uncertain friction and load.
- Contact mechanics and grasp stability are the classical foundations [1, 2]. Data-driven grasping has grown alongside analytic methods [3].
- Humans rely heavily on touch: fingertip signals report contact and slip, and grip force is adjusted to load [4, 5]. Use this to motivate tactile feedback.

**Paragraph 2: Slip detection as a route to stable grasping**
- Slip detection has a large literature across sensors and methods, reviewed in [6].
- Examples: a tactile grasp controller on a PR2 with human-inspired signals [7], random forests predicting slip inside a stabilizing feedback loop on unknown objects [8], learned models using GelSight tactile and visual data [11], and incipient-slip monitoring with GelSlim [12].
- Tactile input can also improve grasp outcome prediction [13].

**Paragraph 3: Learned control for dexterous hands**
- Model-free RL has scaled to high-dimensional hands in simulation [15] and has transferred to real hands with heavy randomization [16].
- Tactile features have been combined with RL for regrasping [14].
- Low-cost open hands such as ORCA make this kind of work more accessible [17].

**Paragraph 4: The gap this paper addresses**
- Detectors are usually reported with static metrics like accuracy and F1.
- In autonomous driving, offline prediction error isn't necessarily correlated with driving quality, and models with identical error can behave very differently [37]. Task-agnostic metrics also treat all errors alike [38].
- RL results depend heavily on seeds and implementation details, and reporting single point estimates from few runs can mislead [33, 34, 35, 36].
- Grasp benchmarking work stresses repeatable conditions and explicit success criteria [39, 40].
- Point to make: the effect of these measurement choices on a slip-aware grasping pipeline should be measured directly.

**Paragraph 5: What this paper does (contributions)**
1. A 1,005-trial paired closed-loop benchmark on a simulated five-fingered hand across 15 shapes.
2. A direct measurement of how the success definition changes the result on identical trials.
3. A closed-loop comparison of detector checkpoints with different static F1.
4. A comparison of hand-tuned control and PPO on the same trials, reporting all training repeats.
5. Failure analysis by object and failure type, identifying where the limits are.
6. Evaluation rules written before results, with public code and result files. `[YOUR CALL: confirm the repo is public before saying this]`

---

## 4. Related work (optional section, or merge into the introduction)

**4.1 Slip detection and tactile grip control**
- Survey [6]. Tactile classifiers and force estimation with biomimetic sensors [9, 10]. Vision-based tactile sensing [11, 12].
- Grip controllers using tactile events [7, 8].
- Difference here: fingertip force and torque signals in simulation, with a causal TCN, and the focus on evaluation.

**4.2 Temporal convolutional networks**
- Dilated causal convolutions came to prominence in WaveNet [24].
- TCNs for action segmentation [23], and a general empirical case that simple TCNs can outperform common recurrent networks on sequence tasks [22].
- Why they suit this task: causal, so usable in real time, with a fixed and known history window (255 steps here).

**4.3 Physics-informed machine learning**
- PINNs embed physical laws into training [25], reviewed in [26].
- Say clearly that this paper doesn't yet use a physics loss. Model B only adds physics-related input features. Present a physics loss as future work.

**4.4 RL for dexterous manipulation**
- PPO [30], and the Stable-Baselines3 implementation used here [31].
- Dexterous RL [15, 16], sim-to-real and randomization [16, 21].
- Residual approaches that learn on top of a controller [46, 47], relevant to future work.

**4.5 Reproducibility and evaluation**
- RL variance and reporting [33, 34], implementation effects [35, 36].
- Leakage in ML-based science [41], relevant to the feature exclusions in 5.5.
- Preregistration separates hypothesis generation from testing [42].

---

## 5. Methods

### 5.1 Simulation platform
- MuJoCo [18] version 3.11.0, with its soft-contact model [19, 20]. Python 3.14.3, Stable-Baselines3 2.9.2 [31], Gymnasium [32], PyTorch.
- ORCA five-fingered hand model [17], in `third_party/orcahand_description/v2/scene_slip.xml`.
- 10 actuators are driven, an MCP and a PIP actuator per finger. Other actuators are held at a neutral value.
- Sensors on each of 5 fingertips: touch, force and torque.
- Physics step 0.002 s, so a 2,300-step trial is 4.6 s of simulated time. `[CHECK: confirm with model.opt.timestep]`

### 5.2 Objects and randomization
- 15 objects (Table 1):
  - **Primitives (8):** cube, sphere, rectangular prism, cylindrical prism, capsule, ellipsoid, flat disk, rod.
  - **Meshes (4):** cone, square pyramid, triangular prism, hexagonal prism.
  - **Compounds (3):** L-block, dumbbell, plus-shaped cross.
- Mass is computed from geometry at PLA density (1,240 kg/m³, solid infill).
- Per-trial randomization, seeded so each trial is repeatable:
  - Position ±5 mm in the plane
  - Yaw 0-360°, roll and pitch ±10°
  - Size and mass ±10%
  - Object friction ±15% around about 0.4
  - Fingertip pad friction 0.35-1.2 per finger
- Explain the friction choice: MuJoCo combines contact friction using the element-wise maximum, so a fixed pad friction of 1.0 would have made object friction meaningless. Randomizing pad friction makes it causal. Source: docstring in `generate_trials.py`, confirmed empirically there.

### 5.3 Trial protocol
1. **Ramp (500 steps):** the object is weightless through gravity compensation while the fingers close.
2. **Settle (300 steps):** gravity on. The reference position for slip is recorded at the end of this phase.
3. **Pull:** a downward force ramps over 500 steps up to 1.5x the object's weight, then holds for 1,000 steps.

Total 2,300 steps. Grip target for hand-tuned control: squeeze 0.95 of the actuator range.

### 5.4 Success definitions (a core methods section)
- **Slip:** displacement from the settle reference exceeds 15% of the object's characteristic length (6 mm for a 4 cm cube).
- **Formation fall:** drop in height from spawn during ramp and settle.
- Report every result under three rules:
  - **No drop rule:** held means no slip. This is the original hand-tuned scoring.
  - **10 cm rule:** also fails trials whose object fell more than 10 cm before the pull.
  - **5 cm rule:** also fails trials whose object fell more than 5 cm. This is RL's training and evaluation rule.
- Explain the flaw: displacement starts at the end of settle, so an object already on the floor barely moves and scores as held.
- Disclose that the 10 cm rule was chosen after seeing results. Justify it with data: for hand-tuned control, any cutoff from 7 to 20 cm gives the same score within 3 of 1,005 trials (316-319 for Model A).

### 5.5 Slip detection models
- **Data:** 20,000 simulated trials, with squeeze 0.30-1.00, pull 0.2-5.0x weight, ramp 400-600 steps, and recorded sensor noise setting 0.05.
- **Filter:** trials with no finger contact are dropped, leaving 17,514 (10,588 slip, 6,926 no-slip).
- **Split:** random by trial: 12,259 train, 2,627 validation, 2,628 test. Note that this tests new trials of the same objects, not new objects.
- **Labels:** per timestep, slip once displacement passes the threshold.
- **Input features, Model A (22 channels):** touch, normal force, tangential force and torque magnitude for each of 5 fingertips, plus pull force and phase. `[FLAG Nov 4: the channels named normal and tangential don't measure that. Force axis 2 reads a median 18% of the true contact force, because the contact normal mostly lies along other sensor axes. Describe them as force components along fixed sensor axes. Results stand, but this wording doesn't.]`
- **Excluded as leakage [41]:** object position, displacement and slip speed. Displacement defines the label, and object position wouldn't come from fingertip sensing.
- **Note on pull force and phase:** these are known commands, not measurements. Keep them in the description, because `results.md` notes that pull force carries visible onset signal while tactile channels showed no visible pre-slip pattern (Oct 15 entry). Mention this honestly.
- **Architecture:** causal TCN [22, 24], hidden size 32, kernel 3, 7 layers, dilations 1 to 64, receptive field 255 steps, with residual connections and a 2-class output per timestep.
- **Model B (25 channels):** Model A plus three per-trial values broadcast across time: combined friction, object mass, and grip force at settle. The code calls this PI-TCN, but it has no physics loss. Call it something like "physics-feature TCN" in the paper. `[FLAG Nov 4: "grip force at settle" comes from that same axis, so it isn't grip force.]`
  - Say that nine candidate physics features and losses were tested in October and did not separate slip onset from stable grasping (`results.md`, "Model B"). `[FLAG Nov 4: those tests may be confounded by the mislabeled axis. Don't present this as evidence that physics features can't work.]`
- **Training:** Adam [27] at learning rate 1e-3, batch 32, class-weighted cross-entropy with label smoothing 0.1 [28]. Model A used 20 epochs, Model B 40. Runs were unseeded.
- **Metrics:** per-timestep F1 and accuracy on the slip class, excluding padding. Because classes are imbalanced (22.6% slip timesteps), also give precision and recall [29].
- **Threshold:** Model B's 0.30 was chosen on validation data only, then applied once to test. Model A uses argmax.

### 5.6 Hand-tuned controller
- Runs during the pull phase, every 5 steps, once 300 steps of history exist.
- If slip probability is at or below 0.3: no change.
- Otherwise each actuator target moves toward its limit: `step = 0.35 x p x (limit - target)`. Corrections only tighten.
- Grip formation is scripted and identical for every object.
- Tuned using earlier 45-trial experiments. Variants that did not help: lower trigger thresholds, smoothing, per-finger correction (`results.md`). Report these as exploratory.

### 5.7 RL controller
- PPO [30] in Stable-Baselines3 [31], with the custom Gymnasium [32] environment `SlipGraspEnv`.
- **Control scope:** the policy controls the whole episode, including grip formation, starting from an open hand.
- **Action:** 5 values, one per finger, each changing that finger's closure by up to 0.08 of its range per step. Each action repeats for 5 physics steps.
- **Observation (115 values):** the last 5 readings of tactile channels, pull force, phase and TCN slip probability. No object position. The TCN is frozen (file `ce813826`, F1 0.691).
- **TCN warm-up:** probability is 0 for the first 300 physics steps, so before the pull begins at step 800 the policy has both zero and live values.
- **Reward:**
  - Formation: contact 1.0, grip force 0.15 (capped at 2x weight), object speed penalty 0.1, drop penalty 15.
  - Pull: displacement penalty 50, slip penalty 20, hold bonus 20.
  - Effort penalty 0.02 throughout.
- **Termination in training:** falling past 5 cm during formation, slipping, or reaching the time limit.
- **Hyperparameters:** rollout 256 per environment, batch 256, learning rate 3e-4, 4 environments, 300,000 steps. Other values are library defaults: gamma 0.99, GAE lambda 0.95, clip 0.2, 10 epochs. No observation or reward normalization.
- **Training objects:** 11. Held-out objects: cone, cylindrical prism, plus cross, sphere.
- **Repeats:** three independent runs of one configuration, not explicitly seeded. They also needed different numbers of process launches because of a MuJoCo memory failure (8, 2 and 1), so they aren't perfectly identical procedures. Say so.

### 5.8 Evaluation
- **Locked set:** 15 objects x seeds 0-66, 1,005 trials, paired across conditions. Seeds 3-66 (960 trials) are the primary figure for RL comparisons, because seeds 0-2 were already used during development.
- **Dev set:** 15 objects x seeds 1000-1009, 150 trials. Used for choices between configurations.
- **Original benchmark:** seeds 0-2, 45 trials, reported for continuity.
- **Pre-registered rules:** written in `results.md` before those runs [42]. A new configuration counts as "improved" only if every new repeat beats every baseline repeat and the mean gain is at least the larger of 15 points or 1 baseline SD.
- **Deviation to disclose:** the 10 cm rule was added after seeing results, and locked-set runs ran before dev-set checks. No choices used the locked set.
- **Evaluating RL under other rules:** an evaluation-only option lets trials continue past 5 cm and records the maximum fall. Policies were never trained past 5 cm, so behaviour beyond it is untrained.
- **Statistics:**
  - Wilson 95% intervals for rates [43, 44]
  - Exact McNemar test for paired trial comparisons [45]
  - For static metrics, intervals at trial level, because timesteps within a trial are strongly correlated
  - For RL, report every repeat, with mean and SD across repeats [33, 34]
- **Reproducibility:** each trial is seeded. Result rows record hashes of the checkpoint, policy, code and scene. Re-running the three RL policies reproduced every logged outcome (R7).

---

## 6. Results

### R1. Static detection (Table 2)
Whole test split, 2,628 trials, per-timestep pooled. Source: `Data/_tcn_f1_subsets.log`.

| Checkpoint | Rule | F1 (95% CI) | Precision | Recall | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| `ce813826` Model A, used closed-loop | argmax | 0.691 (0.673-0.707) | 0.904 | 0.559 | 0.983 | 0.887 |
| `dbb50c5a` Model A, older run | argmax | 0.773 (0.759-0.786) | 0.786 | 0.760 | 0.940 | 0.899 |
| `948a2716` Model B | 0.50 | 0.772 (0.757-0.787) | 0.913 | 0.669 | 0.981 | 0.911 |
| `948a2716` Model B | 0.30 | 0.799 (0.785-0.811) | 0.793 | 0.804 | 0.939 | 0.908 |

- At the same decision rule, the older Model A and Model B are effectively tied. Model B's lead depends on its tuned threshold. It also had 40 epochs against 20.
- All four rows reproduced their logged values within ±0.001 before any further analysis.

### R2. Run-to-run variance in the detector
- The same nominal Model A configuration has produced F1 0.773, 0.722 (at 40 epochs) and 0.691 across unseeded runs (0.773 and 0.691 at 20 epochs). Early runs gave 0.715-0.719 before label smoothing.
- So single-run F1 comparisons aren't reliable here. `[PENDING: mean and SD from 5 seeded runs per model, if completed. This is your training code to change]`
- Disclose: `model_a.pt`, used in all closed-loop and RL work, is the F1 0.691 run.

### R3. Effect of drop trials on F1
Same test split, split at the 5 cm rule (1,224 no-drop trials, 1,404 drop trials).

| Checkpoint | Rule | All | No drop | Drop |
|---|---|---|---|---|
| `ce813826` | argmax | 0.691 | 0.789 | 0.559 |
| `dbb50c5a` | argmax | 0.773 | 0.868 | 0.660 |
| `948a2716` | 0.50 | 0.772 | 0.868 | 0.638 |
| `948a2716` | 0.30 | 0.799 | 0.879 | 0.703 |

- F1 is higher without drop trials, so these trials don't appear to be inflating it.
- Caveat: the slip base rate differs between subsets (26.5% vs. 19.3%). No near-floor split has been run.

### R4. Fall sizes before the pull (Figure 1)
Hand-tuned, 1,005 trials. Falls don't depend on the model: flags match between Models A and B on all 1,005 trials.

| Fall | Trials | Not slipped in pull | Reading |
|---|---|---|---|
| 5 cm or less | 424 | 190 | Gripped |
| 5-7 cm | 346 | 126 | Sank, but appears gripped |
| 7-15 cm | 88 | 3 | Mostly lost |
| Over 15 cm | 147 | 129 | Near floor level. Not real holds |

- The 5 cm cutoff sits in the densest part of the distribution (4.5-5.0 cm: 117 trials, 5.0-5.5 cm: 171 trials).
- Evidence on the 5-7 cm group, from a sample of 1,500 training trials at end of settle:

| Fall | n | Median fingers in contact | Median summed force | Slip rate |
|---|---|---|---|---|
| 5 cm or less | 739 | 4 | 16.2 | 75.1% |
| 5-7 cm | 232 | 5 | 49.4 | 75.4% |
| 7-15 cm | 150 | 3 | 0.15 | 98.0% |
| Over 15 cm | 379 | 3 | 0.15 | 7.9% |

- Caveat: training data at mixed squeeze levels, not the closed-loop benchmark.

### R5. Hand-tuned closed-loop results (Table 3, Figure 2)
2,300 steps, 1,005 trials. Source: `Data/handtuned_model{A,B}_2300step_results.jsonl`.

| Model | No drop rule | 10 cm rule | 5 cm rule |
|---|---|---|---|
| A (F1 0.691) | 44.6% (41.5-47.7), 448 | 31.7% (28.9-34.7), 319 | 18.9% (16.6-21.4), 190 |
| B (F1 0.799) | 44.7% (41.6-47.8), 449 | 31.8% (29.0-34.8), 320 | 19.3% (17.0-21.9), 194 |

- Seeds 3-66 (960 trials): A 31.8% and 18.8%, B 32.1% and 19.3% at 10 cm and 5 cm.
- Test length also matters. On the short 1,500-step test with 200 steps at full pull, the same rules give 58.6%, 45.5% and 26.6% for A, and 57.9%, 44.8% and 25.9% for B. These were corrected using falls from the 2,300-step run. Model A's shorter and longer runs are consistent on every trial.
- Sensitivity: Model A at cutoffs of 5, 6, 7, 8 and 10-20 cm scores 190, 302, 316, 317 and 319.

### R6. Static F1 vs. closed-loop performance
- Model A and Model B are tied on paired trials at every rule:
  - No drop rule: 12 held only by A, 13 only by B (exact McNemar p = 1.00)
  - 10 cm: 12 and 13 (p = 1.00)
  - 5 cm: 6 and 10 (p = 0.45)
- Likely mechanism: most holds needed no correction at all. Under the 10 cm rule, 256 of A's 319 holds (80.3%) and 207 of B's 320 holds (64.7%) had zero corrections. `[FLAG Nov 4: a fixed ramp with no TCN and no corrections holds 40 of 150 on the dev set against A's 35, paired 6 vs. 1, p = 0.125 (R15). The exact no-TCN hand-tuned run is queued.]`
- Second mechanism: at squeeze 0.95, only 5% of the actuator range remains. In all 7 trials tested, every actuator was pinned at its maximum by the end of the pull (`results.md`, "Actuator saturation"; a small diagnostic sample). `[FLAG Nov 4: don't use this as the explanation. Fingertip contact forces reach 20-70 N on objects under 1.5 N, and friction capacity is about 26x the load. The actuators are pinned, but a stronger grip isn't what's missing.]`
- **Be careful:** this is two checkpoints with one controller. They also differ in inputs, epochs and operating point. It shows that the F1 gap didn't matter here, not that detection never matters. A controller with more grip authority might respond differently.
- `[PENDING, optional: hand-tuned run with the F1 0.773 file. Same architecture as the 0.691 file, so it's a cleaner test]`

### R7. Reproducibility of RL results
- All three policies, re-evaluated on the original 45 trials, reproduced their logged results: 29, 16 and 8 of 45. Per-trial outcomes matched for the two repeats with per-trial logs, and per-object counts matched for the third.
- So the spread comes from training, not evaluation noise.

### R8. RL vs. hand-tuned on the same 45 trials (Table 4)

| Controller | 5 cm rule | 10 cm rule |
|---|---|---|
| RL repeat 1 (best of 3) | 64.4% (49.8-76.8) | 64.4% (49.8-76.8) |
| RL repeat 2 | 35.6% (23.2-50.2) | 37.8% (25.1-52.4) |
| RL repeat 3 | 17.8% (9.3-31.3) | 17.8% (9.3-31.3) |
| RL mean (SD across repeats) | 39.3 (23.6) | 40.0 (23.4) |
| Hand-tuned A | 22.2% (12.5-36.3) | 31.1% (19.5-45.7) |
| Hand-tuned B | 20.0% (10.9-33.8) | 26.7% (16.0-41.0) |

Paired against hand-tuned A at 10 cm (held only by RL, only by hand-tuned):
- Repeat 1: 15 and 0 (p = 0.0001), but it is the best of 3, so it's favourably selected
- Repeat 2: 8 and 5 (p = 0.58)
- Repeat 3: 4 and 10 (p = 0.18)

Finding on the drop rule: only 1 of 135 RL scores changed when moving from 5 cm to 10 cm. Of 21 RL trials that fell 5-15 cm, 1 held. So the 5 cm rule isn't what limits these policies, although they were never trained past it.

### R9. RL on the locked set (Table 5)
Deterministic. Seeds 3-66 are primary, because seeds 0-2 were already used to compare and choose policies. Files: `Data/rl_seed{1,2,3}_1005trial_results.jsonl`. No errored trials.

| Held, seeds 3-66 (95% CI) | 5 cm | 10 cm |
|---|---|---|
| Repeat 1 (best of 3 seeds, deterministic) | 588/960, 61.2% (58.1-64.3) | 610/960, 63.5% (60.4-66.5) |
| Repeat 2 | 405/960, 42.2% (39.1-45.3) | 423/960, 44.1% (41.0-47.2) |
| Repeat 3 | 144/960, 15.0% (12.9-17.4) | 147/960, 15.3% (13.2-17.7) |
| **Mean and SD across repeats** | **39.5, SD 23.2** | **41.0, SD 24.3** |
| Hand-tuned A | 180/960, 18.8% | 305/960, 31.8% |
| Hand-tuned B | 185/960, 19.3% | 308/960, 32.1% |

- All 1,005 trials, 5 cm / 10 cm: repeat 1 61.4% / 63.6%, repeat 2 41.9% / 43.8%, repeat 3 15.1% / 15.4%. Mean 39.5 / 40.9, SD 23.2 / 24.2.
- Paired against hand-tuned A, seeds 3-66 (held only by RL, only by hand-tuned, exact McNemar):
  - 5 cm: repeat 1 438 and 30 (p = 5e-94), repeat 2 299 and 74 (p = 4e-33), repeat 3 111 and 147 (p = 0.029).
  - 10 cm: repeat 1 353 and 48 (p = 2e-58), repeat 2 250 and 132 (p = 2e-9), repeat 3 95 and 253 (p = 1e-17).
  - Against hand-tuned B the pattern is the same: at 5 cm 433 and 30, 297 and 77, 109 and 150. At 10 cm 350 and 48, 250 and 135, 92 and 253.
- Two policies beat hand-tuned and one is worse than it. That's the honest statement of finding 3.
- Three-repeat mean against hand-tuned A, not pre-registered, df 2: 5 cm difference 20.7 points (p = 0.26), 10 cm difference 9.2 points (p = 0.58). Three repeats with this spread can't establish that the training procedure beats hand-tuned on average.
- The 45-trial numbers held up roughly at 5 cm: 64.4% became 61.2%, 35.6% became 42.2%, and 17.8% became 15.0%.
- No repeat reaches 70%, and the top of repeat 1's interval is 64.3% at 5 cm and 66.5% at 10 cm.
- Outcomes at 5 cm, all 1,005 (held, fell past 5 cm, slipped without falling): repeat 1 617, 200, 188. Repeat 2 421, 291, 293. Repeat 3 152, 486, 367. Worse repeats fail in both ways, especially falling.
- Training vs. held-out objects at 5 cm, seeds 3-66: repeat 1 57.1% vs. 72.7%, repeat 2 38.8% vs. 51.6%, repeat 3 15.5% vs. 13.7%. Hand-tuned A scores 15.6% vs. 27.3%, even though it has no training objects. So this likely reflects which shapes are held out, not generalization. Don't claim generalization from it.

### R10. Failure analysis by object (Table 6, Figure 3)
Hand-tuned Model A, 67 trials per object, under the 10 cm rule.

| Object | Kind | Fell over 10 cm | Slipped in pull | Held |
|---|---|---|---|---|
| dumbbell | compound | 0 | 0 | 67 |
| sphere | primitive | 0 | 19 | 48 |
| rod | primitive | 0 | 24 | 43 |
| capsule | primitive | 0 | 41 | 26 |
| square pyramid | mesh | 5 | 37 | 25 |
| ellipsoid | primitive | 0 | 44 | 23 |
| L-block | compound | 27 | 21 | 19 |
| triangular prism | mesh | 2 | 46 | 19 |
| cone | mesh | 4 | 48 | 15 |
| hexagonal prism | mesh | 0 | 53 | 14 |
| plus cross | compound | 19 | 38 | 10 |
| rectangular prism | primitive | 40 | 21 | 6 |
| flat disk | primitive | 57 | 7 | 3 |
| cube | primitive | 13 | 53 | 1 |
| cylindrical prism | primitive | 2 | 65 | 0 |

- Totals under the 10 cm rule: 319 held, 169 fell, 517 slipped.
- Failure types differ by shape. Flat disk and rectangular prism mostly fall. Cylinder, cube and hexagonal prism mostly slip.
- Slips are mostly real losses. 267 of 517 went more than 10x past the limit (median slip 14.1 cm), and 168 stayed within 2x.
- Sensitivity to the slip limit: 2x gives 48.5%. Report this as sensitivity, not as a result.
- Dumbbell note: 67/67 held and 0 slipped is a striking outlier, with 35 trials falling past 5 cm. Consider whether its shape lets it rest on the fingers. `[CHECK before claiming anything]` `[FLAG Nov 4: on the dev set, hand-tuned A holds the dumbbell 6 of 10 at 5 cm under default settings and 1 of 10 under noslip, so this outlier may depend on contact settings.]`
- Kind summary at 10 cm: primitives 150/536, meshes 73/268, compounds 96/201.
- RL per object at 5 cm, all seeds, held of 67 (repeat 1, 2, 3, then hand-tuned A at 5 cm): square pyramid 66, 65, 21, 25. Rod 63, 35, 0, 4. Cone 61, 66, 25, 15. Dumbbell 61, 17, 7, 32. Sphere 56, 43, 0, 48. Plus cross 53, 24, 8, 10. Ellipsoid 51, 37, 16, 23. Triangular prism 51, 15, 11, 0. Hexagonal prism 39, 33, 6, 0. Capsule 38, 31, 27, 21. Cylinder 26, 3, 2, 0. L-block 22, 21, 18, 3. Flat disk 14, 16, 1, 3. Rectangular prism 9, 6, 3, 5. Cube 7, 9, 7, 1.
- The best RL repeat reaches 70% on 8 of 15 objects. Cube and rectangular prism stay at 9 of 67 or fewer under every controller. Whether that's a physical limit of this grip isn't measured (see 7.4). `[FLAG Nov 4: under stricter contact settings, seed 1 goes from cube 0 to 5-7 of 10, rectangular prism 1 to 5, and flat disk 0 to 10 (R17). Much of this looks like contact settings rather than a physical limit.]`

### R11. What the RL policy learned
- The best policy never closes the middle finger: closure was exactly 0.00 at every sampled step on all 8 objects traced, including ones it holds (`results.md`, Oct 31). `[FLAG Nov 4: the median trial keeps it at 0, but on 4 of 15 traced trials it closes partway, up to 0.75, so "never" is too strong. All 15 objects and the fixed-grip test are in R16.]`
- It appears to have found one 4-finger strategy that works on curved and pointed objects but not flat or boxy ones. That cause is inferred, not proven.
- Failure types across repeats, on 45 trials (slipped, fell past 5 cm): repeat 1: 5 and 11. Repeat 2: 15 and 14. Repeat 3: 15 and 22.
- Correct the older log claim: hand-tuned control also passes 5 cm on 23 of these 45 trials, so RL didn't create this failure type.

### R12. Exploratory RL configurations
Keep brief, and label as exploratory: 45 trials and single runs.

| Change | Result | Note |
|---|---|---|
| 150,000 more training steps on the best policy | 57.8% | More training did not help |
| Reward term forcing all fingers to close | 44.4% | Fixed the finger, overall worse |
| Object size added to observations | 24.4% | Worse |
| VecNormalize | 2.2% | Not an evaluation bug. Direct and training-style paths agree on 45/45 trials, and a stochastic run matches the training values (reward +1.0%, length -5.3%) |

Reading: given a 17.8-64.4% spread for the unchanged configuration, none of these single runs can be attributed to its change.

### R13. Deterministic vs. stochastic evaluation
Dev set, 150 paired trials, same start conditions, noise seeded per trial. The pre-registered rule trips on all three policies at both cutoffs. Files: `Data/rl_seed{1,2,3}_dev_{deterministic,stochastic}.jsonl`.

| 5 cm | Deterministic | Stochastic | Only deterministic / only stochastic | Difference (limit), points | p |
|---|---|---|---|---|---|
| Repeat 1 | 86, 57.3% | 73, 48.7% | 19 / 6 | 8.7 (6.7) | 0.015 |
| Repeat 2 | 56, 37.3% | 35, 23.3% | 24 / 3 | 14.0 (6.9) | <0.001 |
| Repeat 3 | 19, 12.7% | 8, 5.3% | 14 / 3 | 7.3 (5.5) | 0.013 |

- At 10 cm: 91 vs. 75 (10.7, limit 6.8), 60 vs. 39 (14.0, limit 7.4), 19 vs. 8 (7.3, limit 5.5).
- Every RL rate in this paper measures the policy's mean action. Sampling from the policy as trained holds 7-14 points fewer. State this next to the RL results.
- `[YOUR CALL: which mode is primary. My recommendation is deterministic, with stochastic reported beside it]`

### R14. Longer training rollouts: worse
- 3 new repeats with rollout 512, all else unchanged, each 301,056 steps with no crash.
- Dev set, deterministic: 10.0%, 0.0% and 2.7% at 5 cm (mean 4.2, SD 5.2), and the same at 10 cm. Baseline: 57.3%, 37.3%, 12.7% at 5 cm (mean 35.8, SD 22.4), and 60.7%, 40.0%, 12.7% at 10 cm.
- Pre-registered verdict at both cutoffs: **worse**. Every new repeat is below every baseline repeat, and the mean drop (31.6 and 33.6 points) passes the bar (22.4 and 24.1).
- SD falls only because all three are near the floor. So the hypothesis that rollouts shorter than a trial cause the spread isn't supported.
- Why it got worse isn't established. Both take about the same gradient steps, but this collects half as many rollouts. That's a hypothesis.
- Mean evaluation reward doesn't rank these policies by holding: 434 with 4 held against 415 with 19 held.
- Deterministic only, and the evaluation mode is still your call (R13).

### R15. The TCN signal in closed loop `[NEW Nov 4, dev set]`
- With the slip probability replaced by 0 at test time, the three policies hold 86, 56 and 19 of 150, the same as with it. 0 of 450 trials change at 5 cm or 10 cm.
- Trajectories barely differ: seed 1 has the same trial length on 145 of 150.
- Hand-tuned: a fixed ramp with no TCN and no corrections holds 40 of 150, against A's 35 (paired 6 vs. 1, p = 0.125). 10 cm: 62 vs. 58 (p = 0.344). `[PENDING: the exact no-TCN hand-tuned run]`
- **Be careful:** this tests the signal's use at run time. No policy was trained without it, so it doesn't show training with it was useless.

### R16. A fixed grip matches the best RL policy `[NEW Nov 4, dev set]`
- Fixed grips: no policy, tactile input or TCN. Same environment and scoring. Held of 150, default contact settings:

| Grip or controller | 5 cm | 10 cm | Fell | Slipped |
|---|---|---|---|---|
| Thumb, index, ring, pinky, full speed | 86 (57.3%) | 60.7% | 34 | 30 |
| Same, slower ring and pinky (speeds read from seed 1) | 92 (61.3%) | 64.0% | 29 | 29 |
| Thumb, index, ring, full speed | 76 (50.7%) | 52.7% | 40 | 34 |
| All five fingers, full speed | 34 (22.7%) | 39.3% | 98 | 18 |
| Slow 5-finger close, like the hand-tuned ramp | 40 (26.7%) | 41.3% | 85 | 25 |
| RL seed 1 | 86 (57.3%) | 60.7% | 33 | 31 |

- Paired with seed 1 at 5 cm: 9 vs. 9 (p = 1.00) with the first grip, 7 vs. 1 (p = 0.07) with the second, 8 vs. 60 with all five fingers.
- Seed 1's median closure by RL steps 10, 20 and 50: thumb and index 0.80, 1.00, 1.00. Ring 0.62, 1.00, 1.00. Pinky 0.20, 0.62, 1.00. Middle 0. Raw mean actions sit near or beyond the +1 limit, and 4 of 15 traced trials change any finger by more than 0.01 during the pull.
- **Be careful:** choosing among grips on 150 trials inflates the best one, and seed 1's traces informed these grips. `[PENDING: fixed grips under stricter contact settings; do seeds 2 and 3 differ mainly in which fingers they close?]`

### R17. Contact solver settings `[NEW Nov 4, dev set, exploratory, NOT adopted, YOUR CALL]`
- The scene sets no contact options, so it runs MuJoCo's defaults: pyramidal cone, `impratio` 1, no noslip pass [20].
- At full load under default settings (seed 1, 15 trials): objects move at a median 1.18 mm/s, up to 5.9, while contacts use 3% of their friction limit and capacity is 26x the load. Coulomb friction allows no sliding there. MuJoCo documents this as expected behavior of its soft-contact model and calls slippage "especially problematic in manipulation tasks" [48].
- Other MuJoCo hand models set these options [49]: Allegro, Shadow Hand and Robotiq 2F-85 use `cone="elliptic" impratio="10"`, LEAP uses `impratio="100"`, and Shadow DEXEE uses `cone="elliptic" noslip_iterations="5"`.
- Same controllers, none retrained. Held of 150 at 5 cm (10 cm):

| Controller | Default | Noslip pass | Elliptic, `impratio` 10 |
|---|---|---|---|
| RL seed 1 | 57.3% (60.7%) | 81.3% (86.0%) | 78.7% (82.7%) |
| RL seed 2 | 37.3% (40.0%) | 67.3% (72.0%) | `[PENDING]` |
| RL seed 3 | 12.7% (12.7%) | 40.7% (44.0%) | `[PENDING]` |
| Hand-tuned A | 23.3% (38.7%) | 44.0% (67.3%) | `[PENDING]` |

- Paired against default at 5 cm (held only with / only without): seed 1 42 / 6, seed 2 56 / 11, seed 3 49 / 7, hand-tuned A 41 / 10, all p below 0.0001. Noslip against elliptic, seed 1: 15 / 11, p = 0.56.
- Three-seed mean under noslip: 63.1%, SD 20.6, at 5 cm. So "one policy above 70%" and "consistent 70%" are different claims.
- The grip doesn't change: seed 1's median closure matches at every sampled step. Among its held trials, grips carried mostly by friction go from 16 to 52.
- **What argues against over-reading this:** the settings were chosen after seeing scores, on a much-used dev set. The policies and TCN were trained on default physics. Pad friction doesn't predict holding under any setting (seed 1 under noslip, 84.0% at low friction and 78.7% at high). Contacts at a slip reach a median 0.76 of their limit, so not every slip is sliding at the limit. And MuJoCo calls noslip "just an adhoc mechanism".
- `[PENDING: 1,005 fresh trials per controller under each setting, pre-registered in results.md. The locked set stays unrun under any new setting until you decide.]`

---

## 7. Discussion

**7.1 Success definitions can change the story**
- Identical trials ranged from 18.9% to 44.6%. Before this was measured, near-floor objects were counted as holds.
- Recommendation: benchmarks should confirm the object is still in the hand at the start of the measured phase, and should report sensitivity to cutoffs. This fits calls for explicit success criteria [39].
- The same flaw reached the training labels. In a sample of 1,500 contact-only trials, 349 of 593 no-slip examples (59%) fell more than 15 cm. A label check is worth doing before training.

**7.2 Why better detection didn't help here**
- Most holds needed no correction, and actuators were often at their limit, so the detector had little leverage. The initial grip largely decides the outcome. `[FLAG Nov 4: drop the actuator part (see R6). R15 is stronger evidence for the first part.]`
- This matches the pattern in driving research that offline metrics aren't sufficient [37, 38], but the setting and sample here are limited.
- Practical point: measure a detector in closed loop, and make sure the controller has authority to act on it.

**7.3 RL: a good policy is possible, but getting one isn't reliable** `[FLAG Nov 4: revise this section. Longer rollouts came back worse (R14), a fixed grip matches the best policy (R16), and the slip signal changes no outcome (R15). "A good policy" may mean a policy that found a good fixed grip.]`
- On 960 locked trials at 5 cm, the same setup gave 61.2%, 42.2% and 15.0%. Two policies clearly beat hand-tuned, and one is worse than it.
- So "RL is better" is true of some policies, not established for the training procedure. Report all three, not the best (R9).
- RL acts during formation, which hand-tuned control can't do. This may explain the advantage where there is one. That cause is untested. `[FLAG Nov 4: now partly tested. A fixed grip that closes fast without the middle finger gives the same result on the dev set (R16).]`
- All of these rates use the mean action, and sampled actions do 7-14 points worse (R13).
- The spread between repeats is the main obstacle. This agrees with the wider literature on RL variance [33, 34, 35, 36]. Don't call the size of this spread normal: Henderson et al. may describe PPO as relatively stable, so check [33].
- Reward and observation changes made single runs worse, but with this spread they can't be attributed to the changes.

**7.4 Where the limits probably are**
- Some shapes rarely hold under either controller. Every object uses the same hand pose and closing motion.
- Whether these shapes can be held at all in this setup hasn't been measured. A best-case controller test would answer that (see section 9). `[FLAG Nov 4: under default settings, any of 5 controllers holds 74.1% of the locked set, and the best grip or policy per object totals 67.3% on dev. Under stricter contact settings, the hard shapes improve most (R17).]`
- For a 70% overall rate, if four shapes stayed near zero, the other 11 would need about 95%.

**7.5 Lessons for simulated grasp studies**
- Test length, success rule, checkpoint identity and training repeats each changed results here.
- Record hashes and configuration with every result. That is how the checkpoint mismatch was caught.

---

## 8. Limitations (state all of these)

- Simulation only, with no hardware validation. Contacts are soft in MuJoCo [19, 20], so slow creep under load is possible and could inflate slip counts near the limit. `[FLAG Nov 4: no longer "possible". It is measured, and it is large (R17). Rewrite this, and say which settings the final numbers use.]`
- The pull is downward only, at 1.5x weight. Real disturbances vary.
- Same hand pose and object placement for all objects, with no grasp planning or arm.
- Detector data uses a random split by trial, so static scores don't test generalization to new objects.
- Training labels include near-floor no-slip examples. They were not relabelled.
- The TCN trained on data recorded with a sensor noise setting of 0.05, but a search finds no noise step in `closed_loop_sim.py` or `rl_env.py`, so training and closed-loop conditions differ. Ramp length also varied in training (400-600) and is fixed at 500 in closed loop.
- Only three RL repeats, not explicitly seeded, with different numbers of process restarts.
- RL rates are deterministic, using the policy's mean action, which isn't the sampled behaviour seen during training. Sampled actions held 7-14 points fewer on the dev set (R13).
- Static F1 values for both models are single, unseeded runs unless the seeded runs get done.
- The A vs. B comparison has only two checkpoints that differ in several ways.
- The 10 cm rule was chosen after seeing results.
- RL policies were trained with a 5 cm rule, so 10 cm scoring doesn't show what a policy trained under 10 cm could do.
- The 5-7 cm "gripped" evidence comes from training data, not the benchmark.
- No RL + Model B result exists. There is also a known timing issue in `rl_env.py` for 25-channel models.

---

## 9. Future work (Phase 2)

1. **Measure the ceiling:** a best-case controller shows what share of trials can physically be held. This tells us whether mid-to-high 90s is possible.
2. **Fix labels:** exclude or relabel near-floor examples, verify the object is in hand, and retrain.
3. **Seeded multi-run training:** mean and SD for both models.
4. **True physics-informed TCN:** add a physics loss [25, 26], such as a friction-cone constraint [2], and test it in closed loop. `[FLAG Nov 4: a friction-cone loss assumes slip happens at the friction limit. Under default settings, slips happen at a median 0.56 of the limit and 10% of capacity, and the normal and tangential channels are mislabeled. Fix the contact settings and channels first, and regenerate the data, or this loss would fight its own labels.]`
5. **Better grip for failing shapes:** hand pose, object placement, adaptive formation. Treat falling and slipping as separate problems.
6. **More stable RL:** more repeats, tuned rollouts, residual control on top of the hand-tuned controller [46, 47].
7. **RL with each detector:** test whether RL uses the slip signal at all, by removing it, and compare detectors under RL. `[FLAG Nov 4: removal is done at test time (R15). Training without the signal is still untested.]`
8. **Generalization:** leave-object-out splits, which `export_data.py` already supports.
9. **Toward hardware:** domain randomization [16, 21], then the physical ORCA hand [17].

---

## 10. Conclusion

1. Restate the question.
2. Success definition alone moved identical trials between 18.9% and 44.6%.
3. A large static F1 gap produced no closed-loop difference here, because the initial grip and limited actuator headroom left little for detection to change. `[FLAG Nov 4: drop "limited actuator headroom" (R6).]`
4. Three PPO training repeats held 61.2%, 42.2% and 15.0% at 5 cm (mean 39.5%, SD 23.2) against 18.8% for hand-tuned control, so a learned policy can be far better, but training didn't reliably produce one. `[FLAG Nov 4: rewrite, see finding 3. Add the contact-settings result if you adopt it or report it (R17).]`
5. Take-away: verify what success counts, evaluate detectors in closed loop, and report all training repeats.
6. Next step: measure the physical ceiling and build the true physics-informed model.

---

## 11. Figures and tables

| Item | Content | Status |
|---|---|---|
| Figure 1 | Histogram of formation falls, marking 5 cm and 10 cm | To make, data ready |
| Figure 2 | Score under each rule, A and B, with intervals | To make. `reports/figures/fig1_closed_loop_success.png` is related, but it isn't this chart |
| Figure 3 | Per-object heat table with failure type | Regenerated with final data: `reports/figures/fig3_per_object_success.png`. Numbers may change if the contact settings do |
| Figure 4 | RL per repeat with mean and SD, hand-tuned reference | Regenerated with final data: `reports/figures/fig2_rl_seed_spread.png`. Numbers may change if the contact settings do |
| New, optional | R17: each controller under each contact setting | To make once fresh-seed results are in |
| Figure 5 | Static F1 with spread across seeds | `[PENDING: your seeded runs]` |
| Optional | Diagram of the pipeline and trial phases | To make |
| Table 1 | Objects: kind, size, mass, slip limit | To make from `generate_trials.py` |
| Table 2 | Static metrics (R1) | Ready |
| Table 3 | Hand-tuned under each rule (R5) | Ready |
| Table 4 | Same 45 trials (R8) | Ready |
| Table 5 | RL locked set (R9) | Ready |
| Table 6 | Failures by object (R10) | Ready for hand-tuned; add RL |

---

## 12. References

Numbered in the order used for planning. Renumber to match your final paper and style.

**Grasping and touch**

[1] A. Bicchi and V. Kumar, "Robotic grasping and contact: a review," in *Proc. IEEE ICRA*, 2000, vol. 1, pp. 348-353. doi:10.1109/ROBOT.2000.844081. **[M]**

[2] D. Prattichizzo and J. C. Trinkle, "Grasping," in *Springer Handbook of Robotics*, Springer, 2016, pp. 955-988. doi:10.1007/978-3-319-32552-1_38. **[M]**

[3] J. Bohg, A. Morales, T. Asfour, and D. Kragic, "Data-Driven Grasp Synthesis: A Survey," *IEEE Trans. Robotics*, vol. 30, no. 2, pp. 289-309, 2014. doi:10.1109/TRO.2013.2289018. **[A]**

[4] R. S. Johansson and J. R. Flanagan, "Coding and use of tactile signals from the fingertips in object manipulation tasks," *Nature Reviews Neuroscience*, vol. 10, no. 5, pp. 345-359, 2009. doi:10.1038/nrn2621. **[M]**

[5] G. Westling and R. S. Johansson, "Factors influencing the force control during precision grip," *Experimental Brain Research*, vol. 53, no. 2, 1984. doi:10.1007/BF00238156. **[M]**

**Slip detection and tactile control**

[6] R. A. Romeo and L. Zollo, "Methods and Sensors for Slip Detection in Robotics: A Survey," *IEEE Access*, vol. 8, pp. 73027-73050, 2020. doi:10.1109/ACCESS.2020.2987849. **[A]**

[7] J. M. Romano, K. Hsiao, G. Niemeyer, S. Chitta, and K. J. Kuchenbecker, "Human-Inspired Robotic Grasp Control With Tactile Sensing," *IEEE Trans. Robotics*, vol. 27, no. 6, pp. 1067-1079, 2011. doi:10.1109/TRO.2011.2162271. **[A]**

[8] F. Veiga, H. van Hoof, J. Peters, and T. Hermans, "Stabilizing novel objects by learning to predict tactile slip," in *Proc. IEEE/RSJ IROS*, 2015, pp. 5065-5072. doi:10.1109/IROS.2015.7354090. **[A]**

[9] Z. Su, K. Hausman, Y. Chebotar, A. Molchanov, G. E. Loeb, G. S. Sukhatme, and S. Schaal, "Force estimation and slip detection/classification for grip control using a biomimetic tactile sensor," in *Proc. IEEE-RAS Humanoids*, 2015, pp. 297-303. doi:10.1109/HUMANOIDS.2015.7363558. **[M]**

[10] J. W. James, N. Pestell, and N. F. Lepora, "Slip Detection With a Biomimetic Tactile Sensor," *IEEE Robotics and Automation Letters*, vol. 3, no. 4, pp. 3340-3346, 2018. doi:10.1109/LRA.2018.2852797. **[M]**

[11] J. Li, S. Dong, and E. Adelson, "Slip Detection with Combined Tactile and Visual Information," in *Proc. IEEE ICRA*, 2018, pp. 7772-7777. doi:10.1109/ICRA.2018.8460495. arXiv:1802.10153. **[A]**

[12] S. Dong, D. Ma, E. Donlon, and A. Rodriguez, "Maintaining Grasps within Slipping Bounds by Monitoring Incipient Slip," in *Proc. ICRA*, 2019, pp. 3818-3824. doi:10.1109/ICRA.2019.8793538. arXiv:1810.13381. **[A]**

[13] R. Calandra, A. Owens, M. Upadhyaya, W. Yuan, J. Lin, E. H. Adelson, and S. Levine, "The Feeling of Success: Does Touch Sensing Help Predict Grasp Outcomes?" in *Proc. CoRL*, 2017. arXiv:1710.05512. **[A]**

[14] Y. Chebotar, K. Hausman, Z. Su, G. S. Sukhatme, and S. Schaal, "Self-supervised regrasping using spatio-temporal tactile features and reinforcement learning," in *Proc. IEEE/RSJ IROS*, 2016, pp. 1960-1966. doi:10.1109/IROS.2016.7759309. **[A]**

**Dexterous hands and simulation**

[15] A. Rajeswaran, V. Kumar, A. Gupta, G. Vezzani, J. Schulman, E. Todorov, and S. Levine, "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations," in *Proc. RSS*, 2018. doi:10.15607/RSS.2018.XIV.049. **[A]**

[16] M. Andrychowicz et al. (OpenAI), "Learning dexterous in-hand manipulation," *Int. J. Robotics Research*, vol. 39, no. 1, pp. 3-20, 2020. doi:10.1177/0278364919887447. arXiv:1808.00177. **[A]**

[17] C. C. Christoph et al., "ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning," arXiv:2504.04259, 2025. The arXiv entry says accepted to IROS 2025. Cite the proceedings version if you can find it. **[A]**

[18] E. Todorov, T. Erez, and Y. Tassa, "MuJoCo: A physics engine for model-based control," in *Proc. IEEE/RSJ IROS*, 2012, pp. 5026-5033. doi:10.1109/IROS.2012.6386109. **[M]**

[19] E. Todorov, "Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo," in *Proc. IEEE ICRA*, 2014, pp. 6054-6061. doi:10.1109/ICRA.2014.6907751. **[M]**

[20] MuJoCo documentation, "Computation" chapter, on soft contact and constraints. https://mujoco.readthedocs.io/en/latest/computation/. Cite the version you used (3.11.0). **Read in search results, not in full.**

[21] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," in *Proc. IEEE/RSJ IROS*, 2017, pp. 23-30. doi:10.1109/IROS.2017.8202133. **[M]**

**Temporal models and physics-informed learning**

[22] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018. **[A]**

[23] C. Lea, M. D. Flynn, R. Vidal, A. Reiter, and G. D. Hager, "Temporal Convolutional Networks for Action Segmentation and Detection," in *Proc. IEEE CVPR*, 2017, pp. 1003-1012. doi:10.1109/CVPR.2017.113. **[M]**

[24] A. van den Oord et al., "WaveNet: A Generative Model for Raw Audio," arXiv:1609.03499, 2016. **[A]**

[25] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *J. Computational Physics*, vol. 378, pp. 686-707, 2019. doi:10.1016/j.jcp.2018.10.045. **[A]**

[26] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, "Physics-informed machine learning," *Nature Reviews Physics*, vol. 3, no. 6, pp. 422-440, 2021. doi:10.1038/s42254-021-00314-5. **[A]**

**Training and metrics**

[27] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. ICLR*, 2015. arXiv:1412.6980. **[A]**

[28] R. Müller, S. Kornblith, and G. Hinton, "When Does Label Smoothing Help?" in *Proc. NeurIPS*, 2019. arXiv:1906.02629. **[A]**

[29] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS ONE*, vol. 10, no. 3, e0118432, 2015. doi:10.1371/journal.pone.0118432. **[M]**

**Reinforcement learning**

[30] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017. **[A]**

[31] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *J. Machine Learning Research*, vol. 22, no. 268, pp. 1-8, 2021. https://jmlr.org/papers/v22/20-1364.html. **[M, checked on the journal page]**

[32] M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," arXiv:2407.17032, 2024. The arXiv entry says accepted at NeurIPS Datasets and Benchmarks 2025. **[A]**

[33] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, "Deep Reinforcement Learning That Matters," in *Proc. AAAI*, vol. 32, no. 1, 2018. doi:10.1609/aaai.v32i1.11694. **[M, partial abstract]**

[34] R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," in *Proc. NeurIPS*, 2021. arXiv:2108.13264. **[A]**

[35] M. Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study," arXiv:2006.05990, 2020. Listed at ICLR 2021 as "What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study." **[A]**

[36] L. Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," in *Proc. ICLR*, 2020. arXiv:2005.12729. **[A]**

[46] T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, and S. Levine, "Residual Reinforcement Learning for Robot Control," in *Proc. ICRA*, 2019, pp. 6023-6029. doi:10.1109/ICRA.2019.8794127. **[A]**

[47] T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, "Residual Policy Learning," arXiv:1812.06298, 2018. **[A]**

**Offline metrics, benchmarks and rigor**

[37] F. Codevilla, A. M. López, V. Koltun, and A. Dosovitskiy, "On Offline Evaluation of Vision-based Driving Models," in *Proc. ECCV*, 2018. arXiv:1809.04843. **[A]**

[38] J. Philion, A. Kar, and S. Fidler, "Learning to Evaluate Perception Models Using Planner-Centric Metrics," in *Proc. IEEE/CVF CVPR*, 2020, pp. 14052-14061. doi:10.1109/CVPR42600.2020.01407. **[A]**

[39] Y. Bekiroglu et al., "Benchmarking Protocol for Grasp Planning Algorithms," *IEEE Robotics and Automation Letters*, vol. 5, no. 2, pp. 315-322, 2020. doi:10.1109/LRA.2019.2956411. **[A]**

[40] B. Calli et al., "Yale-CMU-Berkeley dataset for robotic manipulation research," *Int. J. Robotics Research*, vol. 36, no. 3, pp. 261-268, 2017. doi:10.1177/0278364917700714. **[M]**

[41] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, 100804, 2023. doi:10.1016/j.patter.2023.100804. **[M]**

[42] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, "The preregistration revolution," *PNAS*, vol. 115, no. 11, pp. 2600-2606, 2018. doi:10.1073/pnas.1708274114. **[A]**

**Statistics**

[43] E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," *J. American Statistical Association*, vol. 22, no. 158, pp. 209-212, 1927. doi:10.1080/01621459.1927.10502953. **[M]**

[44] L. D. Brown, T. T. Cai, and A. DasGupta, "Interval Estimation for a Binomial Proportion," *Statistical Science*, vol. 16, no. 2, pp. 101-133, 2001. doi:10.1214/ss/1009213286. **[M]**

[45] Q. McNemar, "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages," *Psychometrika*, vol. 12, no. 2, pp. 153-157, 1947. doi:10.1007/BF02295996. **[M]**

**Added Nov 4: read these before relying on them**

[48] MuJoCo documentation, "Modeling" chapter, "Solver settings" and "Preventing slip". https://mujoco.readthedocs.io/en/stable/modeling.html. Read in full on Nov 4, 2026; quotes in `results.md`. Cite the version you used (3.11.0).

[49] MuJoCo Menagerie, model files `wonik_allegro/right_hand.xml`, `shadow_hand/right_hand.xml`, `robotiq_2f85/2f85.xml`, `leap_hand/right_hand.xml` and `shadow_dexee/shadow_dexee.xml`. https://github.com/google-deepmind/mujoco_menagerie. `<option>` lines read on Nov 4, 2026. Cite the repository's own recommended paper and the commit you read.
