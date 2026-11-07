# Paper outline: slip detection and closed-loop grasp control in simulation

**Status:** working draft, updated Nov 7, 2026, evening. Every pre-registered fresh-seed run is in, elliptic and noslip (R22). The reversal check passed, so elliptic stays primary, and every figure includes them. The Nov 4 flags are resolved: each flagged point is rewritten to match the evidence, and every result points to its figure in `reports/figures/paper/`. That folder's `README.md` lists the figures with draft captions, and `stats_summary.md` there has every number and test behind them. **Still your call:** the headline number, the evaluation mode, the venue, and anything marked `[YOUR CALL]`. **Still `[PENDING]`:** only the seeded F1 items, which wait on your runs. **New Nov 7 evening:** every point says exactly what to cite (see "How to read this file"). The references marked "confirm" were checked, and [53] had been cited for something its abstract doesn't say, so that's fixed. **Corrected Nov 7:** the contact settings don't move results more than the choice of controller or seed does (finding 4).

**What changed since Nov 4**
- The benchmark is the elliptic cone with `impratio` 10, with default settings reported beside it. You delegated the choice on Nov 4, and the reasons and the rule for reversing it are in `results.md`.
- No retrain under elliptic for the Nov 8 paper, after the Nov 4 crash. The fresh-seed elliptic runs give the final elliptic numbers, and the locked set gives the default-physics numbers (`results.md`, Nov 6 amendment).
- New analyses: the TCN removed from every controller, what each RL seed's fingers do, where the contact-setting gain comes from, how the reward treats held and dropped objects, training curves, pad friction, slip timing, and agreement between controllers.
- Framing (Nov 6): the paper is built around Figure 0, which ranks every factor by how much it moves the held rate. The detector comes last. See "What's new here, and what isn't" for how to state the novelty.
- Nov 7: every fresh-seed run finished, including noslip for the three RL seeds (R22). Noslip is the sensitivity setting and changes nothing in the headline.

**How to read this file**
- Every point is something to say in that section, in the order it should appear. Rewrite it in your own words.
- `[PENDING: ...]` means the run hasn't finished. Don't estimate these numbers.
- `[YOUR CALL: ...]` means a decision that is yours.
- `[n]` points to the numbered reference list at the end.
- **Cite:** at the end of a point says exactly what to cite there: reference numbers from section 12 with first author and year, or the figure or table for your own results. In section 6, each result section has one **Cite:** line under its heading, and a point gets its own only when it needs something more. Methods carry the references for tests and software, so results don't repeat them. A point with no **Cite:** is an instruction to you or your own argument, and needs no citation.
- Numbers come from `src/training/results.md`, from `reports/figures/paper/stats_summary.md`, or from `paper_stats.py` and `rl_locked_checks.py` in `src/analysis/nov3_drop_analysis/` (run them from the repo root to reproduce them). Result files are named in each section.

**How far the references were checked:** every reference was matched to Crossref, arXiv or the publisher by its exact identifier, except where its entry says otherwise, so they are all real papers with the right authors and venues. I read abstracts, or search summaries of them, for some, marked **[A]**, and only checked publication details for the rest, marked **[M]**. References [50]-[54], added Nov 6 from a web search, were checked on Nov 7 against Crossref and arXiv, with their abstracts read, so they're now **[A]** and list their published versions. Two had been described wrongly and are corrected: [53] reports that vision plus touch beat either alone, so it doesn't support "touch barely matters", and [52]'s policies were trained without touch in their state rather than having it removed at test time. References [55]-[68], added Nov 7 for the statistics and software, were checked by identifier, **[M]**. I did not read any paper in full. Read each paper before relying on it, especially **[M]** ones. Where a point below says what an [M] paper found, that comes from memory, so confirm it first.

---

## 0. The core of the paper

**The story in one line:** you set out to show that a better slip detector makes a better grasp, found that it was the smallest factor you measured, and then measured what does decide whether the object stays in the hand.

**Central question:** What decides whether a simulated slip-aware hand keeps hold of an object, and how much does the slip detector, the part usually reported and optimized, contribute?

**The answer in one figure (Figure 0, `fig00_factor_summary`).** Each factor's effect on the share of trials held at 5 cm, measured on the same trials on both sides (percentage points, 95% paired bootstrap interval). Where the fresh seeds give the same comparison under the primary elliptic settings, it's in the last column and drawn on the same row in blue:

| Factor | Effect | Trials | Same comparison, fresh seeds, elliptic (1,005) |
|---|---|---|---|
| RL training seed (seed 1 vs. seed 3) | +46.2 (+42.7 to +50.0) | Locked set, 960 | +36.5 (+32.5 to +40.4) |
| Controller (best RL seed vs. hand-tuned A) | +42.5 (+39.0 to +46.0) | Locked set, 960 | +32.9 (+29.9 to +36.0) |
| Grip (four fingers vs. all five, fixed grips) | +34.7 (+25.3 to +44.0) | Dev set, 150 | Not run |
| Success definition (no drop rule vs. 5 cm rule) | +25.7 (+23.2 to +28.5) | Seeds 0-66, 1,005 | +27.9 (+25.2 to +30.7), hand-tuned A |
| Contact settings (elliptic vs. default), RL seeds 1 and 2 | +21.3 and +20.7 (each about +11 to +30) | Dev set, 150 | Can't be paired: fresh seeds weren't run under default. Unpaired against the locked set, +18 to +29 for each of four controllers |
| Learning (best RL seed vs. fixed grip) | 0.0 (-5.3 to +5.3) | Dev set, 150 | +0.9 (-1.1 to +3.0) |
| Detector quality (F1 0.799 vs. 0.691) | +0.4 (-0.4 to +1.2) | Seeds 0-66, 1,005 | Not run |
| Detector removed (hand-tuned A; RL seed 1) | -2.0 (-5.3 to +0.7); 0.0 | Dev set, 150 | Not run |

- Say which trials each row comes from. The rows aren't all on one set, so compare their sizes, not their exact values.
- Under the primary settings the ranking holds. Seed and controller stay the largest factors, about 10 points smaller than under default, and learning over a fixed grip is now pinned near zero: at most +3 points.
- The fresh success-definition row is descriptive and wasn't pre-registered. The other fresh rows restate pre-registered comparisons as differences.

**Why this framing:** it is the most defensible result you have, every part of it is measured, and it is honest about where the project stands. Lead with the factor ranking, not a success rate. Don't present this as "we reached X%". In a talk: "I built a robot hand that detects slip, then discovered slip detection wasn't what kept it holding on."

**The five findings to carry the paper**
1. **How "held" is defined changes the result a lot.** The same 1,005 hand-tuned trials score 44.6%, 31.7% or 18.9%, depending on whether objects that fell before the pull count as failures (Figures 1, 2).
2. **The slip detector doesn't change closed-loop outcomes here.** Two detector checkpoints with F1 0.691 and 0.799 tie under every scoring rule (Figure 3). Removing the detector's signal changes 19 of 1,200 paired dev-set outcomes, none significantly, and the hand-tuned controller holds slightly more without it (Figure 4).
3. **RL found a good grip, not a feedback strategy.** Three training seeds held 80.6%, 60.5% and 44.1% of 1,005 fresh trials under elliptic, and 61.2%, 42.2% and 15.0% of the locked set under default (Figure 5). A fixed grip with no learning, sensing or detector matches the best seed: 79.7% against 80.6% on the same fresh trials, a paired difference of +0.9 points (-1.1 to +3.0). It beats the other two seeds and hand-tuned A (Figure 6). The seeds differ mainly in which fingers they close (Figure 11). The reward pays slightly more for trials where the object falls than for trials where it is held (Figure 12), and training reward ranks the seeds in the wrong order (Figure 14).
4. **The simulator's contact settings move every controller by 18-29 points, far more than any detector difference.** Under MuJoCo's default settings, objects creep out of grips that friction should hold. Under the elliptic cone, the same unretrained controllers hold about 21 points more on the same dev trials, and 18 to 29 points more on fresh trials than on the locked set (Figures 5, 7). For three of four controllers the gain comes mostly from fewer objects falling while the grasp forms, and for RL seed 3 mostly from fewer slips (Figure 17). Objects still creep at about 1 mm/s under load and still slip at constant full load (Figures 9, 10). Don't say the contact settings matter more than the controller: the seed and controller gaps are larger (Figure 0).
5. **Failures depend on object shape, and the controllers fail on largely the same trials.** Flat and boxy objects fail under every controller (Figure 8). Under default settings, a quarter of locked-set trials are held by none of the five controllers. For the four controllers run under both settings, that falls to 7% of fresh trials under elliptic (Figure 16).

These five are your results, so cite the figures named in each. Their outside references come where each is written up: the R sections, the Discussion and 4.6.

**What's new here, and what isn't** (from a quick literature search on Nov 6. Do a proper check before claiming any of it.)
- **Not new on its own:**
  - Offline metrics not predicting closed-loop results (driving [37, 38]). **Cite:** [37] Codevilla 2018; [38] Philion 2020.
  - RL results varying by seed [33, 34]. **Cite:** [33] Henderson 2018; [34] Agarwal 2021.
  - Success criteria mattering in grasp benchmarks [39, 40]. **Cite:** [39] Bekiroglu 2020; [40] Calli 2017.
  - MuJoCo's soft contacts letting objects slide (its own documentation [48]; contact models compared across simulators [50]). **Cite:** [48] MuJoCo docs, Modeling; [50] Le Lidec 2024.
  - Learned grasping needing little touch in some setups: with touch in the reward, policies trained without touch in their state lost at most 6.6% [52]. In another, touch helped RL grasping under imperfect visual perception [54]. **Cite:** [52] Koenig 2022; [54] Zhang 2025. Don't cite [53] for this: its abstract reports vision plus touch beating either alone (92% vs. 77%).
- **Plausibly new, phrased with "to our knowledge":**
  1. A factor-by-factor accounting, on the same paired trials, in which the slip detector is the smallest factor measured, while choices that papers rarely report each move success by 20-46 points. The search found no study that ranks these factors in one setup. **Cite:** your Figure 0. No outside reference: this is the novelty claim.
  2. How much the simulator's contact settings change a grasping benchmark, and why: +18 to +29 points for each of four controllers (about +21 paired on the dev set), mostly by keeping objects in the hand while the grasp forms, with creep at full load only slightly reduced. The artifact is documented, but its size in a grasping benchmark didn't turn up. **Cite:** [48] MuJoCo docs, Modeling, for the documented artifact; your Figures 7 and 17.
  3. What this means for physics-informed slip detection: data generated under default settings contains slips at a median 56% of the friction limit, so a Coulomb friction-cone loss would contradict its own labels. Check the simulator's physics before informing a model with physics. **Cite:** [25] Raissi 2019; [26] Karniadakis 2021. The 56% figure is in `results.md`; give it in the text.
- **Suggested novelty sentence:** "To our knowledge, this is the first study to measure, on the same trials, how much a slip detector's quality, the success definition, the simulator's contact settings and RL training variance each change closed-loop grasp success. The detector was the smallest of these factors."
- **Keep the claim narrow.** Closed-loop slip control has helped on real multi-fingered hands (e.g., [51]). So don't say slip feedback doesn't help grasping. Say it can't help in a benchmark where a fixed grip already decides the outcome. **Cite:** [51] James & Lepora 2021.

**Don't claim**
- That this is physics-informed. The current Model B has no physics loss (see 5.5).
- That any of this is novel, unless your literature check supports it. Offline metrics failing to predict closed-loop results is already known in driving research [37, 38], so the possible contribution is showing it for tactile slip detection and grasping, with these specifics.
- Anything about real hardware. This is simulation only.
- That RL training beats hand-tuned control on average. Two of three policies do, one doesn't, and the three-repeat mean isn't established as better, under default (R9) or under elliptic on fresh seeds (R22).
- That the RL policies use touch or the slip signal. Removing the signal changes nothing measurable, and a fixed grip matches the best policy (R15, R16).
- That the elliptic setting is physically correct, or that it removes creep. It shrinks a documented solver artifact. Objects still move about 1 mm/s at full load, and pad friction still doesn't predict holding (R17, R19).
- That higher training reward means a better policy (R18).
- That noslip is better than elliptic. They fail in different places, and that comparison wasn't pre-registered (R22).
- That learned policies ignoring touch, or MuJoCo contacts sliding, are discoveries of this paper. Both are documented (see "What's new here").

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
| 81.3% or 78.7% as a result | Seed 1 on the much-used dev set, best of 3, under settings chosen after seeing scores. Use the fresh-seed elliptic numbers instead (R17, R22) |
| 61.3% for the slower fixed grip as an unbiased figure | Best of several grips on 150 trials, with speeds read from seed 1 (R16) |
| 72.4% (RL seed 2 under noslip) as reaching 70% | Its interval reaches down to 69.6%, and noslip is the sensitivity setting, not the primary one (R22) |
| 67.6% or 71.7% (noslip three-seed means) as the headline | Noslip is the sensitivity result. Elliptic was fixed as primary before any fresh-seed result was read (R17, R22) |
| Dev-set rates as unbiased levels | The dev set understated the controllers, by up to 7.9 points (Figure 18). Use fresh-seed numbers for levels |

**`[YOUR CALL]` before writing**
- **Benchmark contact settings: decided.** Elliptic with `impratio` 10 is primary, with default beside it in every table (R17). You can still overrule it.
- Target venue and format (length, sections, citation style). References below are numbered and can be reformatted.
- Which Model A to cite. My recommendation: label closed-loop results as F1 0.691 and report static F1 as a range (see R2).
- Whether to report 5 cm, 10 cm or both. Plan so far: both, with 5 cm primary for RL because it was the existing rule.
- RL evaluation mode, because the pre-registered check tripped (R13). My recommendation: deterministic as primary, since every locked-set run used it and it's the common way to evaluate PPO, with the stochastic numbers reported beside it.
- RL headline number. The pre-registered rule makes it the three-seed mean and SD: 61.7%, SD 18.3, at 5 cm under elliptic on fresh seeds (the primary setting), with 39.5%, SD 23.2, under default on the locked set beside it. Seed 1 is shown as "best of 3 seeds, deterministic". Changing that now would be changing a rule after seeing results, so I recommend keeping it.
- Whether the fixed grip is presented as the baseline that RL and the detector have to beat. I recommend it, because it is the fairest reference for "did learning or sensing add anything".

---

## 1. Title options

1. What Actually Holds the Object? Measuring the Factors Behind Grasp Success in a Simulated Slip-Aware Hand (my recommendation, because it matches Figure 0)
2. Grip, Not Feedback: What Closed-Loop Evaluation Shows About Slip Detection and Learned Grasping in Simulation
3. Does a Better Slip Detector Mean a Better Grasp? Static Metrics, Success Definitions and Closed-Loop Control on a Simulated Five-Fingered Hand
4. Measuring What Matters in Slip-Aware Grasping: A 1,005-Trial Simulation Study of Hand-Tuned and Learned Control

---

## 2. Abstract (one sentence or two per point)

**Cite:** nothing. Abstracts carry no citations or figure numbers.

1. **Problem:** stable grasping needs detecting and responding to slip, and slip detectors are usually judged by static classification scores.
2. **Question:** in a simulated slip-aware grasping pipeline, how much does the detector actually contribute to keeping hold of an object, compared with the other choices a study makes?
3. **Method:** a simulated five-fingered ORCA hand in MuJoCo, 15 object shapes, a TCN slip detector trained on 20,000 simulated trials, a hand-tuned controller, PPO, and a fixed grip that uses no sensing. 1,005 paired trials per condition, with evaluation rules written in advance.
4. **Approach:** measure, on the same trials, how much the held rate changes with detector quality, removing the detector, the success definition, the simulator's contact settings, the grip, learning, and the RL training seed.
5. **Result 1, the detector was the smallest factor:** checkpoints with F1 0.691 and 0.799 differed by 0.4 points (paired p = 0.45 at 5 cm), and removing the detector changed 19 of 1,200 paired outcomes.
6. **Result 2, choices around the method were large:** the success definition moved identical trials by 26 points (44.6% vs. 18.9%), and stricter contact settings raised each of four unretrained controllers by 18 to 29 points on fresh trials (hand-tuned control from 18.8% to 47.7%), mostly by keeping objects in the hand while the grasp formed.
7. **Result 3, RL found a grip, not feedback:** under the primary contact settings, three training repeats held 80.6%, 60.5% and 44.1% of 1,005 fresh trials (mean 61.7%, SD 18.3) against 47.7% for hand-tuned control. Under MuJoCo's defaults they held 61.2%, 42.2% and 15.0% (mean 39.5%, SD 23.2) against 18.8%. A fixed grip with no learning held 79.7%, matching the best repeat (paired difference +0.9 points, -1.1 to +3.0), and the reward paid as much for dropped objects as for held ones. `[YOUR CALL: confirm the evaluation mode, R13]`
8. **Conclusion:** grasp benchmarks should verify the object is still held, report their contact settings, evaluate detectors in closed loop against a fixed-grip baseline, and report all training repeats. Physics-informed slip detection also needs a simulator whose contacts obey the physics it encodes.

---

## 3. Introduction

**Paragraph 1: Why grasping is hard**
- Robots need to hold objects they haven't seen, under uncertain friction and load. **Cite:** [3] Bohg 2014.
- Contact mechanics and grasp stability are the classical foundations [1, 2]. Data-driven grasping has grown alongside analytic methods [3]. **Cite:** [1] Bicchi & Kumar 2000; [2] Prattichizzo & Trinkle 2016; [3] Bohg 2014.
- Humans rely heavily on touch: fingertip signals report contact and slip, and grip force is adjusted to load [4, 5]. Use this to motivate tactile feedback. **Cite:** [4] Johansson & Flanagan 2009; [5] Westling & Johansson 1984.

**Paragraph 2: Slip detection as a route to stable grasping**
- Slip detection has a large literature across sensors and methods, reviewed in [6]. **Cite:** [6] Romeo & Zollo 2020.
- Examples: a tactile grasp controller on a PR2 with human-inspired signals [7], random forests predicting slip inside a stabilizing feedback loop on unknown objects [8], learned models using GelSight tactile and visual data [11], and incipient-slip monitoring with GelSlim [12]. **Cite:** [7] Romano 2011; [8] Veiga 2015; [11] Li 2018; [12] Dong 2019.
- Tactile input can also improve grasp outcome prediction [13]. **Cite:** [13] Calandra 2017.

**Paragraph 3: Learned control for dexterous hands**
- Model-free RL has scaled to high-dimensional hands in simulation [15] and has transferred to real hands with heavy randomization [16]. **Cite:** [15] Rajeswaran 2018; [16] OpenAI 2020.
- Tactile features have been combined with RL for regrasping [14]. **Cite:** [14] Chebotar 2016.
- Low-cost open hands such as ORCA make this kind of work more accessible [17]. **Cite:** [17] Christoph 2025.

**Paragraph 4: The gap this paper addresses**
- Detectors are usually reported with static metrics like accuracy and F1. **Cite:** [6] Romeo & Zollo 2020, plus two or three detection papers as examples, such as [10] James 2018 and [11] Li 2018. Check that each reports offline accuracy or F1 before citing it for this.
- In autonomous driving, offline prediction error isn't necessarily correlated with driving quality, and models with identical error can behave very differently [37]. Task-agnostic metrics also treat all errors alike [38]. **Cite:** [37] Codevilla 2018; [38] Philion 2020.
- RL results depend heavily on seeds and implementation details, and reporting single point estimates from few runs can mislead [33, 34, 35, 36]. **Cite:** [33] Henderson 2018; [34] Agarwal 2021; [35] Andrychowicz 2021; [36] Engstrom 2020.
- Grasp benchmarking work stresses repeatable conditions and explicit success criteria [39, 40]. **Cite:** [39] Bekiroglu 2020; [40] Calli 2017.
- Simulated contact is itself a modelling choice: MuJoCo's soft contacts allow slow slip inside the friction cone unless configured otherwise [19, 20, 48]. **Cite:** [19] Todorov 2014; [20] MuJoCo docs, Computation; [48] MuJoCo docs, Modeling.
- Point to make: the effect of these measurement choices on a slip-aware grasping pipeline should be measured directly.

**Paragraph 5: What this paper does (contributions)**
1. A factor-by-factor measurement, on the same paired trials, of what decides whether a simulated slip-aware hand keeps hold of an object: detector quality, the detector's presence, the success definition, the simulator's contact settings, the grip, learning and the RL training seed (Figure 0).
2. Evidence that the slip detector is the smallest of these factors here, and why: the grip formed in the first half second decides the outcome.
3. A measurement of how much MuJoCo's contact settings change a grasping benchmark, where the change comes from, and what it means for physics-informed learning.
4. A comparison of hand-tuned control, PPO and a fixed grip on the same trials, reporting all training repeats.
5. A 1,005-trial paired closed-loop benchmark across 15 shapes, with evaluation rules written before results, and public code and result files. `[YOUR CALL: confirm the repo is public before saying this]`

**Cite:** none; these are your contributions. Point to Figure 0 in the first, and cite [42] Nosek 2018 if you mention rules written before results.

---

## 4. Related work (optional section, or merge into the introduction)

**4.1 Slip detection and tactile grip control**
- Survey [6]. Tactile classifiers and force estimation with biomimetic sensors [9, 10]. Vision-based tactile sensing [11, 12]. **Cite:** [6] Romeo & Zollo 2020; [9] Su 2015; [10] James 2018; [11] Li 2018; [12] Dong 2019.
- Grip controllers using tactile events [7, 8], and closed-loop slip control on a multi-fingered tactile hand [51]. These show slip feedback can help on real hardware, so the claim here is narrower: it can't help where a fixed grip already decides the outcome. **Cite:** [7] Romano 2011; [8] Veiga 2015; [51] James & Lepora 2021.
- Difference here: fingertip force and torque signals in simulation, with a causal TCN, and the focus on evaluation. **Cite:** [22] Bai 2018 for the TCN.

**4.2 Temporal convolutional networks**
- Dilated causal convolutions came to prominence in WaveNet [24]. **Cite:** [24] van den Oord 2016.
- TCNs for action segmentation [23], and a general empirical case that simple TCNs can outperform common recurrent networks on sequence tasks [22]. **Cite:** [23] Lea 2017; [22] Bai 2018.
- Why they suit this task: causal, so usable in real time, with a fixed and known history window (255 steps here). **Cite:** [22] Bai 2018.

**4.3 Physics-informed machine learning**
- PINNs embed physical laws into training [25], reviewed in [26]. **Cite:** [25] Raissi 2019; [26] Karniadakis 2021.
- Say clearly that this paper doesn't yet use a physics loss. Model B only adds physics-related input features. Present a physics loss as future work.

**4.4 RL for dexterous manipulation**
- PPO [30], and the Stable-Baselines3 implementation used here [31]. **Cite:** [30] Schulman 2017; [31] Raffin 2021.
- Dexterous RL [15, 16], sim-to-real and randomization [16, 21]. **Cite:** [15] Rajeswaran 2018; [16] OpenAI 2020; [21] Tobin 2017.
- Residual approaches that learn on top of a controller [46, 47], relevant to future work. **Cite:** [46] Johannink 2019; [47] Silver 2018.

**4.5 Reproducibility and evaluation**
- RL variance and reporting [33, 34], implementation effects [35, 36]. **Cite:** [33] Henderson 2018; [34] Agarwal 2021; [35] Andrychowicz 2021; [36] Engstrom 2020.
- Leakage in ML-based science [41], relevant to the feature exclusions in 5.5. **Cite:** [41] Kapoor & Narayanan 2023.
- Preregistration separates hypothesis generation from testing [42]. **Cite:** [42] Nosek 2018.
- How much learned grasping relies on touch depends on the setup. With touch in the reward, policies trained without touch in their state lost at most 6.6% [52]. Touch features helped RL grasping under imperfect visual perception, while complex tactile inputs made training harder [54]. For following deformable cables, vision plus touch beat either alone, 92% against 77% [53]. **Cite:** [52] Koenig 2022; [54] Zhang 2025; [53] Pecyna 2022.

**4.6 Contact modelling in simulation**
- MuJoCo's soft-contact model [19, 20] and its documented slow slip under load [48]. A comparative analysis of the contact models in robotics simulators sets out each engine's physical relaxations, benchmarks them, and finds that common approximations can severely widen the reality gap [50]. **Cite:** [19] Todorov 2014; [20] MuJoCo docs, Computation; [48] MuJoCo docs, Modeling; [50] Le Lidec 2024.
- Published hand models set solver options for this: Allegro, Shadow Hand and Robotiq 2F-85 use an elliptic cone with `impratio` 10 [49]. **Cite:** [49] Zakka 2022, MuJoCo Menagerie.

---

## 5. Methods

### 5.1 Simulation platform
- MuJoCo [18] version 3.11.0, with its soft-contact model [19, 20]. Python 3.14.3, Stable-Baselines3 2.9.2 [31], Gymnasium [32], PyTorch [64]. **Cite:** [18] Todorov 2012; [19] Todorov 2014; [20] MuJoCo docs, Computation; [31] Raffin 2021; [32] Towers 2024; [64] Paszke 2019.
- ORCA five-fingered hand model [17], in `third_party/orcahand_description/v2/scene_slip.xml`. That file wasn't under version control, so an exact copy is archived at `Data/benchmark_scenes/scene_slip.xml` (SHA-256 starting `d9d3aefde2f9`). **Cite:** [17] Christoph 2025.
- 10 actuators are driven, an MCP and a PIP actuator per finger. Other actuators are held at a neutral value.
- Sensors on each of 5 fingertips: touch, force and torque. **Cite:** [69] MuJoCo docs, XML reference (sensors).
- Physics step 0.002 s, so a 2,300-step trial is 4.6 s of simulated time. `[CHECK: confirm with model.opt.timestep]`
- **Contact solver settings.** The scene sets none, so MuJoCo's defaults apply: pyramidal friction cone, `impratio` 1, no noslip pass [20]. Primary results use an elliptic cone with `impratio` 10, the setting MuJoCo's modeling guide recommends first against contact slip [48], and the one the Allegro, Shadow Hand and Robotiq 2F-85 models use [49]. Everything else is unchanged. Default-setting results are reported beside every elliptic result, and a noslip pass (20 iterations) is a sensitivity check. State that the setting was chosen after dev-set scores had been seen, and that it was not tuned further. **Cite:** [20] MuJoCo docs, Computation (the defaults); [48] MuJoCo docs, Modeling (the recommendation); [49] Zakka 2022, MuJoCo Menagerie (the hand models).

### 5.2 Objects and randomization
- 15 objects (Table 1):
  - **Primitives (8):** cube, sphere, rectangular prism, cylindrical prism, capsule, ellipsoid, flat disk, rod.
  - **Meshes (4):** cone, square pyramid, triangular prism, hexagonal prism.
  - **Compounds (3):** L-block, dumbbell, plus-shaped cross.
- Mass is computed from geometry at PLA density (1,240 kg/m³, solid infill). **Cite:** a PLA material datasheet for the density. `[needs a source]`
- Per-trial randomization, seeded so each trial is repeatable. **Cite:** none needed; if you call it domain randomization, cite [21] Tobin 2017. The ranges:
  - Position ±5 mm in the plane
  - Yaw 0-360°, roll and pitch ±10°
  - Size and mass ±10%
  - Object friction ±15% around about 0.4
  - Fingertip pad friction 0.35-1.2 per finger
- Explain the friction choice: MuJoCo combines contact friction using the element-wise maximum, so a fixed pad friction of 1.0 would have made object friction meaningless. Randomizing pad friction makes it causal. Source: docstring in `generate_trials.py`, confirmed empirically there. **Cite:** [48] MuJoCo docs, Modeling, "Contact parameters": with equal priority, "the element-wise maximum of each friction coefficient over the two geoms is used".

### 5.3 Trial protocol
1. **Ramp (500 steps):** the object is weightless through gravity compensation while the fingers close.
2. **Settle (300 steps):** gravity on. The reference position for slip is recorded at the end of this phase.
3. **Pull:** a downward force ramps over 500 steps up to 1.5x the object's weight, then holds for 1,000 steps.

Total 2,300 steps. Grip target for hand-tuned control: squeeze 0.95 of the actuator range.

**Cite:** none; this is your protocol.

### 5.4 Success definitions (a core methods section)
- **Slip:** displacement from the settle reference exceeds 15% of the object's characteristic length (6 mm for a 4 cm cube).
- **Formation fall:** drop in height from spawn during ramp and settle.
- Report every result under three rules:
  - **No drop rule:** held means no slip. This is the original hand-tuned scoring.
  - **10 cm rule:** also fails trials whose object fell more than 10 cm before the pull.
  - **5 cm rule:** also fails trials whose object fell more than 5 cm. This is RL's training and evaluation rule.
- Explain the flaw: displacement starts at the end of settle, so an object already on the floor barely moves and scores as held.
- Disclose that the 10 cm rule was chosen after seeing results. Justify it with data: for hand-tuned control, any cutoff from 7 to 20 cm gives the same score within 3 of 1,005 trials (316-319 for Model A). Figure 1 shows the whole curve.
- **Cite:** [39] Bekiroglu 2020 for stating success criteria explicitly. The rules themselves are yours; point to Figure 1 for the cutoff sensitivity.

### 5.5 Slip detection models
- **Data:** 20,000 simulated trials, with squeeze 0.30-1.00, pull 0.2-5.0x weight, ramp 400-600 steps, and recorded sensor noise setting 0.05. All generated under default contact settings.
- **Filter:** trials with no finger contact are dropped, leaving 17,514 (10,588 slip, 6,926 no-slip).
- **Split:** random by trial: 12,259 train, 2,627 validation, 2,628 test. Note that this tests new trials of the same objects, not new objects. **Cite:** [41] Kapoor & Narayanan 2023, if you explain what a random split can't test.
- **Labels:** per timestep, slip once displacement passes the threshold.
- **Input features, Model A (22 channels):** for each of 5 fingertips, the touch sensor (the actual normal force), the force sensor's reading along its axis 2, the magnitude of its other two axes, and torque magnitude; plus pull force and phase. The code names the second and third channels "normal" and "tangential", but the contact normal mostly lies along the other sensor axes, so axis 2 reads a median 18% of the touch sensor (Figure 15). Describe them as force components along fixed sensor axes, not as normal and tangential force. **Cite:** your Figure 15.
- **Excluded as leakage [41]:** object position, displacement and slip speed. Displacement defines the label, and object position wouldn't come from fingertip sensing. **Cite:** [41] Kapoor & Narayanan 2023.
- **Note on pull force and phase:** these are known commands, not measurements. Keep them in the description, because `results.md` notes that pull force carries visible onset signal while tactile channels showed no visible pre-slip pattern (Oct 15 entry). Mention this honestly.
- **Architecture:** causal TCN [22, 24], hidden size 32, kernel 3, 7 layers, dilations 1 to 64, receptive field 255 steps, with residual connections [63] and a 2-class output per timestep. **Cite:** [22] Bai 2018; [24] van den Oord 2016; [63] He 2016.
- **Model B (25 channels):** Model A plus three per-trial values broadcast across time: combined friction, object mass, and the summed axis-2 force at settle (named "grip force at settle" in the code, but it isn't grip force, for the reason above). The code calls this PI-TCN, but it has no physics loss. Call it something like "physics-feature TCN" in the paper. **Cite:** [25] Raissi 2019, for what "physics-informed" normally means.
  - Say that nine candidate physics features and losses were tested in October and did not separate slip onset from stable grasping (`results.md`, "Model B"). Those tests used the mislabeled channels and default contact settings, so don't present them as evidence that physics features can't work.
- **Training:** Adam [27] at learning rate 1e-3, batch 32, class-weighted cross-entropy with label smoothing 0.1 [28]. Model A used 20 epochs, Model B 40. Runs were unseeded. **Cite:** [27] Kingma & Ba 2015; [28] Müller 2019; [64] Paszke 2019.
- **Metrics:** per-timestep F1 and accuracy on the slip class, excluding padding. Because classes are imbalanced (22.6% slip timesteps), also give precision and recall [29]. **Cite:** [29] Saito & Rehmsmeier 2015.
- **Threshold:** Model B's 0.30 was chosen on validation data only, then applied once to test. Model A uses argmax.

### 5.6 Hand-tuned controller
- Runs during the pull phase, every 5 steps, once 300 steps of history exist.
- If slip probability is at or below 0.3: no change.
- Otherwise each actuator target moves toward its limit: `step = 0.35 x p x (limit - target)`. Corrections only tighten.
- Grip formation is scripted and identical for every object.
- Tuned using earlier 45-trial experiments. Variants that did not help: lower trigger thresholds, smoothing, per-finger correction (`results.md`). Report these as exploratory.
- **Cite:** none for the controller, which is yours. If you link it to grip-force adjustment in humans or to slip-triggered grip controllers, cite [4] Johansson & Flanagan 2009 and [7] Romano 2011.

### 5.7 RL controller
- PPO [30] in Stable-Baselines3 [31], with the custom Gymnasium [32] environment `SlipGraspEnv`. **Cite:** [30] Schulman 2017; [31] Raffin 2021; [32] Towers 2024.
- **Control scope:** the policy controls the whole episode, including grip formation, starting from an open hand.
- **Action:** 5 values, one per finger, each changing that finger's closure by up to 0.08 of its range per step. Each action repeats for 5 physics steps.
- **Observation (115 values):** the last 5 readings of tactile channels, pull force, phase and TCN slip probability. No object position. The TCN is frozen (file `ce813826`, F1 0.691).
- **TCN warm-up:** probability is 0 for the first 300 physics steps, so before the pull begins at step 800 the policy has both zero and live values.
- **Reward:**
  - Formation: contact 1.0, grip force 0.15 (capped at 2x weight, and computed from the axis-2 channel, so it isn't grip force), object speed penalty 0.1, drop penalty 15.
  - Pull: displacement penalty 50, slip penalty 20, hold bonus 20.
  - Effort penalty 0.02 throughout.
- **Termination in training:** falling past 5 cm during formation, slipping, or reaching the time limit.
- **Hyperparameters:** rollout 256 per environment, batch 256, learning rate 3e-4, 4 environments, 300,000 steps. Other values are library defaults [31]: gamma 0.99, GAE lambda 0.95 [62], clip 0.2 [30], 10 epochs. No observation or reward normalization. **Cite:** [31] Raffin 2021; [62] Schulman 2016; [30] Schulman 2017.
- **Training physics:** all policies were trained under default contact settings. None was retrained under elliptic.
- **Training objects:** 11. Held-out objects: cone, cylindrical prism, plus cross, sphere.
- **Repeats:** three independent runs of one configuration, not explicitly seeded. They also needed different numbers of process launches because of a MuJoCo memory failure (8, 2 and 1), so they aren't perfectly identical procedures. Say so. **Cite:** [33] Henderson 2018; [34] Agarwal 2021, for why every repeat is reported.

### 5.8 Fixed-grip baseline
- No policy, tactile input or TCN. Each finger closes to a fixed target at a fixed speed in the same environment, with the same scoring.
- The main grip closes thumb, index, ring and pinky at full speed and leaves the middle finger open. It was designed after looking at seed 1's finger traces, so on the dev set it is optimistic. That's why it was re-measured on fresh seeds (R16, R22).
- **Cite:** none; it's your baseline. If you relate it to residual RL, which learns corrections on top of a fixed controller, cite [46] Johannink 2019 and [47] Silver 2018.

### 5.9 Evaluation
- **Locked set:** 15 objects x seeds 0-66, 1,005 trials, paired across conditions. Seeds 3-66 (960 trials) are the primary figure for RL comparisons, because seeds 0-2 were already used during development. These give the default-physics numbers.
- **Fresh seeds:** 15 objects x seeds 2000-2066, 1,005 trials, untouched before Nov 4. Pre-registered measurement of the existing controllers under noslip and elliptic. With no retrain, these give the final elliptic numbers (`results.md`, Nov 6 amendment, written before any elliptic fresh-seed row existed). **Cite:** [42] Nosek 2018.
- **Dev set:** 15 objects x seeds 1000-1009, 150 trials. Used for choices between configurations and for the paired contact-setting comparisons. Results on it are exploratory.
- **Original benchmark:** seeds 0-2, 45 trials, reported for continuity.
- **Pre-registered rules:** written in `results.md` before those runs [42]. A new configuration counts as "improved" only if every new repeat beats every baseline repeat and the mean gain is at least the larger of 15 points or 1 baseline SD. **Cite:** [42] Nosek 2018.
- **Deviations to disclose:** the 10 cm rule was added after seeing results, and locked-set runs ran before dev-set checks. The elliptic setting was proposed after dev-set scores had been seen. No choices used the locked set. **Cite:** [42] Nosek 2018, on reporting deviations from a pre-registered plan.
- **Evaluating RL under other rules:** an evaluation-only option lets trials continue past 5 cm and records the maximum fall. Policies were never trained past 5 cm, so behaviour beyond it is untrained.
- **Statistics:**
  - Wilson 95% intervals for rates [43, 44]. **Cite:** [43] Wilson 1927; [44] Brown 2001.
  - Exact McNemar test for paired trial comparisons [45]. **Cite:** [45] McNemar 1947.
  - Paired bootstrap 95% intervals for differences in held rate (Figure 0): trials resampled with their pairs kept together, 4,000 resamples [59]. **Cite:** [59] Efron 1979.
  - Chi-square test that training seeds share one held rate [55]. **Cite:** [55] Pearson 1900.
  - Mann-Whitney tests for reward distributions [56], Wilcoxon signed-rank for matched object speeds [57], and a Mantel-Haenszel test, stratified by object, for pad friction [58]. **Cite:** [56] Mann & Whitney 1947; [57] Wilcoxon 1945; [58] Mantel & Haenszel 1959.
  - Bonferroni correction across the 12 pad-friction tests [60]. **Cite:** [60] Dunn 1961.
  - One-sample t-test on the three seed-minus-hand-tuned differences (df 2), labelled not pre-registered [61]. **Cite:** [61] Student 1908.
  - For static metrics, intervals at trial level, resampling whole trials, because timesteps within a trial are strongly correlated (`src/analysis/tcn_f1_without_drops.py`). **Cite:** [59] Efron 1979.
  - Non-significant differences are reported with their intervals, not read as "no effect" [67, 68]. **Cite:** [67] Altman & Bland 1995; [68] Wasserstein & Lazar 2016.
  - For RL, report every repeat, with mean and SD across repeats [33, 34]. **Cite:** [33] Henderson 2018; [34] Agarwal 2021.
  - Tests computed with SciPy [65], figures drawn with Matplotlib [66]. **Cite:** [65] Virtanen 2020; [66] Hunter 2007.
- **Reproducibility:** each trial is seeded. Result rows record hashes of the checkpoint, policy, code and scene. Re-running the three RL policies reproduced every logged outcome (R7). Every figure is built from stored rows by `src/analysis/paper_figures.py`, which checks each file's trial count and held count against `results.md` before using it.

---

## 6. Results

### R1. Static detection (Table 2)
Whole test split, 2,628 trials, per-timestep pooled. Source: `Data/_tcn_f1_subsets.log`. **Cite:** Table 2.

| Checkpoint | Rule | F1 (95% CI) | Precision | Recall | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| `ce813826` Model A, used closed-loop | argmax | 0.691 (0.673-0.707) | 0.904 | 0.559 | 0.983 | 0.887 |
| `dbb50c5a` Model A, older run | argmax | 0.773 (0.759-0.786) | 0.786 | 0.760 | 0.940 | 0.899 |
| `948a2716` Model B | 0.50 | 0.772 (0.757-0.787) | 0.913 | 0.669 | 0.981 | 0.911 |
| `948a2716` Model B | 0.30 | 0.799 (0.785-0.811) | 0.793 | 0.804 | 0.939 | 0.908 |

- At the same decision rule, the older Model A and Model B are effectively tied. Model B's lead depends on its tuned threshold. It also had 40 epochs against 20.
- All four rows reproduced their logged values within ±0.001 before any further analysis.

### R2. Run-to-run variance in the detector
**Cite:** none outside; Figure 19 once your seeded runs exist.
- The same nominal Model A configuration has produced F1 0.773, 0.722 (at 40 epochs) and 0.691 across unseeded runs (0.773 and 0.691 at 20 epochs). Early runs gave 0.715-0.719 before label smoothing.
- So single-run F1 comparisons aren't reliable here. `[PENDING: mean and SD from 5 seeded runs per model, if completed. This is your training code to change]`
- Disclose: `model_a.pt`, used in all closed-loop and RL work, is the F1 0.691 run.

### R3. Effect of drop trials on F1
Same test split, split at the 5 cm rule (1,224 no-drop trials, 1,404 drop trials). **Cite:** this table, as supplementary.

| Checkpoint | Rule | All | No drop | Drop |
|---|---|---|---|---|
| `ce813826` | argmax | 0.691 | 0.789 | 0.559 |
| `dbb50c5a` | argmax | 0.773 | 0.868 | 0.660 |
| `948a2716` | 0.50 | 0.772 | 0.868 | 0.638 |
| `948a2716` | 0.30 | 0.799 | 0.879 | 0.703 |

- F1 is higher without drop trials, so these trials don't appear to be inflating it.
- Caveat: the slip base rate differs between subsets (26.5% vs. 19.3%). No near-floor split has been run.

### R4. Fall sizes before the pull (Figure 2)
Hand-tuned, 1,005 trials. Falls don't depend on the model: flags match between Models A and B on all 1,005 trials. **Cite:** Figure 2.

| Fall | Trials | Not slipped in pull | Reading |
|---|---|---|---|
| 5 cm or less | 424 | 190 | Gripped |
| 5-7 cm | 346 | 126 | Sank, but appears gripped |
| 7-15 cm | 88 | 3 | Mostly lost |
| Over 15 cm | 147 | 129 | Near floor level. Not real holds |

- The 5 cm cutoff sits in the densest part of the distribution (4.5-5.0 cm: 117 trials, 5.0-5.5 cm: 171 trials). Of hand-tuned A's 581 falls past 5 cm, 412 are between 5 and 10 cm (Figure 2).
- Evidence on the 5-7 cm group, from a sample of 1,500 training trials at end of settle:

| Fall | n | Median fingers in contact | Median summed force | Slip rate |
|---|---|---|---|---|
| 5 cm or less | 739 | 4 | 16.2 | 75.1% |
| 5-7 cm | 232 | 5 | 49.4 | 75.4% |
| 7-15 cm | 150 | 3 | 0.15 | 98.0% |
| Over 15 cm | 379 | 3 | 0.15 | 7.9% |

- Caveat: training data at mixed squeeze levels, not the closed-loop benchmark.

### R5. Hand-tuned closed-loop results (Table 3, Figure 1)
2,300 steps, 1,005 trials. Source: `Data/handtuned_model{A,B}_2300step_results.jsonl`. **Cite:** Table 3, Figure 1.

| Model | No drop rule | 10 cm rule | 5 cm rule |
|---|---|---|---|
| A (F1 0.691) | 44.6% (41.5-47.7), 448 | 31.7% (28.9-34.7), 319 | 18.9% (16.6-21.4), 190 |
| B (F1 0.799) | 44.7% (41.6-47.8), 449 | 31.8% (29.0-34.8), 320 | 19.3% (17.0-21.9), 194 |

- Seeds 3-66 (960 trials): A 31.8% and 18.8%, B 32.1% and 19.3% at 10 cm and 5 cm.
- Test length also matters. On the short 1,500-step test with 200 steps at full pull, the same rules give 58.6%, 45.5% and 26.6% for A, and 57.9%, 44.8% and 25.9% for B. These were corrected using falls from the 2,300-step run. Model A's shorter and longer runs are consistent on every trial.
- Sensitivity: Model A at cutoffs of 5, 6, 7, 8 and 10-20 cm scores 190, 302, 316, 317 and 319 (Figure 1, panel a).

### R6. Static F1 vs. closed-loop performance (Figure 3)
**Cite:** Figure 3, and Figure 4 for the TCN removal. Keep the driving comparison [37, 38] for the Discussion. For the "Be careful" point, a tie isn't proof of no effect: [67] Altman & Bland 1995.
- Model A and Model B are tied on paired trials at every rule:
  - No drop rule: 12 held only by A, 13 only by B (exact McNemar p = 1.00)
  - 10 cm: 12 and 13 (p = 1.00)
  - 5 cm: 6 and 10 (p = 0.45)
  - 1,500-step test, no drop rule: 14 and 7 (p = 0.19)
- First mechanism: most holds needed no correction at all. Under the 10 cm rule, 256 of A's 319 holds (80.3%) and 207 of B's 320 holds (64.7%) had zero corrections.
- Stronger evidence: removing the TCN from the hand-tuned controller doesn't lower its held rate. On the dev set it holds 38 of 150 without the TCN against 35 with it under default settings (paired 4 vs. 1, p = 0.375), and 71 against 66 under noslip (6 vs. 1, p = 0.125) (R15, Figure 4). So the corrections don't help here, and may slightly hurt.
- Grip strength isn't the limit. The actuators are pinned at their maximum by the end of the pull in the 7 trials checked, but fingertip contact forces reach 20-70 N on objects under 1.5 N. Don't use "limited actuator headroom" as the explanation.
- **Be careful:** this is two checkpoints with one controller. They also differ in inputs, epochs and operating point. It shows that the F1 gap didn't matter here, not that detection never matters. It can only matter where a fixed grip fails and the controller has a reason to act.
- `[PENDING, optional: hand-tuned run with the F1 0.773 file. Same architecture as the 0.691 file, so it's a cleaner test]`

### R7. Reproducibility of RL results
**Cite:** none; give the numbers in the text.
- All three policies, re-evaluated on the original 45 trials, reproduced their logged results: 29, 16 and 8 of 45. Per-trial outcomes matched for the two repeats with per-trial logs, and per-object counts matched for the third.
- So the spread comes from training, not evaluation noise.

### R8. RL vs. hand-tuned on the same 45 trials (Table 4)
**Cite:** Table 4.

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

### R9. RL on the locked set (Table 5, Figure 5)
Deterministic, default contact settings. Seeds 3-66 are primary, because seeds 0-2 were already used to compare and choose policies. Files: `Data/rl_seed{1,2,3}_1005trial_results.jsonl`. No errored trials. **Cite:** Table 5, Figure 5. Outside: [33] Henderson 2018 and [34] Agarwal 2021 where you note the seed spread.

| Held, seeds 3-66 (95% CI) | 5 cm | 10 cm |
|---|---|---|
| Repeat 1 (best of 3 seeds, deterministic) | 588/960, 61.2% (58.1-64.3) | 610/960, 63.5% (60.4-66.5) |
| Repeat 2 | 405/960, 42.2% (39.1-45.3) | 423/960, 44.1% (41.0-47.2) |
| Repeat 3 | 144/960, 15.0% (12.9-17.4) | 147/960, 15.3% (13.2-17.7) |
| **Mean and SD across repeats** | **39.5, SD 23.2** | **41.0, SD 24.3** |
| Hand-tuned A | 180/960, 18.8% | 305/960, 31.8% |
| Hand-tuned B | 185/960, 19.3% | 308/960, 32.1% |

- The three repeats differ far more than sampling noise allows: chi-square 434, p < 0.0001, while sampling SE per repeat is at most 1.6 points.
- All 1,005 trials, 5 cm / 10 cm: repeat 1 61.4% / 63.6%, repeat 2 41.9% / 43.8%, repeat 3 15.1% / 15.4%. Mean 39.5 / 40.9, SD 23.2 / 24.2.
- Paired against hand-tuned A, seeds 3-66 (held only by RL, only by hand-tuned, exact McNemar):
  - 5 cm: repeat 1 438 and 30 (p = 5e-94), repeat 2 299 and 74 (p = 4e-33), repeat 3 111 and 147 (p = 0.029).
  - 10 cm: repeat 1 353 and 48 (p = 2e-58), repeat 2 250 and 132 (p = 2e-9), repeat 3 95 and 253 (p = 1e-17).
  - Against hand-tuned B the pattern is the same: at 5 cm 433 and 30, 297 and 77, 109 and 150. At 10 cm 350 and 48, 250 and 135, 92 and 253.
- Two policies beat hand-tuned and one is worse than it. That's the honest statement of finding 3.
- Three-repeat mean against hand-tuned A, not pre-registered, df 2: 5 cm difference 20.7 points (p = 0.26), 10 cm difference 9.2 points (p = 0.58). Three repeats with this spread can't establish that the training procedure beats hand-tuned on average.
- The 45-trial numbers held up roughly at 5 cm: 64.4% became 61.2%, 35.6% became 42.2%, and 17.8% became 15.0%.
- No repeat reaches 70% under default settings, and the top of repeat 1's interval is 64.3% at 5 cm and 66.5% at 10 cm.
- Outcomes at 5 cm, all 1,005 (held, fell past 5 cm, slipped without falling): repeat 1 617, 200, 188. Repeat 2 421, 291, 293. Repeat 3 152, 486, 367. Worse repeats fail in both ways, especially falling.
- Training vs. held-out objects at 5 cm, seeds 3-66: repeat 1 57.1% vs. 72.7%, repeat 2 38.8% vs. 51.6%, repeat 3 15.5% vs. 13.7%. Hand-tuned A scores 15.6% vs. 27.3%, even though it has no training objects. So this likely reflects which shapes are held out, not generalization. Don't claim generalization from it.

### R10. Failure analysis by object (Table 6, Figure 8)
Hand-tuned Model A, 67 trials per object, under the 10 cm rule. **Cite:** Table 6, Figure 8.

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
- Dumbbell: 67/67 held under the 10 cm rule but 32/67 at 5 cm, because 35 trials sink 5-10 cm first. On the dev set, hand-tuned A holds it 6 of 10 under default settings and 1 of 10 under noslip, so this outlier depends on contact settings. Don't interpret it.
- Kind summary at 10 cm: primitives 150/536, meshes 73/268, compounds 96/201.
- RL per object at 5 cm, all seeds, held of 67 (repeat 1, 2, 3, then hand-tuned A at 5 cm): square pyramid 66, 65, 21, 25. Rod 63, 35, 0, 4. Cone 61, 66, 25, 15. Dumbbell 61, 17, 7, 32. Sphere 56, 43, 0, 48. Plus cross 53, 24, 8, 10. Ellipsoid 51, 37, 16, 23. Triangular prism 51, 15, 11, 0. Hexagonal prism 39, 33, 6, 0. Capsule 38, 31, 27, 21. Cylinder 26, 3, 2, 0. L-block 22, 21, 18, 3. Flat disk 14, 16, 1, 3. Rectangular prism 9, 6, 3, 5. Cube 7, 9, 7, 1.
- The best RL repeat reaches 70% on 8 of 15 objects. Cube and rectangular prism stay at 9 of 67 or fewer under every controller under default settings.
- Part of these "limits" is the contact model. Under elliptic on the dev set, seed 1's hardest shapes improve the most: flat disk 0 to 10 of 10, cube 0 to 5, rectangular prism 1 to 5 (R17). On fresh seeds under elliptic (Figure 8), seed 1 holds flat disk 61 of 67, cube 37 and rectangular prism 29, and reaches 70% on 10 of 15 objects, as does the fixed grip. Rectangular prism and cube stay among the three hardest for both. Full per-object counts are in `results.md` (Nov 7).

### R11. What the RL policies learned (Figure 11)
Median finger closure over time, dev set, noslip traces (closure under default is the same for seed 1 at every sampled step). **Cite:** Figure 11.
- **Seed 1** closes thumb, index and ring fully within 0.2 s and the pinky by 0.5 s, and keeps the middle finger open in the median trial. Only 19 of 150 trials move the middle finger at all by step 50.
- **Seed 2** closes thumb, index and pinky, holds the ring finger at about 0.75, and opens the middle finger again after briefly moving it.
- **Seed 3** closes only thumb, index and pinky. It opens the ring finger again after briefly closing it.
- These patterns hardly change during the pull. So each seed settles on a grip in the first half second, and the seeds differ mainly in which fingers that grip uses. The weakest seed grips with three fingers.
- Failure types across repeats, on 45 trials (slipped, fell past 5 cm): repeat 1: 5 and 11. Repeat 2: 15 and 14. Repeat 3: 15 and 22.
- Correct the older log claim: hand-tuned control also passes 5 cm on 23 of these 45 trials, so RL didn't create this failure type.

### R12. Exploratory RL configurations
Keep brief, and label as exploratory: 45 trials and single runs. **Cite:** none; VecNormalize is a Stable-Baselines3 wrapper, so cite [31] Raffin 2021 if you name it.

| Change | Result | Note |
|---|---|---|
| 150,000 more training steps on the best policy | 57.8% | More training did not help |
| Reward term forcing all fingers to close | 44.4% | Fixed the finger, overall worse |
| Object size added to observations | 24.4% | Worse |
| VecNormalize | 2.2% | Not an evaluation bug. Direct and training-style paths agree on 45/45 trials, and a stochastic run matches the training values (reward +1.0%, length -5.3%) |

Reading: given a 17.8-64.4% spread for the unchanged configuration, none of these single runs can be attributed to its change.

### R13. Deterministic vs. stochastic evaluation
Dev set, 150 paired trials, same start conditions, noise seeded per trial. The pre-registered rule trips on all three policies at both cutoffs. Files: `Data/rl_seed{1,2,3}_dev_{deterministic,stochastic}.jsonl`. **Cite:** none outside; [31] Raffin 2021 if you explain that deterministic evaluation uses the policy's mean action.

| 5 cm | Deterministic | Stochastic | Only deterministic / only stochastic | Difference (limit), points | p |
|---|---|---|---|---|---|
| Repeat 1 | 86, 57.3% | 73, 48.7% | 19 / 6 | 8.7 (6.7) | 0.015 |
| Repeat 2 | 56, 37.3% | 35, 23.3% | 24 / 3 | 14.0 (6.9) | <0.001 |
| Repeat 3 | 19, 12.7% | 8, 5.3% | 14 / 3 | 7.3 (5.5) | 0.013 |

- At 10 cm: 91 vs. 75 (10.7, limit 6.8), 60 vs. 39 (14.0, limit 7.4), 19 vs. 8 (7.3, limit 5.5).
- Every RL rate in this paper measures the policy's mean action. Sampling from the policy as trained holds 7-14 points fewer. State this next to the RL results.
- `[YOUR CALL: which mode is primary. My recommendation is deterministic, with stochastic reported beside it]`

### R14. Longer training rollouts: worse
**Cite:** none; give the numbers in the text.
- 3 new repeats with rollout 512, all else unchanged, each 301,056 steps with no crash.
- Dev set, deterministic: 10.0%, 0.0% and 2.7% at 5 cm (mean 4.2, SD 5.2), and the same at 10 cm. Baseline: 57.3%, 37.3%, 12.7% at 5 cm (mean 35.8, SD 22.4), and 60.7%, 40.0%, 12.7% at 10 cm.
- Pre-registered verdict at both cutoffs: **worse**. Every new repeat is below every baseline repeat, and the mean drop (31.6 and 33.6 points) passes the bar (22.4 and 24.1).
- SD falls only because all three are near the floor. So the hypothesis that rollouts shorter than a trial cause the spread isn't supported.
- Why it got worse isn't established. Both take about the same gradient steps, but this collects half as many rollouts. That's a hypothesis.
- Mean evaluation reward doesn't rank these policies by holding: 434 with 4 held against 415 with 19 held.
- Deterministic only, and the evaluation mode is still your call (R13).

### R15. The TCN signal in closed loop (Figure 4)
Dev set, the same 150 trials with and without the TCN. For RL the slip probability is replaced by 0; for hand-tuned control the TCN is removed, so no corrections happen. **Cite:** Figure 4. Compare with other tactile ablations, [52] Koenig 2022 and [54] Zhang 2025, in the Discussion, not here.

| Controller and setting | With TCN | Without | Held only with / only without | p |
|---|---|---|---|---|
| RL seed 1, default | 86 | 86 | 0 / 0 | 1.00 |
| RL seed 2, default | 56 | 56 | 0 / 0 | 1.00 |
| RL seed 3, default | 19 | 19 | 0 / 0 | 1.00 |
| RL seed 1, noslip | 122 | 122 | 0 / 0 | 1.00 |
| RL seed 2, noslip | 101 | 99 | 4 / 2 | 0.69 |
| RL seed 3, noslip | 61 | 60 | 1 / 0 | 1.00 |
| Hand-tuned A, default | 35 | 38 | 1 / 4 | 0.375 |
| Hand-tuned A, noslip | 66 | 71 | 1 / 6 | 0.125 |

- 19 of 1,200 outcomes change: 7 trials are held only with the TCN, 12 only without. The 10 cm results are the same story.
- Trajectories barely differ: seed 1 has the same trial length on 145 of 150.
- **Be careful:** this tests the signal's use at run time. No policy was trained without it, so it doesn't show that training with it was useless.

### R16. A fixed grip matches the best RL policy (Figure 6)
**Cite:** Figure 6 (panel a dev set, panel b fresh seeds).
Fixed grips: no policy, tactile input or TCN. Same environment and scoring. Held of 150 at 5 cm, dev set:

| Grip or controller | Default | Noslip | Default: fell, slipped |
|---|---|---|---|
| Thumb, index, ring, pinky, full speed | 86 (57.3%) | 118 (78.7%) | 34, 30 |
| Same, slower ring and pinky (speeds read from seed 1) | 92 (61.3%) | 119 (79.3%) | 29, 29 |
| Thumb, index, ring, full speed | 76 (50.7%) | 119 (79.3%) | 40, 34 |
| All five fingers, full speed | 34 (22.7%) | 78 (52.0%) | 98, 18 |
| Slow 5-finger close, like the hand-tuned ramp | 40 (26.7%) | 71 (47.3%) | 85, 25 |
| RL seed 1 | 86 (57.3%) | 122 (81.3%) | 33, 31 |
| RL seed 2 | 56 (37.3%) | 101 (67.3%) | |
| RL seed 3 | 19 (12.7%) | 61 (40.7%) | |
| Hand-tuned A | 35 (23.3%) | 66 (44.0%) | |

- Paired with seed 1 at 5 cm, the first grip: 9 vs. 9 under default (p = 1.00) and 7 vs. 11 under noslip (p = 0.48).
- Closing the middle finger is what breaks a grip: all five fingers at full speed holds 34, against 86 without the middle finger.
- **Be careful:** choosing among grips on 150 trials inflates the best one, and seed 1's traces informed these grips.
- **On fresh seeds the first grip holds up (Figure 6b).** Under elliptic it held 801 of 1,005 at 5 cm (79.7%, 77.1-82.1) and 875 at 10 cm. Paired with RL seed 1: held only by the grip 53, only by seed 1 62, p = 0.46, a difference of +0.9 points for seed 1 (-1.1 to +3.0). It beats RL seed 2 (279 / 86), seed 3 (465 / 107) and hand-tuned A (349 / 27), all p < 0.0001. Nothing was chosen on these trials, so the dev-set optimism doesn't apply.

### R17. Contact solver settings (Figures 7, 10, 17)
**Cite:** Figures 7, 10, 17. Outside: [20] MuJoCo docs, Computation (the defaults); [48] MuJoCo docs, Modeling (its description of slip, quoted below); [49] Zakka 2022, MuJoCo Menagerie (other hands' options); [2] Prattichizzo & Trinkle 2016 where you say Coulomb friction allows no sliding.
- The scene sets no contact options, so it runs MuJoCo's defaults: pyramidal cone, `impratio` 1, no noslip pass [20].
- At full load under default settings, objects move although their contacts are far inside the friction limit: seed 1 on the dev set, median 1.10 mm/s with a median 4% of the friction limit used and capacity 29x the load (Figure 10a). Coulomb friction allows no sliding there. MuJoCo documents this as expected behaviour of its soft-contact model and calls slippage "especially problematic in manipulation tasks" [48].
- Other MuJoCo hand models set these options [49]: Allegro, Shadow Hand and Robotiq 2F-85 use `cone="elliptic" impratio="10"`, LEAP uses `impratio="100"`, and Shadow DEXEE uses `cone="elliptic" noslip_iterations="5"`.
- Same controllers, none retrained. Held of 150 at 5 cm (10 cm), dev set (Figure 7):

| Controller | Default | Noslip pass | Elliptic, `impratio` 10 |
|---|---|---|---|
| RL seed 1 | 57.3% (60.7%) | 81.3% (86.0%) | 78.7% (82.7%) |
| RL seed 2 | 37.3% (40.0%) | 67.3% (72.0%) | 58.0% (66.0%) |
| RL seed 3 | 12.7% (12.7%) | 40.7% (44.0%) | not run to completion on the dev set |
| Hand-tuned A | 23.3% (38.7%) | 44.0% (67.3%) | not run on the dev set |
| Fixed grip | 57.3% (60.7%) | 78.7% (87.3%) | not run on the dev set |

- **These dev rates understate the controllers.** 11 of the 12 dev figures checked against fresh seeds are below their fresh figure, by 0.6 to 7.9 points, and seeds 2 and 3 under noslip fall outside their fresh intervals (R22, Figure 18). The paired comparisons below still compare like with like, on a harder-than-average sample.
- Paired against default at 5 cm (held only under the stricter setting / only under default): noslip: seed 1 42 / 6, seed 2 56 / 11, seed 3 49 / 7, hand-tuned A 41 / 10, fixed grip 37 / 5, all p below 0.0001. Elliptic: seed 1 41 / 9, seed 2 43 / 12, both p below 0.0001. Noslip against elliptic, seed 1: 15 / 11, p = 0.56.
- **Where the gain comes from (Figure 17).** Mostly fewer objects falling while the grasp forms. Seed 1: falls 33 to 10 under elliptic (24 trials fall only under default, 1 only under elliptic), slips after forming 31 to 22. Seed 2: falls 52 to 32, slips 42 to 31. The same pattern holds under noslip for every controller.
- **Creep is reduced, not removed (Figure 10b).** On the same trials held at full load under both, seed 1's objects move at a median 1.19 mm/s under default and 0.99 under elliptic, slower on 46 of 87 trials (Wilcoxon p = 0.041). Under noslip: 1.17 and 0.69 mm/s, slower on 50 of 88 (p = 0.010). Friction use rises from 4% to about 28% of the limit, so the stricter settings carry the load through friction rather than letting it creep away.
- **What argues against over-reading this:** the settings were chosen after seeing scores, on a much-used dev set. The policies and TCN were trained on default physics. Pad friction doesn't predict holding under any setting (R19). Contacts at a slip reach a median 0.76 of their limit, so not every slip is sliding at the limit. And MuJoCo calls noslip "just an adhoc mechanism".
- **Decision:** elliptic with `impratio` 10 is primary, default is reported beside it, and noslip stays a sensitivity result. This was recorded in `results.md` before any fresh-seed result was read, along with what would reverse it. Say in the paper that the setting was proposed after scores had been seen.
- **Noslip against elliptic on fresh seeds (R22).** On the same 1,005 trials, noslip holds seed 2 11.9 points more and seed 3 4.5 more, while seed 1 and hand-tuned A don't differ. The two settings fail differently: noslip loses more objects while the grasp forms, and has far fewer slips afterwards. The Nov 4 rule kept elliptic primary whether or not noslip scored higher, so this changes nothing.

### R18. The training signal (Figures 12, 14)
**Cite:** Figures 12, 14. Outside: [62] Schulman 2016 where you explain discounted return.
- **The reward barely separates holding from dropping.** Seed 1, locked set, median total reward per trial: 612 when the object is held (n = 588), 617 when it slips after the grasp forms (n = 183), and 645 when it falls while the grasp forms (n = 189). Held against fell: Mann-Whitney p = 0.037, in the wrong direction. Held against slipped: p = 0.71.
- The pull phase does penalize failure (median -12 held, -51 slipped, -29 fell), but formation reward is about 20 times larger and slightly higher for trials that fail.
- The +20 hold bonus at the end of a trial is worth 0.20 at the start of a trial and 0.98 when the pull starts, after discounting at 0.99 per step.
- **Training reward ranks the seeds wrongly.** Final mean training reward: seed 1 448, seed 2 329, seed 3 361. Held on the locked set ranks them 1, 2, 3. Seed 3 earns more training reward than seed 2 while holding a third as often (Figure 14).
- **Be careful:** total reward per trial isn't exactly what PPO optimizes, which is discounted return from each state. But both point the same way: nothing in the reward makes holding clearly better than dropping.

### R19. Pad friction doesn't predict holding (Figure 13)
**Cite:** Figure 13.
- Within each object, trials split into fifths by pad friction. Under default settings on the locked set, held rates are flat across fifths for every controller. Mantel-Haenszel odds ratio for above vs. below each object's median friction: 0.90 (p = 0.56), 1.04 (p = 0.88), 1.00 (p = 0.92) for the three seeds, and 0.89 (p = 0.63) for hand-tuned A.
- Hand-tuned A under noslip on fresh seeds goes the other way: 53% at the lowest fifth to 42% at the highest, odds ratio 0.67 (p = 0.020), opposite to the prediction.
- Under elliptic on fresh seeds: 0.94 (p = 0.79), 0.86 (p = 0.31) and 1.55 (p = 0.011) for the three seeds, and 0.89 (p = 0.56) for hand-tuned A. Seed 3's is in the predicted direction.
- Under noslip on fresh seeds, the three seeds: 0.87 (p = 0.49), 1.16 (p = 0.37) and 1.06 (p = 0.74).
- That makes 12 tests, and none survives a Bonferroni correction [60]: the smallest corrected p is seed 3 under elliptic, 0.011 x 12 = 0.13, then hand-tuned A under noslip, 0.24. **Cite:** [58] Mantel & Haenszel 1959; [60] Dunn 1961.
- Likely reason: friction capacity is usually 20-50 times the load, and about half of the contact force goes through links and the palm, whose friction is fixed. So friction isn't the binding constraint in these grips.
- This was a pre-stated prediction that failed. Say so.

### R20. When failures happen (Figure 9)
**Cite:** Figure 9. Outside: [2] Prattichizzo & Trinkle 2016 for the Coulomb friction point.
- Under default settings, many slips happen at constant load. Seed 1, locked set, 771 trials in the hand when the pull starts: 67 slip while the pull ramps up, 73 in the first second at full load, 43 in the second. Seeds 2 and 3 show the same shape.
- Hand-tuned A, no drop rule: of 589 trials not slipped after 200 steps at full pull, 141 slip by 1,000 steps, with no change in load.
- Under Coulomb friction, a grip that holds a constant load for a second should keep holding it. This is the creep of R17 showing up as failures.
- Under elliptic on fresh seeds, slips during the ramp mostly disappear, but slips at constant load don't. Seed 1: of 922 trials in the hand when the pull starts, 21 slip during the ramp, 47 in the first second at full load and 44 in the second. Seed 2: 848; 54, 103, 83. Seed 3: 632; 47, 83, 59. For seed 2 the share slipping in the second full-load second rises, to 9.8% of trials in the hand against 5.7% under default. So elliptic removes most slips while the load rises, but slips at constant load remain, and for seed 2 they rise (Figure 9).
- Noslip cuts slips at constant load further. Slips at full load, as a share of trials in the hand when the pull starts, for seeds 1, 2 and 3: 6.6%, 8.5% and 11.7% under noslip, against 9.9%, 21.9% and 22.5% under elliptic, and 15.0%, 17.1% and 40.1% under default. They still don't disappear (Figure 9).

### R21. Agreement between controllers (Figure 16)
**Cite:** Figure 16.
- Locked set, default settings, 5 controllers (hand-tuned A and B, three RL seeds): 711 of 960 trials (74.1%) are held by at least one, 249 by none, and 15 by all five.
- Choosing the best of the five controllers for each object would hold 597 (62.2%). So under default settings no mix of these controllers reaches 70%.
- Under elliptic on fresh seeds (Figure 16b; hand-tuned A, three RL seeds and the fixed grip): 954 of 1,005 (94.9%) are held by at least one, 51 by none, and 161 by all five. Best controller per object: 876 (87.2%). For the four controllers run on both sets (hand-tuned A and the three seeds), trials held by none fall from 26.1% to 7.1%, and the best-per-object figure rises from 62.2% to 85.2%. Under noslip on fresh seeds, the same four hold none of 4.6% of trials, and the best per object gives 86.2%.
- The best-per-object figures are picked on the same trials, so they are optimistic ceilings, not results. Exploratory.

### R22. Fresh seeds (Figures 0, 5, 6, 8, 9, 13, 16, 17, 18)
**Cite:** Table 5 and the figures in the heading. Outside: [42] Nosek 2018 for the pre-registration.
- Pre-registered in `results.md`: the three RL seeds, hand-tuned A and the fixed grip on 1,005 fresh trials (seeds 2000-2066) under elliptic, plus the noslip runs the Nov 4 crash interrupted. Every run finished, elliptic and noslip, with no solver warnings. The noslip runs resumed from their own files after the Nov 4 crash. Scripts: `fresh_seed_report.py` and `elliptic_gate_check.py` in `src/analysis/nov3_drop_analysis/`.
- **Reversal check: passed, elliptic stays primary.** All four controllers' 95% lower bounds are above their locked default rates (78.0 vs. 61.2, 57.4 vs. 42.2, 41.0 vs. 15.0, 44.6 vs. 18.8). Three were needed.

| Fresh seeds, elliptic, 1,005 each (95% CI) | 5 cm | 10 cm | Fell, slipped |
|---|---|---|---|
| RL seed 1 (best of 3 seeds, deterministic) | 810, 80.6% (78.0-82.9) | 855, 85.1% (82.7-87.1) | 83, 112 |
| RL seed 2 | 608, 60.5% (57.4-63.5) | 651, 64.8% (61.8-67.7) | 157, 240 |
| RL seed 3 | 443, 44.1% (41.0-47.2) | 475, 47.3% (44.2-50.4) | 373, 189 |
| **Mean and SD across seeds** | **61.7, SD 18.3** | **65.7, SD 18.9** | |
| Hand-tuned A | 479, 47.7% (44.6-50.8) | 649, 64.6% (61.6-67.5) | 352, 174 |
| Fixed grip, no learning | 801, 79.7% (77.1-82.1) | 875, 87.1% (84.8-89.0) | 100, 104 |

- The seeds still differ far more than sampling allows: chi-square 284.6, p < 0.0001.
- Paired against hand-tuned A at 5 cm (held only by RL / only by hand-tuned): seed 1 355 / 24 and seed 2 273 / 144, both p < 0.0001. Seed 3 247 / 283, p = 0.13. Two seeds beat hand-tuned A and the third doesn't differ from it. Three-seed mean against hand-tuned A, not pre-registered, df 2: 14.1 points, p = 0.31.
- The fixed grip matches seed 1 (53 / 62, p = 0.46) and beats the rest (R16).
- **The 70% target:** seed 1 and the fixed grip clear it with their whole interval. Seeds 2 and 3 don't, and the mean is 61.7%. So the training procedure doesn't reach 70% reliably, and the seed that does is matched by a grip with no learning.
- **Noslip, the sensitivity setting, on the same fresh trials (complete Nov 7):**

| Fresh seeds, noslip, 1,005 each (95% CI) | 5 cm | 10 cm | Fell, slipped |
|---|---|---|---|
| RL seed 1 (best of 3 seeds, deterministic) | 823, 81.9% (79.4-84.1) | 877, 87.3% (85.1-89.2) | 107, 75 |
| RL seed 2 | 728, 72.4% (69.6-75.1) | 782, 77.8% (75.1-80.3) | 173, 104 |
| RL seed 3 | 488, 48.6% (45.5-51.6) | 504, 50.1% (47.1-53.2) | 416, 101 |
| **Mean and SD across seeds** | **67.6, SD 17.2** | **71.7, SD 19.3** | |
| Hand-tuned A | 468, 46.6% (43.5-49.7) | 707, 70.3% (67.5-73.1) | 432, 105 |

- The pattern is the same as under elliptic. The seeds differ (chi-square 271.0, p < 0.0001). Seeds 1 and 2 beat hand-tuned A (392 / 37 and 367 / 107, both p < 0.0001), and seed 3 doesn't differ from it (267 / 247, p = 0.40). Three-seed mean against hand-tuned A, not pre-registered, df 2: +21.1 points, p = 0.17. Only seed 1 clears 70% with its whole interval; seed 2's reaches down to 69.6%. So even the setting that scores highest doesn't make the training procedure reach 70% reliably.
- **Noslip against elliptic, same trials (not pre-registered).** Noslip minus elliptic at 5 cm: seed 1 +1.3 (-1.0 to +3.6, p = 0.31), seed 2 +11.9 (+8.8 to +15.0, p < 0.0001), seed 3 +4.5 (+1.4 to +7.6, p = 0.005), hand-tuned A -1.1 (-3.5 to +1.5, p = 0.45). Noslip loses more objects while the grasp forms (hand-tuned A: 432 falls against 352) and has far fewer slips afterwards (seed 2: 104 against 240). Elliptic stays primary under the rule set on Nov 4, which applied whether or not noslip scored higher.
- **Dev set against fresh (Figure 18), the pre-registered check:** of 12 dev figures (elliptic and noslip, 5 and 10 cm), 11 are below their fresh figure, by 0.6 to 7.9 points, and 6 fall outside the fresh interval. At 5 cm, seeds 2 and 3 under noslip are outside (67.3% against 69.6-75.1, and 40.7% against 45.5-51.6). At 10 cm, so are seeds 2 and 3 under noslip, and, just, seed 1 under elliptic and hand-tuned A under noslip. As pre-registered, say that these dev figures misstated the controllers: every one understated them. All 12 come from the same 150 trials, so this looks like a harder-than-average dev set, not optimism from choices made on it.

---

## 7. Discussion

**7.1 Success definitions can change the story**
- Identical trials ranged from 18.9% to 44.6%. Before this was measured, near-floor objects were counted as holds. **Cite:** Figure 1.
- Recommendation: benchmarks should confirm the object is still in the hand at the start of the measured phase, and should report sensitivity to cutoffs. This fits calls for explicit success criteria [39]. **Cite:** [39] Bekiroglu 2020, and [40] Calli 2017 for benchmark protocols.
- The same flaw reached the training labels. In a sample of 1,500 contact-only trials, 349 of 593 no-slip examples (59%) fell more than 15 cm. A label check is worth doing before training. **Cite:** none outside (your R4 numbers).

**7.2 Why better detection didn't help here**
- The outcome is set by the grip formed in the first half second. Most holds needed no correction, and removing the detector entirely changes almost nothing for either controller (R6, R15). **Cite:** Figures 3, 4 and 11. Give the intervals with "almost nothing", because a non-significant difference isn't proof of no effect: [67] Altman & Bland 1995.
- Grip strength isn't the constraint: fingertip forces reach 20-70 N on objects under 1.5 N.
- This matches the pattern in driving research that offline metrics aren't sufficient [37, 38], but the setting and sample here are limited. **Cite:** [37] Codevilla 2018; [38] Philion 2020.
- Practical point: a detector can only matter if the benchmark contains situations a fixed grip can't handle, and the controller has a reason and the authority to act on it. Evaluate detectors in closed loop, against a fixed-grip baseline. **Cite:** [51] James & Lepora 2021, whose slip detection helped when added weight destabilized the grasp: the kind of situation a fixed grip can't handle.

**7.3 RL: a good grip, found unreliably**
- On 960 locked trials at 5 cm, the same setup gave 61.2%, 42.2% and 15.0%. Two policies clearly beat hand-tuned, and one is worse than it. Under elliptic on fresh trials: 80.6%, 60.5% and 44.1%. Again two beat hand-tuned, and the third doesn't differ from it (R22). Under noslip, the sensitivity setting: 81.9%, 72.4% and 48.6%, the same pattern. **Cite:** Figure 5, Table 5.
- The best policy's advantage over hand-tuned control is matched by a fixed grip that closes four fingers fast and leaves the middle finger open, on the dev set and on fresh trials (79.7% against 80.6%, R16). The seeds differ mainly in which fingers they close (R11). So what was learned is a grip, not a feedback strategy. **Cite:** Figures 6 and 11.
- The reward explains part of this. It pays mostly for forming a grasp, and a trial whose object falls earns at least as much as one that is held (R18). Training reward ranked the seeds 1, 3, 2 while holding ranked them 1, 2, 3, so training reward can't be used to pick a policy here. **Cite:** Figures 12 and 14.
- All of these rates use the mean action, and sampled actions do 7-14 points worse (R13).
- The spread between repeats is the main obstacle. This agrees with the wider literature on RL variance [33, 34, 35, 36]. Don't call the size of this spread normal: Henderson et al. may describe PPO as relatively stable, so check [33]. **Cite:** [33] Henderson 2018; [34] Agarwal 2021; [35] Andrychowicz 2021; [36] Engstrom 2020.
- Reward, observation and rollout changes made single runs worse (R12, R14), but with this spread single runs can't be attributed to the changes.

**7.4 Where the limits are**
- Under default settings, some shapes rarely hold under any controller, and the controllers fail on largely the same trials (R21). Every object uses the same hand pose and closing motion. **Cite:** Figures 8 and 16.
- Under default settings, a quarter of trials are held by none of five controllers, and the best controller per object gives 62.2%. **Cite:** Figure 16.
- Much of that limit is the contact model. For the four controllers run under both settings, trials held by none fall from 26.1% under default to 7.1% under elliptic, and the best controller per object rises from 62.2% to 85.2% (different trials, so unpaired; best per object is optimistic, since it's picked on the same trials). Under elliptic, seed 1 and the fixed grip reach 70% on 10 of 15 objects (R10, R21). **Cite:** Figures 8 and 16.
- For a 70% overall rate, if four shapes stayed near zero, the other 11 would need about 95%. Under elliptic no shape stays near zero for the best controllers: seed 1's lowest is rectangular prism, at 29 of 67.

**7.5 Lessons for simulated grasp studies**
- Test length, success rule, checkpoint identity, contact settings and training repeats each changed results here. **Cite:** Figure 0; for the general point, [33] Henderson 2018 (training repeats) and [39] Bekiroglu 2020 (success rules).
- Record hashes and configuration with every result. That is how the checkpoint mismatch was caught.
- Archive the scene file. Here it was outside version control.

**7.6 Simulator contact settings are part of the result**
- Under MuJoCo's defaults, objects creep with contacts at 4% of their friction limit (R17, R20). **Cite:** Figure 10; [48] MuJoCo docs, Modeling.
- Changing two solver options moved every controller: about +21 points paired on the dev set, and 18 to 29 points from the locked set under default to fresh trials under elliptic. That is far more than any detector difference, but less than the gap between RL seeds or between controllers (Figure 0). **Cite:** Figures 0 and 7.
- For three of four controllers most of the change happens while the grasp forms, not during the pull. Creep at full load is reduced, not removed, and objects still slip at constant full load under elliptic (R20). **Cite:** Figures 9 and 17.
- The two stricter settings help in different places. Elliptic keeps more objects in the hand while the grasp forms; noslip stops more slips during the pull. On the same fresh trials, noslip holds seed 2 11.9 points more than elliptic, while seed 1 and hand-tuned A don't change (R22). So the choice between stricter settings also changes which controllers look best. **Cite:** Figures 5 and 9.
- A simulated grasping benchmark should report its contact settings and how sensitive its results are to them. Say plainly that this setting was chosen after seeing scores, that default is reported beside it everywhere, and that it was not tuned further. **Cite:** [50] Le Lidec 2024; [48] MuJoCo docs, Modeling.

---

## 8. Limitations (state all of these)

**Cite:** mostly none; these are your caveats. Where a limitation is a known issue, the reference is given.

- Simulation only, with no hardware validation.
- Contacts are soft in MuJoCo [19, 20]. Under default settings objects creep at about 1 mm/s at full load with contacts at 4% of their friction limit. Under the elliptic setting used for the primary results they still creep at about 1 mm/s (R17). It reduces failures but doesn't make contacts stick. **Cite:** [19] Todorov 2014; [20] MuJoCo docs, Computation; [48] MuJoCo docs, Modeling.
- The elliptic setting was chosen after dev-set scores had been seen.
- Policies and the TCN were trained under default contact settings and are evaluated under elliptic. No policy was trained under elliptic.
- The default and elliptic headline numbers come from different trials (locked seeds 3-66 and fresh seeds 2000-2066), so they are compared unpaired. The paired comparison exists only on the 150-trial dev set, whose figures ran 2-3 points below the fresh ones (R22).
- The pull is downward only, at 1.5x weight. Real disturbances vary.
- Same hand pose and object placement for all objects, with no grasp planning or arm.
- Detector data uses a random split by trial, so static scores don't test generalization to new objects. **Cite:** [41] Kapoor & Narayanan 2023.
- Training labels include near-floor no-slip examples. They were not relabelled.
- The channels named normal and tangential aren't normal and tangential force (Figure 15). Model B's grip-force feature and the RL grip-force reward use the same channel.
- The TCN trained on data recorded with a sensor noise setting of 0.05, but a search finds no noise step in `closed_loop_sim.py` or `rl_env.py`, so training and closed-loop conditions differ. Ramp length also varied in training (400-600) and is fixed at 500 in closed loop.
- Only three RL repeats, not explicitly seeded, with different numbers of process restarts. **Cite:** [33] Henderson 2018; [34] Agarwal 2021.
- RL rates are deterministic, using the policy's mean action, which isn't the sampled behaviour seen during training. Sampled actions held 7-14 points fewer on the dev set (R13).
- Static F1 values for both models are single, unseeded runs unless the seeded runs get done.
- The A vs. B comparison has only two checkpoints that differ in several ways.
- The 10 cm rule was chosen after seeing results.
- RL policies were trained with a 5 cm rule, so 10 cm scoring doesn't show what a policy trained under 10 cm could do.
- The 5-7 cm "gripped" evidence comes from training data, not the benchmark.
- Dev-set results (150 trials, looked at many times) are exploratory. They also understated the controllers: 11 of 12 dev figures checked are below their fresh figure, by up to 7.9 points (Figure 18).
- No RL + Model B result exists. There is also a known timing issue in `rl_env.py` for 25-channel models.

---

## 9. Future work (Phase 2)

1. **A benchmark where a fixed grip fails:** heavier or changing pulls, mid-grasp disturbances. Until then, no detector can show it helps. **Cite:** [51] James & Lepora 2021 (added weight to destabilize grasps); [39] Bekiroglu 2020.
2. **Retrain under the primary setting:** RL repeats and TCN data regenerated under elliptic.
3. **A reward that pays for holding:** so that holding clearly beats dropping, then check that training reward ranks policies by holding.
4. **Fix the sensor channels:** touch for normal force and direct contact forces for tangential, then regenerate the data. **Cite:** Figure 15; [69] MuJoCo docs (`mj_contactForce`).
5. **Measure the ceiling:** a best-case controller shows what share of trials can physically be held. This tells us whether mid-to-high 90s is possible.
6. **Fix labels:** exclude or relabel near-floor examples, verify the object is in hand, and retrain.
7. **Seeded multi-run training:** mean and SD for both models. **Cite:** [33] Henderson 2018; [34] Agarwal 2021.
8. **True physics-informed TCN:** add a physics loss [25, 26], such as a friction-cone constraint [2], and test it in closed loop. Do it after items 2 and 4: under default settings, slips happen at a median 0.56 of the friction limit, so a friction-cone loss would fight its own labels. **Cite:** [25] Raissi 2019; [26] Karniadakis 2021; [2] Prattichizzo & Trinkle 2016.
9. **Better grip for failing shapes:** hand pose, object placement, adaptive formation. Treat falling and slipping as separate problems. **Cite:** [3] Bohg 2014.
10. **More stable RL:** more repeats, and residual control on top of the fixed grip [46, 47], so every repeat starts from a working grip and the learned part has to add something. **Cite:** [46] Johannink 2019; [47] Silver 2018; [34] Agarwal 2021.
11. **RL with each detector:** train with and without the slip signal, and compare detectors under RL. **Cite:** [52] Koenig 2022; [54] Zhang 2025, for how others ablated touch.
12. **Generalization:** leave-object-out splits, which `export_data.py` already supports. **Cite:** [41] Kapoor & Narayanan 2023.
13. **Toward hardware:** domain randomization [16, 21], then the physical ORCA hand [17]. **Cite:** [16] OpenAI 2020; [21] Tobin 2017; [17] Christoph 2025.

---

## 10. Conclusion

**Cite:** nothing new; restate your results. At most repeat [39] Bekiroglu 2020 with the success-criteria take-away.

1. Restate the question.
2. Success definition alone moved identical trials between 18.9% and 44.6%.
3. A large static F1 gap produced no closed-loop difference, and removing the detector changed almost nothing, because the grip formed in the first half second decides the outcome.
4. Three PPO training repeats held 80.6%, 60.5% and 44.1% at 5 cm under the primary contact settings (mean 61.7%, SD 18.3) against 47.7% for hand-tuned control, and 61.2%, 42.2% and 15.0% under MuJoCo's defaults. A fixed grip with no learning matched the best repeat on 1,005 fresh trials. What was learned is a grip.
5. The simulator's contact settings moved every controller by 18 to 29 points, far more than any detector difference and more than learning added over a fixed grip.
6. Take-away: verify what success counts, report contact settings, evaluate detectors in closed loop against a fixed-grip baseline, and report all training repeats.
7. Next step: a benchmark where a fixed grip fails, then the true physics-informed model.

---

## 11. Figures and tables

All figures are in `reports/figures/paper/` as PNG (300 dpi) and PDF, each with a CSV of the plotted numbers. `README.md` there has a draft caption for each. Rebuild with `venv/Scripts/python.exe src/analysis/paper_figures.py`.

| Figure | File | Shows | Status |
|---|---|---|---|
| 0 | `fig00_factor_summary` | Each factor's effect on the held rate, same trials on both sides, paired intervals; fresh elliptic estimates on four rows | Ready |
| 1 | `fig01_success_definition` | Held rate against the formation-fall cutoff, hand-tuned and RL | Ready |
| 2 | `fig02_formation_falls` | Distribution of formation falls, 5 and 10 cm marked | Ready |
| 3 | `fig03_detector_vs_closed_loop` | Model A and B, same trials, every scoring rule | Ready |
| 4 | `fig04_tcn_removal` | Every controller with and without the TCN | Ready |
| 5 | `fig05_rl_seed_spread` | Each RL seed, mean and SD, hand-tuned; default (locked), elliptic and noslip (fresh) | Ready |
| 6 | `fig06_fixed_grips` | Fixed grips against the controllers: dev set, then the fresh-seed grip paired with each controller | Ready |
| 7 | `fig07_contact_settings_dev` | Each controller under each contact setting, same trials | Ready |
| 8 | `fig08_per_object` | Held by object and controller, locked default and fresh elliptic | Ready |
| 9 | `fig09_slip_timing` | Share of in-hand trials not yet slipped over the pull, under each setting | Ready |
| 10 | `fig10_creep_at_full_load` | Object speed against friction used; same trials across settings | Ready |
| 11 | `fig11_finger_closure` | Each seed's median finger closure over time | Ready |
| 12 | `fig12_reward_structure` | Reward for held, slipped and fallen trials; discounted hold bonus | Ready |
| 13 | `fig13_pad_friction` | Held rate by pad-friction fifth, under each setting | Ready |
| 14 | `fig14_training_curves` | Training reward and episode length per seed | Ready |
| 15 | `fig15_sensor_channel` | The "normal" channel against the touch sensor | Ready |
| 16 | `fig16_controller_agreement` | How many controllers hold each trial: locked default, then fresh elliptic | Ready |
| 17 | `fig17_outcome_breakdown` | Held, slipped and fell, per controller and setting | Ready |
| 18 | `fig18_dev_vs_fresh` | Each dev-set figure minus its fresh-seed figure, with the fresh interval; 12 comparisons | Ready |
| 19 | Static F1 with spread across seeds | | `[PENDING: your seeded runs]` |
| Optional | Diagram of the pipeline and trial phases | | To make |

Suggested main-text set for a short paper: Figure 0 as the opening figure, then Figures 1, 3, 4, 5, 6, 7 and 17, with the rest as supplementary. Renumber when you assemble the paper.

| Table | Content | Status |
|---|---|---|
| Table 1 | Objects: kind, size, mass, slip limit | To make from `generate_trials.py` |
| Table 2 | Static metrics (R1) | Ready |
| Table 3 | Hand-tuned under each rule (R5) | Ready |
| Table 4 | Same 45 trials (R8) | Ready |
| Table 5 | RL locked set under default (R9) and fresh seeds under elliptic (R22) | Ready |
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

[17] C. C. Christoph et al., "ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning," in *Proc. IEEE/RSJ IROS*, 2025, pp. 8503-8510. doi:10.1109/IROS60139.2025.11246681. arXiv:2504.04259. Proceedings version found on Crossref, Nov 7. **[A]**

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

[48] MuJoCo documentation, "Modeling" chapter, "Solver settings" and "Preventing slip". https://mujoco.readthedocs.io/en/stable/modeling.html. Read in full on Nov 4, 2026; quotes in `results.md`. Its "Contact parameters" section, checked Nov 7, gives the friction rule cited in 5.2. Cite the version you used (3.11.0).

[49] K. Zakka, Y. Tassa, and MuJoCo Menagerie Contributors, "MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo," 2022. http://github.com/google-deepmind/mujoco_menagerie. This is the citation the repository's README asks for (checked Nov 7). Model files read on Nov 4, 2026 (`<option>` lines): `wonik_allegro/right_hand.xml`, `shadow_hand/right_hand.xml`, `robotiq_2f85/2f85.xml`, `leap_hand/right_hand.xml` and `shadow_dexee/shadow_dexee.xml`. Add the commit you read if your citation style allows.

**Added Nov 6 from a web search; confirmed Nov 7 against Crossref and arXiv, abstracts read**

[50] Q. Le Lidec, W. Jallet, L. Montaut, I. Laptev, C. Schmid and J. Carpentier, "Contact Models in Robotics: a Comparative Analysis," *IEEE Transactions on Robotics*, vol. 40, pp. 3716-3733, 2024. doi:10.1109/TRO.2024.3434208. arXiv:2304.06372. Surveys the contact models in robotics simulators, sets out each engine's physical relaxations, and benchmarks them; finds that common approximations can severely widen the reality gap. **[A]**

[51] J. W. James and N. F. Lepora, "Slip Detection for Grasp Stabilization With a Multifingered Tactile Robot Hand," *IEEE Trans. Robotics*, vol. 37, no. 2, pp. 506-519, 2021. doi:10.1109/TRO.2020.3031245. arXiv:2010.01928. Slip detection with support vector machines on the Tactile Model O hand, reacting in real time to stabilize grasps, including when weight is added. **[A]**

[52] A. Koenig, Z. Liu, L. Janson, and R. Howe, "The Role of Tactile Sensing in Learning and Deploying Grasp Refinement Algorithms," in *Proc. IEEE/RSJ IROS*, 2022, pp. 7766-7772. doi:10.1109/IROS47612.2022.9981915. arXiv:2109.11234. In simulation, with tactile information in the reward, leaving touch out of the policy's state cost at most 6.6%. **[A]**

[53] L. Pecyna, S. Dong, and S. Luo, "Visual-Tactile Multimodality for Following Deformable Linear Objects Using Reinforcement Learning," in *Proc. IEEE/RSJ IROS*, 2022. doi:10.1109/IROS47612.2022.9982218. arXiv:2204.00117. Vision plus touch completed the task in up to 92% of cases, against 77% with one signal. Corrected Nov 7: an earlier note said removing touch at test time had practically no effect on reward. The abstract doesn't say that and its headline points the other way, so don't cite it for that unless you find it in the paper. **[A]**

[54] B. Zhang, I. Andrussow, A. Zell, and G. Martius, "The Role of Tactile Sensing for Learning Reach and Grasp," in *Proc. IEEE ICRA*, 2025, pp. 11817-11824. doi:10.1109/ICRA55743.2025.11127409. arXiv:2502.20367. Under imperfect visual perception, tactile features improved RL grasping; complex tactile inputs made training harder. **[A]**

**Added Nov 7: statistics, software and documentation (checked by identifier on Crossref or arXiv, not read)**

[55] K. Pearson, "On the criterion that a given system of deviations from the probable in the case of a correlated system of variables is such that it can be reasonably supposed to have arisen from random sampling," *Philosophical Magazine*, ser. 5, vol. 50, no. 302, pp. 157-175, 1900. doi:10.1080/14786440009463897. **[M]**

[56] H. B. Mann and D. R. Whitney, "On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other," *Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50-60, 1947. doi:10.1214/aoms/1177730491. **[M]**

[57] F. Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80-83, 1945. doi:10.2307/3001968. **[M]**

[58] N. Mantel and W. Haenszel, "Statistical Aspects of the Analysis of Data From Retrospective Studies of Disease," *J. National Cancer Institute*, vol. 22, no. 4, pp. 719-748, 1959. doi:10.1093/jnci/22.4.719. Crossref's record has the title, journal and year but no authors or pages, so confirm those on the journal page. **[M]**

[59] B. Efron, "Bootstrap Methods: Another Look at the Jackknife," *Annals of Statistics*, vol. 7, no. 1, pp. 1-26, 1979. doi:10.1214/aos/1176344552. **[M]**

[60] O. J. Dunn, "Multiple Comparisons among Means," *J. American Statistical Association*, vol. 56, no. 293, pp. 52-64, 1961. doi:10.1080/01621459.1961.10482090. The usual reference for the Bonferroni correction. **[M]**

[61] Student, "The Probable Error of a Mean," *Biometrika*, vol. 6, no. 1, pp. 1-25, 1908. doi:10.2307/2331554. **[M]**

[62] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2015. Published at ICLR 2016; the arXiv record doesn't name the venue, so confirm it. **[M]**

[63] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE CVPR*, 2016, pp. 770-778. doi:10.1109/CVPR.2016.90. **[M]**

[64] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in *Proc. NeurIPS*, 2019. arXiv:1912.01703. **[M]**

[65] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261-272, 2020. doi:10.1038/s41592-019-0686-2. **[M]**

[66] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90-95, 2007. doi:10.1109/MCSE.2007.55. **[M]**

[67] D. G. Altman and J. M. Bland, "Absence of evidence is not evidence of absence," *BMJ*, vol. 311, no. 7003, p. 485, 1995. doi:10.1136/bmj.311.7003.485. **[M]**

[68] R. L. Wasserstein and N. A. Lazar, "The ASA Statement on p-Values: Context, Process, and Purpose," *The American Statistician*, vol. 70, no. 2, pp. 129-133, 2016. doi:10.1080/00031305.2016.1154108. **[M]**

[69] MuJoCo documentation: the XML reference for the `touch`, `force` and `torque` sensors, and the API reference for `mj_contactForce`. https://mujoco.readthedocs.io. Not read for this paper yet: find the exact pages and cite the version you used (3.11.0).
