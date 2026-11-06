# Paper figures

Built by `src/analysis/paper_figures.py` from stored trial rows. Each figure has a PNG (300 dpi), a PDF and a CSV of the plotted numbers. `stats_summary.md` has every number and test behind them. Rebuild from the repo root:

    venv/Scripts/python.exe src/analysis/paper_figures.py

The script checks each input file's trial count and held count against `src/training/results.md`, and leaves out any run that is missing or unfinished rather than estimating it. Fresh-seed elliptic series appear once those runs finish.

**Conventions.** Color is the contact setting: blue elliptic (primary), orange MuJoCo default, aqua noslip, each with its own marker shape. Intervals are 95% Wilson. Paired comparisons are exact McNemar tests on the same trials. Locked set: seeds 3-66, 960 trials. Dev set: seeds 1000-1009, 150 trials, exploratory. Fresh seeds: 2000-2066, 1,005 trials.

Draft captions below. Rewrite them in your own words.

| # | File | Draft caption |
|---|---|---|
| 1 | `fig01_success_definition` | Share of trials scored as held against the largest fall allowed while the grasp forms. (a) Hand-tuned controller with Model A and Model B, 1,005 trials; the lines overlap. (b) Three RL training seeds, 960 locked trials. Squares mark 5, 10 and 15 cm; hollow squares, no drop rule. The same trials score 18.9% to 44.6% depending on the rule. |
| 2 | `fig02_formation_falls` | Largest fall during grasp formation, log scale. Most falls past 5 cm are under 10 cm, where objects often still appear gripped; a second cluster near 20-25 cm is objects on the floor. |
| 3 | `fig03_detector_vs_closed_loop` | Hand-tuned control with two slip detectors, static F1 0.691 (filled) and 0.799 (hollow), on the same 1,005 trials under each scoring rule. Right: held only with A, only with B, and exact McNemar p. No rule separates them. |
| 4 | `fig04_tcn_removal` | Each controller with the TCN (filled) and without it (hollow) on the same 150 dev trials. For RL the slip probability is set to 0; for hand-tuned control the TCN is removed. 19 of 1,200 outcomes change, none significantly. |
| 5 | `fig05_rl_seed_spread` | Held at 5 cm for each RL training seed, their mean (± 1 SD across seeds) and hand-tuned control. Default settings on the 960 locked trials; elliptic on 1,005 fresh trials. Right: rate, interval and count. |
| 6 | `fig06_fixed_grips` | Fixed grips with no policy, tactile input or detector, against the controllers, dev set. Closing thumb, index, ring and pinky fast and leaving the middle finger open matches the best RL seed under both settings. When the fresh-seed run lands, panel b repeats the comparison on 1,005 fresh trials under elliptic. |
| 7 | `fig07_contact_settings_dev` | The same controllers, none retrained, under three contact settings on the same 150 dev trials. Every controller gains 20-30 points under the stricter settings. |
| 8 | `fig08_per_object` | Percent held at 5 cm by object and controller. Objects sorted by mean under default settings. Elliptic columns appear when the fresh-seed runs finish. |
| 9 | `fig09_slip_timing` | Share of trials still in the hand that haven't slipped, over the pull. Gray: the pull ramping up; after 1 s the load is constant at 1.5x the object's weight. Under default settings, slips continue at constant load. |
| 10 | `fig10_creep_at_full_load` | (a) Object speed at full load against the share of the friction limit in use, RL seed 1, dev set; points are trials with the object in the hand. Objects move about 1 mm/s with contacts far inside the limit under every setting. (b) The same trials under default (x) and a stricter setting (y); the stricter settings slow creep only slightly. |
| 11 | `fig11_finger_closure` | Median closure of each finger over time for the three RL seeds, dev set. Seed 1 never closes the middle finger; seed 3 grips with thumb, index and pinky only. Each seed settles on its grip in the first half second. |
| 12 | `fig12_reward_structure` | (a) Total reward per trial for RL seed 1 on the locked set, by outcome. (b) The pull-phase part. (c) What the +20 hold bonus is worth after discounting, by when the policy is acting. Trials whose object falls earn at least as much reward as trials where it is held. |
| 13 | `fig13_pad_friction` | Held at 5 cm by pad-friction fifth within each object. Holding doesn't rise with friction under any setting tested. |
| 14 | `fig14_training_curves` | Mean training reward and episode length for the three seeds. Seed 3 ends with more training reward than seed 2 but holds a third as often. |
| 15 | `fig15_sensor_channel` | The channel named "normal" against the touch sensor's actual normal force, fingertip samples from RL seed 1's dev trials. It reads a median 18% of the true value. |
| 16 | `fig16_controller_agreement` | Number of the five controllers (hand-tuned A and B, RL seeds 1-3) that hold each locked-set trial. 249 trials are held by none. |
| 17 | `fig17_outcome_breakdown` | Each trial as held, slipped after the grasp formed, or fell while it formed, per controller and contact setting. Most of the gain under the stricter settings comes from fewer falls during grasp formation. |
