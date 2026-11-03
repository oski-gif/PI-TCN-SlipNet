# Overnight log: night of Nov 2-3, 2026

Status: **stopped early. Final as of 6:15am, Nov 3.** Queue item 1 and the F1 check finished. Items 2-5 did not run, so no RL job ran last night.

## Summary

- **The run stopped early.** The worker hit the 5-hour usage limit shortly after starting Model B, around 12:40am. Model B finished on its own at 1:23am. My own 12:41am check never scheduled another one, so nothing restarted after the limit reset at 4:40am. You asked me to wait and keep going, and that didn't happen. Nothing ran from 1:23am to 6am.
- **Hand-tuned scores were too high.** At RL's test length, Model A goes from 44.6% to 31.7% once trials where the object fell more than 15 cm stop counting as holds. Under RL's 5 cm rule it is 18.9%. Model B is 44.7%, 31.8% and 19.3%. The old 58.6% and 57.9% came from a shorter test with no drop rule.
- **The 5 cm rule needs your decision.** It sits in the densest part of the fall distribution. Moving the cutoff from 5 cm to 6 cm takes Model A from 18.9% to 30.0%, and any cutoff from 7 to 20 cm gives 31.4-31.7%. In recorded training trials, objects that sank 5-7 cm look gripped. RL ends a trial as soon as it crosses 5 cm, so RL's drop failures may include real grips.
- **RL vs. hand-tuned, first fair comparison.** On the same 45 trials and the same 5 cm rule, hand-tuned scores 22.2% (A) and 20.0% (B). RL's three seeds score 64.4%, 35.6% and 17.8%, mean 39.3%. RL's mean is higher, but one seed is below hand-tuned and this is only 45 trials.
- **Model A and Model B are still tied in closed loop.** 190 vs. 194 of 1,005 at 5 cm, 319 vs. 320 at 15 cm, even though their static F1 is 0.691 vs. 0.799.
- **`model_a.pt` measures F1 0.691, not 0.773.** Every Model A closed-loop result and every RL policy uses that file. The 0.773 model is an older file in git, and it also reproduced when measured.
- **The drop problem does not appear to be inflating F1.** F1 is higher on trials without a drop, for all four checkpoint and threshold rows.
- **Nothing was committed, pushed, retrained or relabelled, and no checkpoint was replaced.** The figures were not re-run, because they depend on the cutoff decision.

## Decisions waiting on you

1. **Which drop rule should the paper use?** This changes every closed-loop number, and it would be a scoring change made after seeing results, so the paper has to say so.
   - **My recommendation:** report both 5 cm and 10 cm. 10 cm sits in the gap in the data, and for hand-tuned any cutoff from 7 to 20 cm lands within 3 trials of 1,005. Add an evaluation-only option so RL trials run on past 5 cm and record the maximum fall, then re-score the existing policies under both rules. That needs no retraining.
   - **Limit:** these policies were trained to avoid crossing 5 cm, so re-scoring may not show what a policy trained under a 10 cm rule could do. Retraining would be a separate decision.
2. **Which Model A does the paper cite?** `model_a.pt` measures F1 0.691. The 0.773 file is at commit 9d4d079, with a copy at `Data/model_a_git9d4d079_dbb50c5a.pt`.
   - **My recommendation:** keep `model_a.pt`, because all RL results depend on it, and label closed-loop Model A results as F1 0.691. Don't cite 0.773 next to them. I won't replace it.
   - **Optional:** a hand-tuned run with the 0.773 file takes about 45 minutes and would show whether that F1 gap matters in closed loop. It did not run last night.
3. **Fix the training data before Nov 5?** In a sample of 1,500 trials, 349 of 593 no-slip examples (59%) fell more than 15 cm, so they are near floor level. That is lower than the 75% I first reported, which also counted objects that sank just past 5 cm.
   - **My recommendation:** don't relabel or retrain before Nov 5. Disclose it in the paper and fix it in Phase 2. The F1 check lowers the urgency, but it does not show how much these examples affect behaviour in closed loop.
4. **Commit and push?** Nothing from last night is committed. I'd do it after decision 1.

Still open from before: 40 or 20 epochs (I recommended 40, and both models must match), and F1 step 1 in `train.py`, which is yours to type.

## Numbers you can use

Each cell is the share of trials held under that scoring. "No drop rule" is the old scoring. The two right columns also fail trials where the object fell more than 15 cm or more than 5 cm before the pull. RL uses 5 cm.

| Condition | Trials | Total steps | No drop rule | 15 cm rule | 5 cm rule |
|---|---|---|---|---|---|
| Hand-tuned A (F1 0.691 file) | 1,005 | 2300 | 44.6% (448) | 31.7% (319) | 18.9% (190) |
| Hand-tuned B (F1 0.799 file) | 1,005 | 2300 | 44.7% (449) | 31.8% (320) | 19.3% (194) |
| Hand-tuned A, seeds 3-66 | 960 | 2300 | 44.8% (430) | 31.8% (305) | 18.8% (180) |
| Hand-tuned B, seeds 3-66 | 960 | 2300 | 45.1% (433) | 32.1% (308) | 19.3% (185) |
| Hand-tuned A, seeds 0-2 | 45 | 2300 | 40.0% (18) | 31.1% (14) | 22.2% (10) |
| Hand-tuned B, seeds 0-2 | 45 | 2300 | 35.6% (16) | 26.7% (12) | 20.0% (9) |
| RL, 3 seeds, seeds 0-2 | 45 each | 2300 | Can't be scored | Can't be scored | 64.4%, 35.6%, 17.8% (mean 39.3%) |
| Hand-tuned A, older run | 1,005 | 1500 | 58.6% (589) | 45.5% (457) | 26.6% (267) |
| Hand-tuned B, older run | 1,005 | 1500 | 57.9% (582) | 44.8% (450) | 25.9% (260) |
| Hand-tuned A, older run | 45 | 2500 | 40.0% (18) | 31.1% (14) | 22.2% (10) |
| Hand-tuned B, older run | 45 | 2500 | 33.3% (15) | 24.4% (11) | 17.8% (8) |

- **Only compare rows with the same total steps.** The 64.4% is the best of 3 seeds, so it isn't the headline.
- **RL can't be re-scored yet,** because its trials end at the 5 cm crossing.
- **Older runs were corrected using falls recorded in the 2300-step runs.** That is valid because formation doesn't depend on the model or test length: the drop flags match between A and B on all 1,005 trials, and Model A's shorter and longer runs are consistent on every trial.

**Static F1, whole test split (2,628 trials), split at the 5 cm rule:**

| Checkpoint | Threshold | All trials | No drop (1,224) | Drop (1,404) |
|---|---|---|---|---|
| `ce813826`, current `model_a.pt` | argmax | 0.691 | 0.789 | 0.559 |
| `dbb50c5a`, older Model A in git | argmax | 0.773 | 0.868 | 0.660 |
| `948a2716`, `model_b.pt` | 0.50 | 0.772 | 0.868 | 0.638 |
| `948a2716`, `model_b.pt` | 0.30 | 0.799 | 0.879 | 0.703 |

These are pooled per timestep. The slip base rate differs between subsets (26.5% vs. 19.3%), so the F1 gap isn't purely a model effect. At argmax or 0.50, older Model A and Model B are effectively tied. Model B's 0.799 needs its tuned threshold, and it also had 40 epochs against Model A's 20. Precision, recall and specificity are in `Data/_tcn_f1_subsets.log` and `reports/tcn_f1_without_drops.json`.

## Log

Newest entries go at the bottom. Each entry gives the time, what ran, the result, and what changed.

- **11:42pm:** Both agents had hit the usage limit, which reset at 11:40pm. Model B's 1,005-trial run had finished at 7:51pm. On the same trials, 14 were held only by Model A and 7 only by Model B. Most holds had no corrections at all (511 of 589 for A, 373 of 582 for B), so this short test mostly measures the initial grip, not the controller.
- **Before midnight:** Figure scripts corrected and re-run (`src/analysis/make_figures.py`, output in `reports/figures/`, not committed). Figures now group results by test length, and the RL seed-spread chart has no baseline line until the 2300-step runs exist. Paths for those future runs are placeholders. The RL eval logs don't record `--max-steps`, so RL's 1500 is taken from the script default rather than read from a log.
- **11:55pm, preliminary:** Possible major data problem, still being checked. A read-only scan of 1,000 random trials from the contact-only training population found the object fell more than 5 cm during grip formation in 297 of 398 no-slip trials (74.6%, median fall 21.5 cm) and 245 of 602 slip trials (40.7%). If that holds up, most of what the TCN learned as "no slip" is an object that has already dropped. The new 2300-step run shows the same pattern in scoring: its first trial (cube, seed 0) counts as a hold with 0.59 cm of movement after falling 22.7 cm during formation. Hand-tuned scoring has no drop rule, so the old hand-tuned percentages may be inflated. RL's are not affected this way. This is one scan and one trial, not a final result.
- **11:53pm:** Drop recording added to the hand-tuned test and independently reviewed; 5 known trials reproduce exactly, so it changes nothing else. Hand-tuned Model A at RL's test length (2300 steps, 1,005 trials) is now running, Model B follows. No rates from these yet.
- **11:53pm, surprise:** Some trials scored "held" are objects lying near the floor. `cube` seed 0 fell 0.23 m from a 0.245 m start and still counted as held, because displacement is only measured from the end of settle. This explains the old `cube` seed 0 puzzle, and it means all hand-tuned rates so far are too high by an amount the current runs will measure. The TCN's training data has the same issue: about 75% of sampled "no slip" trials had already dropped. This is now under "Decisions waiting on you". Nothing was retrained or relabelled.
- **After midnight, RL failures by type:** Formation drops are a large share of RL's failures in all three seeds, and the largest share in two of them. These counts come from the existing eval records, so nothing new was run. RL's policy does act during formation, so these drops are not automatically out of its reach.

  | RL seed | Held | Slipped | Dropped in formation | Drops as share of failures |
  |---|---|---|---|---|
  | Seed 1 (64.4%) | 29/45 | 5 | 11 | 11 of 16 |
  | Seed 2 (35.6%) | 16/45 | 15 | 14 | 14 of 29 |
  | Seed 3 (17.8%) | 8/45 | 15 | 22 | 22 of 37 |
  | All three | 53/135 | 35 | 47 | 47 of 82 (57%) |

- **After midnight, approved:** I told the worker to run the read-only F1 check on trials without a drop. It isn't system-altering: it uses the existing checkpoints and changes no code, data or models. It must first reproduce the recorded F1 scores, and it will report precision, recall and specificity alongside F1, because removing easy no-slip trials can lower F1 on its own. Relabelling, regenerating data and retraining still wait for you.
- **After midnight, figures:** `make_figures.py` now points at the real 2300-step files and reads their `max_steps_total` field, so the horizon check actually runs on them. The worker had flagged two reader bugs, but in the current code error rows don't crash the reader and drops are scored whenever a file records them, so no fix was needed for those. It also flagged that `read_rl_log` undercounts after a resume. That's still true, but the two logs using that reader have the expected 45 trials, and the locked-set slot reads a results file instead, so no current figure is affected. The figures have not been re-run yet.
- **12:05am, surprise:** Preparing the F1 check, a review found that `model_a.pt` is the same file as `model_a_f1_0691_run.pt`. Model A's closed-loop and RL results appear to use the F1 0.691 retrain, not the cited 0.773 model. The F1 check's own safety gate should stop on this as designed, so it has produced no numbers yet. The script is written and reviewed. In the next gap I'll measure this checkpoint and an earlier `model_a.pt` from git history, then wait for a decision on the gate. Nothing was overwritten. Hand-tuned Model A at 2300 steps is still running.
- **12:05am, checked and decided:** I confirmed the hash match, the matching timestamps, and the `results.md` entry saying the restore run landed at F1 0.691. This is still a file match, and F1 hasn't been measured on this file yet. `model_b.pt` has the same hash as `model_b_f1_0799_backup.pt`, so Model B is the 0.799 file. I told the worker to check `model_a.pt` against the logged 0.691 and accuracy 0.887, since that is the file behind every Model A closed-loop and RL number. The older `model_a.pt` from git (commit 9d4d079) gets its own row if it reproduces 0.773. No checkpoint gets replaced, and which Model A the paper cites is your call. If the older file does reproduce 0.773, an optional hand-tuned run with it comes after queue item 4, if the night allows.
- **12:37-12:38am:** F1 check ran on CPU between simulation jobs. All four rows reproduced their logged F1 within the +/-0.001 tolerance, including 0.691 for `model_a.pt` and 0.773 for the older file. Falls also matched 40 raw trial files. The script's last edits came after its code review and were only compiled and re-read, so the gate is their main check. I recomputed every F1 from the confusion counts, and all match.
- **12:36am and 1:23am:** Hand-tuned Model A and Model B finished at 2300 steps: 1,005 trials each, no errors, no duplicates. Results are in the table above.
- **Around 12:40am:** The worker hit the 5-hour usage limit before logging either run, and queue items 2-5 never started. The optional 0.773 hand-tuned run didn't happen either.
- **6am, correction to my earlier messages:** I called objects past 5 cm "dropped" and said drops were probably a big part of why RL isn't reaching 70%. Both claims went further than the data. Fall sizes for hand-tuned Model A:

  | Fall before the pull | Trials | Still "held" under old scoring | What it looks like |
  |---|---|---|---|
  | 5 cm or less | 424 | 190 | Gripped |
  | 5-15 cm | 434 | 129 | Mixed: 5-7 cm looks gripped, 7-15 cm mostly lost |
  | Over 15 cm | 147 | 129 | Near floor level, not a real hold |

  In 1,500 recorded training trials, objects that sank 5-7 cm had stopped moving, with a median of 5 fingers in contact and more grip force than the 5 cm or less group (49 vs. 16). Their slip rate was the same (75%). Objects that fell 7-15 cm had almost no force (0.15) and 98% slipped. That is training data at mixed squeeze levels, not the closed-loop benchmark, so it is strong evidence but not a direct measurement.
- **6am, correction to `results.md`:** Its older line says formation drops are "structurally impossible" under hand-tuned control. On the same 45 trials, hand-tuned passes 5 cm in 23, and RL in 11, 14 and 22. RL didn't create this failure mode. A correction is appended there.
- **6am, per object (hand-tuned Model A, 67 trials each):**

  | Object | Over 5 cm | Over 15 cm | Held, 5 cm rule | Held, 15 cm rule |
  |---|---|---|---|---|
  | dumbbell | 35 | 0 | 32 | 67 |
  | sphere | 0 | 0 | 48 | 48 |
  | rod | 61 | 0 | 4 | 43 |
  | capsule | 9 | 0 | 21 | 26 |
  | square_pyramid | 8 | 2 | 25 | 25 |
  | ellipsoid | 11 | 0 | 23 | 23 |
  | l_block | 62 | 17 | 3 | 19 |
  | triangular_prism | 67 | 0 | 0 | 19 |
  | cone | 9 | 1 | 15 | 15 |
  | hexagonal_prism | 67 | 0 | 0 | 14 |
  | plus_cross | 41 | 19 | 10 | 10 |
  | rectangular_prism | 59 | 37 | 5 | 6 |
  | flat_disk | 58 | 57 | 3 | 3 |
  | cube | 44 | 13 | 1 | 1 |
  | cylindrical_prism | 50 | 1 | 0 | 0 |
