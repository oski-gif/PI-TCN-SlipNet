"""Read-only: two questions from the Nov 4 code review, using stored result rows only.
1. Do slips look like sudden force-limit failures, or like slow drift under constant load?
2. How does the reward score holding against slipping?
Dev-set rows are used where a choice could follow. Run from the repo root."""
import json
import statistics as st

SETTLE_END_MACRO = 160  # (500 ramp + 300 settle) / 5 physics steps per RL step
FULL_PULL_MACRO = SETTLE_END_MACRO + 100  # the pull ramp takes 500 physics steps
END_MACRO = SETTLE_END_MACRO + 300


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return {(r["object"], int(r["seed"])): r for r in rows if "error" not in r}


def held(r, cutoff=0.05):
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff


print("1a. When do RL trials slip? Trials that kept the object through formation, all 1,005")
print("    Physics steps after settle: ramp 0-500, full pull 500-1000, full pull 1000-1500")
for s in ("seed1", "seed2", "seed3"):
    D = load(f"Data/rl_{s}_1005trial_results.jsonl")
    slips = [r for r in D.values() if r["slipped"] and r["formation_max_fall"] <= 0.05]
    times = [r["ep_len"] for r in slips]
    ramp = sum(t <= FULL_PULL_MACRO for t in times)
    early = sum(FULL_PULL_MACRO < t <= FULL_PULL_MACRO + 100 for t in times)
    late = sum(t > FULL_PULL_MACRO + 100 for t in times)
    held_n = sum(held(r) for r in D.values())
    print(f"  {s}: {len(slips)} slips: ramp {ramp}, first half of full pull {early}, second half {late}; held {held_n}")
    print(f"    held rate if the test had ended earlier: "
          f"at ramp end {100 * (held_n + early + late) / len(D):.1f}%, "
          f"halfway through full pull {100 * (held_n + late) / len(D):.1f}%, "
          f"at the real end {100 * held_n / len(D):.1f}%")

print("\n1b. The same trials, hand-tuned, 200 vs. 1,000 steps at full pull (no drop rule)")
for name, short, long_ in (("A", "Data/baseline_modelA_1000trial_results.jsonl", "Data/handtuned_modelA_2300step_results.jsonl"),
                           ("B", "Data/baseline_modelB_1000trial_results.jsonl", "Data/handtuned_modelB_2300step_results.jsonl")):
    S, L = load(short), load(long_)
    keys = sorted(set(S) & set(L))
    survived = [k for k in keys if not S[k]["slipped"]]
    lost = [k for k in survived if L[k]["slipped"]]
    ratios = sorted(L[k]["max_displacement"] / L[k]["threshold"] for k in lost)
    print(f"  {name}: held at 200 steps {len(survived)}, of those failed by 1,000 steps {len(lost)} "
          f"({100 * len(lost) / len(survived):.0f}%)")
    print(f"     final slip of those, in limits: median {st.median(ratios):.2f}; under 2x {sum(x < 2 for x in ratios)}, "
          f"2-10x {sum(2 <= x < 10 for x in ratios)}, over 10x {sum(x >= 10 for x in ratios)}")

print("\n1c. Hand-tuned A, 2,300 steps, kept through formation: final displacement in limits")
A = load("Data/handtuned_modelA_2300step_results.jsonl")
kept = [r for r in A.values() if r["formation_max_fall"] <= 0.05]
ratios = [r["max_displacement"] / r["threshold"] for r in kept]
edges = [0, .25, .5, .75, 1, 1.5, 2, 5, 10, 1e9]
for lo, hi in zip(edges, edges[1:]):
    label = f"{lo}-{hi}" if hi < 1e9 else f"{lo}+"
    print(f"  {label:>8} limits: {sum(lo <= x < hi for x in ratios)}")
print(f"  of {len(kept)} kept; the limit itself is 1.0")

print("\n1d. Limit sizes and the displacement rate they allow")
limits = {}
for r in A.values():
    limits[r["object"]] = r["threshold"]
for obj, t in sorted(limits.items(), key=lambda kv: kv[1]):
    print(f"  {obj:<20} limit {1000 * t:5.1f} mm")

print("\n2. Reward by outcome, dev set, deterministic")
for s in ("seed1", "seed2", "seed3"):
    D = load(f"Data/rl_{s}_dev_deterministic.jsonl")
    for label, rows in (("held", [r for r in D.values() if held(r)]),
                        ("slipped, kept through formation", [r for r in D.values() if r["slipped"] and r["formation_max_fall"] <= .05])):
        if rows:
            ph2 = [r["phase2_reward"] for r in rows]
            form = [r["formation_reward"] for r in rows]
            disp = [1000 * r["max_displacement"] for r in rows]
            print(f"  {s} {label:<32} n={len(rows):>3}  formation {st.mean(form):7.1f}  pull phase {st.mean(ph2):7.1f}  "
                  f"(min {min(ph2):.1f}, max {max(ph2):.1f})  max displacement {st.mean(disp):5.1f} mm")
D = load("Data/rl_seed1_dev_deterministic.jsonl")
h = [r for r in D.values() if held(r)]
print(f"  seed1 held trials scoring below the -20 slip penalty in the pull phase: "
      f"{sum(r['phase2_reward'] < -20 for r in h)}/{len(h)}")
print(f"  discount factors with gamma 0.99: 300 steps {0.99 ** 300:.3f}, 460 steps {0.99 ** 460:.3f}, "
      f"so +20 at the end is worth {20 * 0.99 ** 300:.2f} at pull start and {20 * 0.99 ** 460:.2f} at trial start")
