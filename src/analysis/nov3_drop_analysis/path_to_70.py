"""Read-only: where the missing trials are if the target is 70% held at 5 cm, using the
finished locked-set runs. Descriptive only: this looks at locked-set data, so nothing here
may be used to pick a policy. Choices stay on the dev set. Run from the repo root."""
import json
import math
import statistics as st
from collections import defaultdict


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return {(r["object"], int(r["seed"])): r for r in rows if "error" not in r}


def held(r, cutoff=0.05):
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff


RL = {s: load(f"Data/rl_{s}_1005trial_results.jsonl") for s in ("seed1", "seed2", "seed3")}
HAND = {"A": load("Data/handtuned_modelA_2300step_results.jsonl"),
        "B": load("Data/handtuned_modelB_2300step_results.jsonl")}
ALL = {**RL, **HAND}
best = RL["seed1"]
keys = sorted(best)
primary = [k for k in keys if k[1] >= 3]
objects = sorted({k[0] for k in keys})

print("Trials needed for 70%")
for label, ks in (("seeds 3-66", primary), ("all 1,005", keys)):
    need = math.ceil(0.7 * len(ks))
    have = sum(held(best[k]) for k in ks)
    print(f"  {label}: need {need}, seed 1 has {have}, short by {need - have}")

print("Held by at least one controller, a floor on what can be held, not the ceiling")
for label, group in (("any RL policy", RL), ("any of 3 RL + hand-tuned A, B", ALL)):
    for ks_label, ks in (("seeds 3-66", primary), ("all 1,005", keys)):
        h = sum(any(held(D[k]) for D in group.values()) for k in ks)
        print(f"  {label}, {ks_label}: {h}/{len(ks)} {100 * h / len(ks):.1f}%")

print("Best existing controller for each object, if it were known in advance, seeds 3-66")
total = 0
for obj in objects:
    ks = [k for k in primary if k[0] == obj]
    counts = {name: sum(held(D[k]) for k in ks) for name, D in ALL.items()}
    winner = max(counts, key=counts.get)
    total += counts[winner]
    print(f"  {obj:<20} {counts[winner]:>3}/{len(ks)} ({winner})")
print(f"  total {total}/{len(primary)} {100 * total / len(primary):.1f}%")

print("Seed 1 per object at 5 cm, seeds 3-66: held, fell past 5 cm, slipped; any controller; never held")
weak, strong = [], []
for obj in sorted(objects, key=lambda o: sum(held(best[k]) for k in primary if k[0] == o)):
    ks = [k for k in primary if k[0] == obj]
    h = sum(held(best[k]) for k in ks)
    fell = sum(best[k]["formation_max_fall"] > 0.05 for k in ks)
    slipped = sum(best[k]["slipped"] and best[k]["formation_max_fall"] <= 0.05 for k in ks)
    union = sum(any(held(D[k]) for D in ALL.values()) for k in ks)
    (weak if h / len(ks) < 0.7 else strong).append((obj, h, len(ks)))
    print(f"  {obj:<20} {h:>3} {fell:>3} {slipped:>3}   {union:>3}   {len(ks) - union:>3}")
for label, group in (("at or above 70%", strong), ("below 70%", weak)):
    h, n = sum(g[1] for g in group), sum(g[2] for g in group)
    print(f"  {len(group)} objects {label}: {h}/{n} {100 * h / n:.1f}%")
n_strong, h_strong = sum(g[2] for g in strong), sum(g[1] for g in strong)
n_weak = sum(g[2] for g in weak)
need = math.ceil(0.7 * len(primary))
print(f"  for 70% with the strong group unchanged, the weak group needs {need - h_strong}/{n_weak} "
      f"{100 * (need - h_strong) / n_weak:.1f}%; with the strong group perfect, {need - n_strong}/{n_weak} "
      f"{100 * (need - n_strong) / n_weak:.1f}%")

print("Seed 1 falls past 5 cm, seeds 3-66. Slip sizes aren't listed: RL trials end at the slip limit")
falls = [best[k]["formation_max_fall"] for k in primary if best[k]["formation_max_fall"] > 0.05]
for lo, hi, label in ((.05, .07, "5-7 cm"), (.07, .10, "7-10 cm"), (.10, .15, "10-15 cm"), (.15, 9, "over 15 cm")):
    print(f"  {label}: {sum(lo < f <= hi for f in falls)}")
