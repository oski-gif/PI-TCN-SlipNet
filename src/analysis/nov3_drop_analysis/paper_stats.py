"""Read-only: confidence intervals, paired tests and breakdowns used in reports/paper_outline.md.
Run from the repo root. Files that don't exist yet are skipped."""
import json
import math
import os
import re
from collections import defaultdict

KIND = {"cube": "primitive", "sphere": "primitive", "rectangular_prism": "primitive",
        "cylindrical_prism": "primitive", "capsule": "primitive", "ellipsoid": "primitive",
        "flat_disk": "primitive", "rod": "primitive", "cone": "mesh", "square_pyramid": "mesh",
        "triangular_prism": "mesh", "hexagonal_prism": "mesh", "l_block": "compound",
        "dumbbell": "compound", "plus_cross": "compound"}
HOLDOUT = {"cone", "cylindrical_prism", "plus_cross", "sphere"}


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                rows[(r["object"], int(r["seed"]))] = r
    return rows


def held(r, cutoff):
    return not r["slipped"] and r["formation_max_fall"] <= cutoff


def rate(label, k, n):
    lo, hi = wilson(k, n)
    print(f"  {label:<34} {k:>4}/{n:<5} {100 * k / n:5.1f}%  CI {lo:5.1f}-{hi:5.1f}")


A = load("Data/handtuned_modelA_2300step_results.jsonl")
B = load("Data/handtuned_modelB_2300step_results.jsonl")
fall = {k: A[k]["formation_max_fall"] for k in A}

print("Hand-tuned, 2300 steps, all 1,005 trials")
for name, D in (("A", A), ("B", B)):
    rate(f"{name} no rule", sum(not r["slipped"] for r in D.values()), len(D))
    for c in (0.10, 0.05):
        rate(f"{name} {int(c * 100)} cm", sum(held(r, c) for r in D.values()), len(D))
print("Hand-tuned, seeds 3-66 (960)")
for name, D in (("A", A), ("B", B)):
    keys = [k for k in D if k[1] >= 3]
    for c in (0.10, 0.05):
        rate(f"{name} {int(c * 100)} cm", sum(held(D[k], c) for k in keys), len(keys))
print("A vs B paired, 1,005 trials")
for c in (0.10, 0.05):
    b = sum(held(A[k], c) and not held(B[k], c) for k in A)
    d = sum(held(B[k], c) and not held(A[k], c) for k in A)
    print(f"  {int(c * 100)} cm: A only {b}, B only {d}, exact McNemar p={mcnemar_exact(b, d):.3f}")
b = sum((not A[k]["slipped"]) and B[k]["slipped"] for k in A)
d = sum((not B[k]["slipped"]) and A[k]["slipped"] for k in A)
print(f"  no rule: A only {b}, B only {d}, exact McNemar p={mcnemar_exact(b, d):.3f}")

print("Short test, 1500 steps, 1,005 trials, falls joined from the 2300-step run")
for name, path in (("A", "Data/baseline_modelA_1000trial_results.jsonl"), ("B", "Data/baseline_modelB_1000trial_results.jsonl")):
    D = load(path)
    rate(f"{name} no rule", sum(not r["slipped"] for r in D.values()), len(D))
    for c in (0.10, 0.05):
        rate(f"{name} {int(c * 100)} cm", sum((not D[k]["slipped"]) and fall[k] <= c for k in D), len(D))

print("Same 45 trials (seeds 0-2), 2300 steps")
keys = [k for k in A if k[1] <= 2]
runs = {n: load(f"Data/rescore_45_{n}.jsonl") for n in ("seed1", "seed2", "seed3")}
for name, D in {"hand-tuned A": A, "hand-tuned B": B, **runs}.items():
    for c in (0.05, 0.10):
        rate(f"{name} {int(c * 100)} cm", sum(held(D[k], c) for k in keys), 45)
for name, D in runs.items():
    for c in (0.05, 0.10):
        b = sum(held(D[k], c) and not held(A[k], c) for k in keys)
        d = sum(held(A[k], c) and not held(D[k], c) for k in keys)
        print(f"  {name} vs hand-tuned A, {int(c * 100)} cm: RL only {b}, hand-tuned only {d}, exact McNemar p={mcnemar_exact(b, d):.4f}")

print("Fall bands, hand-tuned A")
for lab, lo, hi in (("<=5", -1, .05), ("5-7", .05, .07), ("7-15", .07, .15), (">15", .15, 9)):
    ks = [k for k in fall if lo < fall[k] <= hi]
    print(f"  {lab:<6} trials {len(ks):>4}, not slipped {sum(not A[k]['slipped'] for k in ks)}")

print("By object kind and holdout group, hand-tuned A (held, 5 cm / 10 cm)")
groups = defaultdict(list)
for k in A:
    groups[KIND[k[0]]].append(k)
    groups["holdout" if k[0] in HOLDOUT else "train objects"].append(k)
for g, ks in groups.items():
    print(f"  {g:<14} n={len(ks):>4}  5cm {sum(held(A[k], .05) for k in ks):>3}  10cm {sum(held(A[k], .10) for k in ks):>3}")

print("Corrections, hand-tuned, 2300 steps, held under the 10 cm rule")
for name, D in (("A", A), ("B", B)):
    ks = [k for k in D if held(D[k], .10)]
    zero = sum(D[k]["n_corrections"] == 0 for k in ks)
    all_zero = sum(r["n_corrections"] == 0 for r in D.values())
    print(f"  {name}: held {len(ks)}, of which zero corrections {zero} ({100 * zero / len(ks):.1f}%); all trials with zero corrections {all_zero}")

for s in ("seed1", "seed2", "seed3"):
    path = f"Data/rl_{s}_1005trial_results.jsonl"
    if os.path.exists(path):
        D = load(path)
        print(f"RL {s}, locked set so far: {len(D)} trials")
        if len(D) == 1005:
            for c in (0.05, 0.10):
                rate(f"{s} {int(c * 100)} cm, all", sum(held(r, c) for r in D.values()), len(D))
                ks = [k for k in D if k[1] >= 3]
                rate(f"{s} {int(c * 100)} cm, seeds 3-66", sum(held(D[k], c) for k in ks), len(ks))
                b = sum(held(D[k], c) and not held(A[k], c) for k in D)
                d = sum(held(A[k], c) and not held(D[k], c) for k in D)
                print(f"    vs hand-tuned A: RL only {b}, hand-tuned only {d}, exact McNemar p={mcnemar_exact(b, d):.2e}")
