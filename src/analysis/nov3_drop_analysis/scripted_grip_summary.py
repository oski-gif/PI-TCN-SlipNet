"""Read-only: compares the scripted grips with the RL policies on the same dev-set trials.
Pass the .jsonl from diagnostic_scripted_grip.py. Dev set only, so it can inform choices,
but a best-of-many grip needs confirming on fresh seeds. Run from the repo root."""
import json
import math
import sys
from collections import defaultdict


def held(r, cutoff=0.05):
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
groups = defaultdict(dict)
for r in rows:
    # The same grip can run under several contact settings, and those are separate results.
    variant = r.get("variant", "default")
    name = r["grip"] if variant == "default" else f'{r["grip"]} [{variant}]'
    groups[name][(r["object"], int(r["seed"]))] = r
rl = {}
for s in ("seed1", "seed2", "seed3"):
    rl[f"RL {s}"] = {(r["object"], int(r["seed"])): r for r in map(json.loads, open(f"Data/rl_{s}_dev_deterministic.jsonl", encoding="utf-8"))}

print(f"{'controller':<36}{'5 cm':>16}{'95% CI':>14}{'10 cm':>10}{'fell>5cm':>10}{'slipped':>9}")
for name, D in {**groups, **rl}.items():
    n = len(D)
    h5 = sum(held(r) for r in D.values())
    lo, hi = wilson(h5, n)
    h10 = sum(held(r, .10) for r in D.values())
    fell = sum(r["formation_max_fall"] > .05 for r in D.values())
    slip = sum(r["slipped"] and r["formation_max_fall"] <= .05 for r in D.values())
    print(f"{name:<36}{h5:>7}/{n:<3}{100 * h5 / n:>5.1f}%{lo:>8.1f}-{hi:<5.1f}{100 * h10 / n:>9.1f}%{fell:>10}{slip:>9}")

print("\nPaired with RL seed 1 at 5 cm (held only by the grip / only by seed 1, exact McNemar)")
base = rl["RL seed1"]
for name, D in groups.items():
    keys = [k for k in D if k in base]
    b = sum(held(D[k]) and not held(base[k]) for k in keys)
    c = sum(held(base[k]) and not held(D[k]) for k in keys)
    print(f"  {name:<36} {b:>3} / {c:<3} p={mcnemar(b, c):.3f}")

print("\nHeld of 10 at 5 cm, per object")
objects = sorted({k[0] for D in groups.values() for k in D})
names = list(groups) + list(rl)
print(f"{'object':<20}" + "".join(f"{n.replace('RL ', '')[:11]:>12}" for n in names))
for obj in objects:
    line = f"{obj:<20}"
    for name in names:
        D = {**groups, **rl}[name]
        line += f"{sum(held(r) for k, r in D.items() if k[0] == obj):>12}"
    print(line)
