"""Does removing the TCN signal change closed-loop outcomes? Dev set, same 150 trials, paired.

Pairs each with-TCN run with its no-TCN run (signal replaced by 0 for RL, no TCN for hand-tuned)
and counts trials whose outcome flips. Read-only. Run from the repo root:
    venv/Scripts/python.exe src/analysis/nov3_drop_analysis/tcn_removal_compare.py
"""
import json
import math
import os

D = "Data/diag_contact_physics"
PAIRS = [
    ("RL seed 1, default", "seed1_dev150_default", "seed1_dev150_noTCN"),
    ("RL seed 1, noslip", "seed1_dev150_noslip20", "seed1_dev150_noslip20_noTCN"),
    ("RL seed 2, noslip", "seed2_dev150_noslip20", "seed2_dev150_noslip20_noTCN"),
    ("RL seed 3, noslip", "seed3_dev150_noslip20", "seed3_dev150_noslip20_noTCN"),
    ("Hand-tuned A, default", "handA_dev150_default", "handA_dev150_default_noTCN"),
    ("Hand-tuned A, noslip", "handA_dev150_noslip20", "handA_dev150_noslip20_noTCN"),
]


def rows(name):
    out = {}
    with open(os.path.join(D, name + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["object"], int(r["seed"]))] = r
    return out


def held(r, cutoff):
    # Hand-tuned rows from diagnostic_handtuned_physics.py have no "held" field; RL's rule is the
    # same test: not slipped and no formation fall past the cutoff.
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff / 100


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


print(f"{'pair':<24} {'n':>4}  5 cm with / without  only-with / only-without   p     | 10 cm with / without  flips")
for label, a, b in PAIRS:
    if not (os.path.exists(os.path.join(D, a + ".jsonl")) and os.path.exists(os.path.join(D, b + ".jsonl"))):
        print(f"{label:<24} missing file")
        continue
    ra, rb = rows(a), rows(b)
    keys = sorted(set(ra) & set(rb))
    out = []
    for cutoff in (5, 10):
        w = sum(held(ra[k], cutoff) for k in keys)
        wo = sum(held(rb[k], cutoff) for k in keys)
        only_w = sum(held(ra[k], cutoff) and not held(rb[k], cutoff) for k in keys)
        only_wo = sum(held(rb[k], cutoff) and not held(ra[k], cutoff) for k in keys)
        out.append((w, wo, only_w, only_wo))
    (w5, wo5, b5, c5), (w10, wo10, b10, c10) = out
    print(f"{label:<24} {len(keys):>4}  {w5:>4} / {wo5:<4}            {b5:>3} / {c5:<3}            {mcnemar(b5, c5):.3f} | "
          f"{w10:>4} / {wo10:<4}          {b10 + c10}")

print()
for name in ("seed2_dev150_elliptic_impratio10", "seed3_dev150_elliptic_impratio10", "seed1_dev150_noslip5_noTCN"):
    p = os.path.join(D, name + ".jsonl")
    if os.path.exists(p):
        r = rows(name)
        h5 = sum(held(x, 5) for x in r.values())
        h10 = sum(held(x, 10) for x in r.values())
        fell = sum(x["formation_max_fall"] > 0.05 for x in r.values())
        print(f"{name}: {len(r)} rows, held {h5} at 5 cm, {h10} at 10 cm, fell past 5 cm {fell}")
