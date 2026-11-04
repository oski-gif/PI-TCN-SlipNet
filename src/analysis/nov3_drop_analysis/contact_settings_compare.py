"""Read-only: compares the contact settings with each other on the 150 dev trials, and asks
whether outcomes depend on the friction each trial got. Run from the repo root, after
Synthetic data creation/Simulation/diagnostic_trial_friction.py has recorded the friction."""
import json
import math
import statistics as st
from pathlib import Path

DIAG = Path("Data/diag_contact_physics")


def load(path):
    return {(r["object"], int(r["seed"])): r for r in map(json.loads, open(path, encoding="utf-8"))}


def held(r, cutoff=.05):
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


default = load("Data/rl_seed1_dev_deterministic.jsonl")
noslip = load(DIAG / "seed1_dev150_noslip20.jsonl")
elliptic = load(DIAG / "seed1_dev150_elliptic_impratio10.jsonl")

print("Seed 1, 150 dev trials")
for name, D in (("default", default), ("noslip 20", noslip), ("elliptic, impratio 10", elliptic)):
    print(f"  {name:<24} 5 cm {sum(held(r) for r in D.values()):>3}   10 cm {sum(held(r, .1) for r in D.values()):>3}   "
          f"fell {sum(r['formation_max_fall'] > .05 for r in D.values()):>3}   "
          f"slipped {sum(r['slipped'] and r['formation_max_fall'] <= .05 for r in D.values()):>3}")
for a, b, label in ((noslip, default, "noslip vs default"), (elliptic, default, "elliptic vs default"), (noslip, elliptic, "noslip vs elliptic")):
    x = sum(held(a[k]) and not held(b[k]) for k in a)
    y = sum(held(b[k]) and not held(a[k]) for k in a)
    print(f"  {label:<24} only first {x:>3}, only second {y:>3}, p={mcnemar(x, y):.2g}")
both = sum(held(noslip[k]) and held(elliptic[k]) and not held(default[k]) for k in default)
print(f"  Rescued from default by both settings: {both}. By noslip: {sum(held(noslip[k]) and not held(default[k]) for k in default)}, "
      f"by elliptic: {sum(held(elliptic[k]) and not held(default[k]) for k in default)}")
print("  Per object, held of 10 (default, noslip, elliptic):")
for obj in sorted({k[0] for k in default}):
    c = [sum(held(r) for k, r in D.items() if k[0] == obj) for D in (default, noslip, elliptic)]
    print(f"    {obj:<20} {c[0]:>3} {c[1]:>3} {c[2]:>3}")


def friction_table(title, D, friction, cutoff=.05):
    """Within each object, trials split into low and high friction at that object's median, so
    shape can't explain the difference. Trials that fell before the pull say little about
    friction limits at the pull, so they are left out of the second column."""
    low, high, low_s, high_s = [], [], [], []
    for obj in {k[0] for k in D}:
        keys = [k for k in D if k[0] == obj and k in friction]
        if len(keys) < 2:
            continue
        mid = st.median(friction[k]["tip_mu_mean"] for k in keys)
        for k in keys:
            side = high if friction[k]["tip_mu_mean"] > mid else low
            side.append(held(D[k], cutoff))
            if D[k]["formation_max_fall"] <= cutoff:
                (high_s if friction[k]["tip_mu_mean"] > mid else low_s).append(D[k]["slipped"])
    print(f"  {title:<30} held: low friction {100 * st.mean(low):5.1f}% (n={len(low)})  high {100 * st.mean(high):5.1f}% (n={len(high)})   "
          f"slipped, of those not fallen: low {100 * st.mean(low_s):5.1f}% (n={len(low_s)})  high {100 * st.mean(high_s):5.1f}% (n={len(high_s)})")


path = Path("Data/diag_fresh_seeds/trial_friction.jsonl")
friction = {(r["object"], int(r["seed"])): r for r in map(json.loads, open(path, encoding="utf-8"))}
print("\nHeld against pad friction, split at each object's own median")
for s in ("seed1", "seed2", "seed3"):
    friction_table(f"RL {s} default", load(f"Data/rl_{s}_dev_deterministic.jsonl"), friction)
    friction_table(f"RL {s} noslip", load(DIAG / f"{s}_dev150_noslip20.jsonl"), friction)
friction_table("RL seed1 elliptic", elliptic, friction)
friction_table("hand-tuned A default", load(DIAG / "handA_dev150_default.jsonl"), friction)
friction_table("hand-tuned A noslip", load(DIAG / "handA_dev150_noslip20.jsonl"), friction)
