"""Read-only: reads the Nov 4 diagnostic chains and the longer-rollout retrain, all on the
15 x seeds 1000-1009 dev set. Changes nothing. Run from the repo root.

1. Every controller at 5 cm and 10 cm, with what happened to the trials it lost.
2. Longer rollouts against the pre-registered verdict in results.md.
3. Each policy with the TCN signal removed, paired with itself.
4. Each controller with the noslip pass on, paired with itself under default contact settings.
5. Per object, default against noslip.
6. Movement and friction at full load, default against noslip.
"""
import json
import math
import statistics as st
from pathlib import Path

DATA = Path("Data")
DIAG = DATA / "diag_contact_physics"
GRIPS = DATA / "diag_scripted_grip"


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            key = (r["object"], int(r["seed"]))
            if key in rows:
                raise ValueError(f"duplicate row {key} in {path}")
            rows[key] = r
    return rows


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


def row_line(name, D):
    n = len(D)
    h5, h10 = sum(held(r) for r in D.values()), sum(held(r, .10) for r in D.values())
    lo, hi = wilson(h5, n)
    fell = sum(r["formation_max_fall"] > .05 for r in D.values())
    slip = sum(r["slipped"] and r["formation_max_fall"] <= .05 for r in D.values())
    return f"{name:<30}{h5:>5}/{n:<4}{100 * h5 / n:>6.1f}%{lo:>7.1f}-{hi:<6.1f}{100 * h10 / n:>7.1f}%{fell:>7}{slip:>9}"


def paired(a, b, cutoff=0.05):
    """(held only by a, held only by b, p). Fails loudly if the trial sets differ."""
    if set(a) != set(b):
        raise ValueError("trial sets differ")
    x = sum(held(a[k], cutoff) and not held(b[k], cutoff) for k in a)
    y = sum(held(b[k], cutoff) and not held(a[k], cutoff) for k in a)
    return x, y, mcnemar(x, y)


NAMES = {"seed1": "latest", "seed2": "seedcheck", "seed3": "seed3"}
base = {s: load(DATA / f"rl_{s}_dev_deterministic.jsonl") for s in NAMES}
retrain = {s: load(DATA / f"rl_nsteps512_{s}_dev_deterministic.jsonl") for s in NAMES}
no_tcn = {s: load(DIAG / f"{s}_dev150_noTCN.jsonl") for s in NAMES}
noslip = {s: load(DIAG / f"{s}_dev150_noslip20.jsonl") for s in NAMES}
hand = load(DIAG / "handA_dev150_default.jsonl")
hand_noslip = load(DIAG / "handA_dev150_noslip20.jsonl")

print(f"{'dev set, 150 trials':<30}{'held at 5 cm':>16}{'95% CI':>14}{'10 cm':>8}{'fell':>7}{'slipped':>9}")
for title, group in (("baseline", base), ("n_steps 512", retrain), ("TCN removed", no_tcn), ("noslip pass", noslip)):
    for s, D in group.items():
        print(row_line(f"RL {s} {title}", D))
print(row_line("hand-tuned A", hand))
print(row_line("hand-tuned A noslip pass", hand_noslip))

print("\nLonger rollouts, pre-registered verdict, 5 cm (10 cm reported beside it)")
for cutoff, label in ((.05, "5 cm"), (.10, "10 cm")):
    b = [100 * sum(held(r, cutoff) for r in base[s].values()) / 150 for s in NAMES]
    n = [100 * sum(held(r, cutoff) for r in retrain[s].values()) / 150 for s in NAMES]
    bar = max(15, st.stdev(b))
    verdict = ("improved" if min(n) > max(b) and st.mean(n) - st.mean(b) >= bar else
               "worse" if max(n) < min(b) and st.mean(b) - st.mean(n) >= bar else "inconclusive")
    print(f"  {label}: baseline {[round(v, 1) for v in b]} mean {st.mean(b):.1f} SD {st.stdev(b):.1f}   "
          f"n_steps 512 {[round(v, 1) for v in n]} mean {st.mean(n):.1f} SD {st.stdev(n):.1f}   "
          f"difference {st.mean(n) - st.mean(b):+.1f}, bar {bar:.1f}: {verdict}")

print("\nTCN signal removed, paired with the same policy with it (held only without / only with)")
for s in NAMES:
    for cutoff, label in ((.05, "5 cm"), (.10, "10 cm")):
        x, y, p = paired(no_tcn[s], base[s], cutoff)
        print(f"  {s} {label:<6} {x:>3} / {y:<3} p={p:.3f}")

print("\nNoslip pass, paired with the same controller without it (held only with / only without)")
for cutoff, label in ((.05, "5 cm"), (.10, "10 cm")):
    for s in NAMES:
        x, y, p = paired(noslip[s], base[s], cutoff)
        print(f"  {s:<14} {label:<6} {x:>3} / {y:<3} p={p:.2g}")
    x, y, p = paired(hand_noslip, hand, cutoff)
    print(f"  {'hand-tuned A':<14} {label:<6} {x:>3} / {y:<3} p={p:.2g}")

print("\nRL against hand-tuned A, each under its own contact settings (held only by RL / only by hand-tuned)")
for title, group, h in (("default", base, hand), ("noslip", noslip, hand_noslip)):
    for s in NAMES:
        for cutoff, label in ((.05, "5 cm"), (.10, "10 cm")):
            x, y, p = paired(group[s], h, cutoff)
            print(f"  {title:<8} {s} {label:<6} {x:>3} / {y:<3} p={p:.2g}")

print("\nHeld of 10 at 5 cm, per object, default > noslip")
objects = sorted({k[0] for k in base["seed1"]})
print(f"{'object':<22}{'seed 1':>10}{'seed 2':>10}{'seed 3':>10}{'hand A':>10}   seed 1 noslip losses (fell/slipped)")
for obj in objects:
    def pair(a, b):
        return f"{sum(held(r) for k, r in a.items() if k[0] == obj)} > {sum(held(r) for k, r in b.items() if k[0] == obj)}"
    lost = [r for k, r in noslip["seed1"].items() if k[0] == obj and not held(r)]
    print(f"{obj:<22}{pair(base['seed1'], noslip['seed1']):>10}{pair(base['seed2'], noslip['seed2']):>10}"
          f"{pair(base['seed3'], noslip['seed3']):>10}{pair(hand, hand_noslip):>10}"
          f"   {sum(r['formation_max_fall'] > .05 for r in lost)}/{sum(r['formation_max_fall'] <= .05 and r['slipped'] for r in lost)}")

print("\nAt full load, held-through trials only (median of per-trial values)")
print(f"{'controller':<30}{'trials':>7}{'speed mm/s':>12}{'friction used':>15}{'capacity/load':>15}")
for name, path in (("RL seed1 default", DIAG / "seed1_dev150_default.jsonl"),
                   ("RL seed1 noslip", DIAG / "seed1_dev150_noslip20.jsonl"),
                   ("RL seed2 noslip", DIAG / "seed2_dev150_noslip20.jsonl"),
                   ("RL seed3 noslip", DIAG / "seed3_dev150_noslip20.jsonl"),
                   ("hand-tuned A default", DIAG / "handA_dev150_default.jsonl"),
                   ("hand-tuned A noslip", DIAG / "handA_dev150_noslip20.jsonl")):
    fl = [r["full_load"] for r in load(path).values() if r.get("full_load")]
    if not fl:
        continue

    def med(*names):
        vals = [f[n] for f in fl for n in names if f.get(n) is not None]
        return st.median(vals) if vals else float("nan")
    keys = sorted(fl[0])
    print(f"{name:<30}{len(fl):>7}{med('speed_mm_s', 'object_speed_mm_s'):>12.2f}"
          f"{med('friction_used_median', 'friction_used'):>15.3f}{med('capacity_over_load'):>15.1f}")
print("  full_load fields available:", keys)
