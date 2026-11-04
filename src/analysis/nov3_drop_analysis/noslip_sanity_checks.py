"""Read-only: is the behavior under the noslip pass physically sensible, or a new artifact?
Uses the per-step samples in Data/diag_contact_physics/*_dev150_{default,noslip20}.jsonl.
Run from the repo root.

1. At the moment of a slip, how close are the contacts to their friction limit?
   Coulomb friction allows sliding only at the limit.
2. Same trials under both settings: does the object still creep at full load?
3. Any sign of instability: unphysical speeds, or the object thrown upward?
4. The trials noslip loses: how do they fail?
5. Does the policy do the same thing with its fingers under both settings?
"""
import json
import statistics as st
from pathlib import Path

DIAG = Path("Data/diag_contact_physics")


def load(path):
    return {(r["object"], int(r["seed"])): r
            for r in map(json.loads, open(path, encoding="utf-8")) if r}


def held(r):
    return (not r["slipped"]) and r["formation_max_fall"] <= 0.05


def q(values, f):
    v = sorted(values)
    return v[min(len(v) - 1, int(f * len(v)))] if v else float("nan")


default = {"seed1": load(DIAG / "seed1_dev150_default.jsonl")}
noslip = {s: load(DIAG / f"{s}_dev150_noslip20.jsonl") for s in ("seed1", "seed2", "seed3")}

print("1. Contacts in the last samples before a slip, trials that slipped without falling past 5 cm")
print(f"{'':<18}{'slips':>6}{'worst contact / limit':>32}{'load carried / capacity':>28}")
print(f"{'':<18}{'':>6}{'10th':>10}{'median':>10}{'90th':>12}{'median':>24}")
for name, D in (("seed 1 default", default["seed1"]), ("seed 1 noslip", noslip["seed1"]),
                ("seed 2 noslip", noslip["seed2"]), ("seed 3 noslip", noslip["seed3"])):
    worst, used = [], []
    for r in D.values():
        if not r["slipped"] or r["formation_max_fall"] > 0.05:
            continue
        last = [s["contact"] for s in r["samples"][-3:] if s["contact"]["n_active"] > 0]
        if not last:
            continue
        worst.append(st.mean(c["worst_ratio"] for c in last))
        used.append(st.mean(c["tangential"] / c["capacity"] for c in last if c["capacity"] > 0))
    print(f"{name:<18}{len(worst):>6}{q(worst, .1):>10.3f}{st.median(worst):>10.3f}{q(worst, .9):>12.3f}{st.median(used):>24.3f}")

print("\n2. Trials that reach full load under both settings, seed 1")
pairs = [(default["seed1"][k], noslip["seed1"][k]) for k in default["seed1"]
         if default["seed1"][k].get("full_load") and noslip["seed1"][k].get("full_load")]
for label, get in (("object speed mm/s", lambda f: f["object_speed_mm_s"]),
                   ("object moved mm", lambda f: f["object_move_mm"]),
                   ("friction used", lambda f: f["friction_used"]),
                   ("worst contact / limit", lambda f: f["worst_contact_ratio"]),
                   ("up: normal share", lambda f: f["up_normal"] / max(1e-9, f["load_expected"])),
                   ("up: friction share", lambda f: f["up_friction"] / max(1e-9, f["load_expected"]))):
    a = [get(x["full_load"]) for x, _ in pairs]
    b = [get(y["full_load"]) for _, y in pairs]
    print(f"  {label:<24} n={len(pairs)}  default median {st.median(a):7.3f}  noslip median {st.median(b):7.3f}")
print("  Speeds, held-through trials only, by what carries the load:")
for name, D in (("default", default["seed1"]), ("noslip", noslip["seed1"])):
    fl = [r["full_load"] for r in D.values() if r.get("full_load") and held(r)]
    fr = [f["object_speed_mm_s"] for f in fl if f["up_friction"] > f["up_normal"]]
    no = [f["object_speed_mm_s"] for f in fl if f["up_friction"] <= f["up_normal"]]
    print(f"    {name:<8} carried by friction: n={len(fr):<3} median {st.median(fr) if fr else float('nan'):.2f} mm/s   "
          f"by normal force: n={len(no):<3} median {st.median(no) if no else float('nan'):.2f} mm/s")

print("\n3. Largest object speed between samples (10 RL steps = 0.1 s), and highest rise above the settle position")
for name, D in (("seed 1 default", default["seed1"]), ("seed 1 noslip", noslip["seed1"]),
                ("seed 2 noslip", noslip["seed2"]), ("seed 3 noslip", noslip["seed3"])):
    fastest, rises = [], []
    for r in D.values():
        S = r["samples"]
        spd = []
        for a, b in zip(S, S[1:]):
            dt = (b["rl_step"] - a["rl_step"]) * 0.01
            if dt > 0:
                spd.append(sum((x - y) ** 2 for x, y in zip(a["object_pos"], b["object_pos"])) ** .5 / dt)
        fastest.append(max(spd) if spd else 0.0)
        settle = [s for s in S if s["rl_step"] >= 160]
        if settle:
            rises.append(max(s["object_pos"][2] for s in settle) - settle[0]["object_pos"][2])
    print(f"  {name:<16} fastest per trial: median {st.median(fastest):5.2f} m/s, max {max(fastest):5.2f} m/s;  "
          f"rise after settle: median {1000 * st.median(rises):6.2f} mm, max {1000 * max(rises):6.1f} mm")

print("\n4. Trials held under default but lost under noslip, seed 1")
for k in sorted(default["seed1"]):
    a, b = default["seed1"][k], noslip["seed1"][k]
    if held(a) and not held(b):
        print(f"  {k[0]:<20} seed={k[1]}  fall {1000 * a['formation_max_fall']:6.1f} > {1000 * b['formation_max_fall']:6.1f} mm   "
              f"max_disp {1000 * a['max_displacement']:6.1f} > {1000 * b['max_displacement']:6.1f} mm  (limit {1000 * b['threshold']:.1f})  "
              f"steps {a['ep_len']} > {b['ep_len']}")

print("\n5. Finger closure on the same trials, seed 1, median over 150 trials (thumb, index, middle, ring, pinky)")
for step in (10, 20, 50, 160):
    def closure(D):
        out = []
        for r in D.values():
            s = [x for x in r["samples"] if x["rl_step"] == step]
            if s:
                out.append(s[0]["fraction"])
        return [round(st.median(v[i] for v in out), 2) for i in range(5)]
    print(f"  RL step {step:<4} default {closure(default['seed1'])}   noslip {closure(noslip['seed1'])}")
print("  Slip probability during the pull, median and 90th percentile:")
for name, D in (("default", default["seed1"]), ("noslip", noslip["seed1"])):
    probs = [s["slip_probability"] for r in D.values() for s in r["samples"] if s["phase"] == 2]
    print(f"    {name:<8} {st.median(probs):.3f}  {q(probs, .9):.3f}")
