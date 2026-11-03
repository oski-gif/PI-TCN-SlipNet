"""Read-only: does the new evaluation reproduce the logged RL results, and how do both rules score?"""
import json
import os
import re

PAT = re.compile(r"^(\S+)\s+seed=(\d+)\s+held=(True|False)\s+slipped=(True|False)\s+"
                 r"dropped_during_formation=(True|False)\s+max_disp=([\d.]+)")
SEED1_PER_OBJECT = {"cone": 3, "square_pyramid": 3, "triangular_prism": 3, "hexagonal_prism": 2,
                    "plus_cross": 3, "capsule": 3, "cylindrical_prism": 1, "sphere": 3, "rod": 3,
                    "dumbbell": 3, "ellipsoid": 1, "cube": 0, "rectangular_prism": 0,
                    "flat_disk": 0, "l_block": 1}
CUTOFFS = (0.05, 0.10, 0.15)


def load_jsonl(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                rows[(r["object"], int(r["seed"]))] = r
    return rows


def load_log(path):
    rows = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = PAT.match(line.strip())
        if m:
            rows[(m.group(1), int(m.group(2)))] = {
                "held": m.group(3) == "True", "slipped": m.group(4) == "True",
                "dropped": m.group(5) == "True", "max_disp": float(m.group(6))}
    return rows


def compare(new, old, full):
    """full: also compare slipped and displacement on dropped trials, which only a
    default run can match, because a continuing run keeps going after the drop."""
    bad = []
    for k, o in old.items():
        n = new.get(k)
        if n is None:
            bad.append((k, "missing"))
            continue
        checks = [n["held"] == o["held"], n["dropped_during_formation"] == o["dropped"]]
        if full or not o["dropped"]:
            checks += [n["slipped"] == o["slipped"], abs(n["max_displacement"] - o["max_disp"]) < 6e-5]
        if not all(checks):
            bad.append((k, {"new": (n["held"], n["slipped"], n["dropped_during_formation"], round(n["max_displacement"], 4)), "old": o}))
    return bad


def held(r, cutoff):
    return not r["slipped"] and r["formation_max_fall"] <= cutoff


old2, old3 = load_log("Data/_seedcheck_eval_output.log"), load_log("Data/_seed3_eval_output.log")
d3 = load_jsonl("Data/rescore_check_seed3_default.jsonl")
print(f"default mode, seed 3: {len(d3)} trials, held {sum(r['held'] for r in d3.values())} (logged {sum(o['held'] for o in old3.values())}); mismatches: {compare(d3, old3, True)}")
runs = {name: load_jsonl(f"Data/rescore_45_{name}.jsonl") for name in ("seed1", "seed2", "seed3")}
for name, old in (("seed2", old2), ("seed3", old3)):
    print(f"continue mode, {name}: {len(runs[name])} trials, held {sum(r['held'] for r in runs[name].values())} (logged {sum(o['held'] for o in old.values())}); mismatches: {compare(runs[name], old, False)}")
s1 = runs["seed1"]
per = {o: sum(r["held"] for k, r in s1.items() if k[0] == o) for o in SEED1_PER_OBJECT}
print(f"continue mode, seed1: {len(s1)} trials, held {sum(r['held'] for r in s1.values())} (logged 29); per-object differences: { {o: (per[o], SEED1_PER_OBJECT[o]) for o in per if per[o] != SEED1_PER_OBJECT[o]} }")

hand = {"hand-tuned A": load_jsonl("Data/handtuned_modelA_2300step_results.jsonl"),
        "hand-tuned B": load_jsonl("Data/handtuned_modelB_2300step_results.jsonl")}
keys = sorted(k for k in hand["hand-tuned A"] if k[1] <= 2)
print("\nheld on the same 45 trials, by fall cutoff (cm)")
for name, rows in {**runs, **hand}.items():
    if all(k in rows for k in keys):
        print(f"  {name:<14}" + "  ".join(f"{int(c*100)}cm {sum(held(rows[k], c) for k in keys):>2}/45 ({100*sum(held(rows[k], c) for k in keys)/45:.1f}%)" for c in CUTOFFS))
for name, rows in runs.items():
    if all(k in rows for k in keys):
        falls = [rows[k]["formation_max_fall"] for k in keys]
        print(f"  {name}: falls <=5cm {sum(f <= .05 for f in falls)}, 5-15cm {sum(.05 < f <= .15 for f in falls)}, >15cm {sum(f > .15 for f in falls)}")
        a = hand["hand-tuned A"]
        print(f"     paired at 10cm with hand-tuned A: RL only {sum(held(rows[k], .10) and not held(a[k], .10) for k in keys)}, hand-tuned only {sum(held(a[k], .10) and not held(rows[k], .10) for k in keys)}")
        print("     per object at 5cm/10cm:", {o: f"{sum(held(rows[k], .05) for k in keys if k[0] == o)}/{sum(held(rows[k], .10) for k in keys if k[0] == o)}" for o in SEED1_PER_OBJECT})
