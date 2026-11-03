"""Read-only: joins the 2300-step drop records onto older hand-tuned results and RL logs."""
import collections
import json
import re

import numpy as np


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" not in r:
            rows[(r["object"], int(r["seed"]))] = r
    return rows


A = load("Data/handtuned_modelA_2300step_results.jsonl")
B = load("Data/handtuned_modelB_2300step_results.jsonl")

# 1. Is a "drop" really a fall out of the hand?
falls = np.array([r["formation_max_fall"] for r in A.values()])
print("fall histogram (m), all", len(falls))
edges = [0, .01, .02, .05, .10, .15, .20, .25, 1]
print(dict(zip([f"{a}-{b}" for a, b in zip(edges, edges[1:])], np.histogram(falls, edges)[0].tolist())))
dropped = [r for r in A.values() if r["dropped_during_formation"]]
print("median fall of dropped", round(float(np.median([r["formation_max_fall"] for r in dropped])), 3))
print("dropped with fall <= 0.10:", sum(r["formation_max_fall"] <= .10 for r in dropped),
      "of which not slipped:", sum(r["formation_max_fall"] <= .10 and not r["slipped"] for r in dropped))

# 2. Formation runs before any controller action, so drops should not depend on the model.
shared = set(A) & set(B)
print("A vs B drop flags differ on", sum(A[k]["dropped_during_formation"] != B[k]["dropped_during_formation"] for k in shared),
      "of", len(shared), "shared trials; max fall difference",
      max(abs(A[k]["formation_max_fall"] - B[k]["formation_max_fall"]) for k in shared))

# 3. Correct older hand-tuned results with A's drop flags. Valid only if shorter
# runs are prefixes of the 2300-step run, so test that first for same-model files.
for name, path, reference in [
    ("A 1500", "Data/baseline_modelA_1000trial_results.jsonl", A),
    ("A 2500 (45)", "Data/closed_loop_scenes/batch_results.jsonl", A),
    ("B 1500", "Data/baseline_modelB_1000trial_results.jsonl", A),
    ("B 2500 (45)", "Data/closed_loop_scenes/batch_results_modelb.jsonl", A),
]:
    rows = load(path)
    keys = set(rows) & set(reference)
    held = sum(not rows[k]["slipped"] for k in rows)
    real = sum(not rows[k]["slipped"] and not reference[k]["dropped_during_formation"] for k in keys)
    print(f"{name}: n={len(rows)} joined={len(keys)} held_no_rule={held} ({100*held/len(rows):.1f}%) "
          f"held_drop_rule={real} ({100*real/len(keys):.1f}%)")
    if name.startswith("A"):
        shorter = "1500" in name
        a, b = (rows, reference) if shorter else (reference, rows)
        bad = sum(a[k]["max_displacement"] > b[k]["max_displacement"] + 1e-9 for k in keys)
        print(f"  prefix check: shorter run's max displacement exceeds longer run's on {bad} of {len(keys)}")

# 4. RL vs hand-tuned Model A on the same 45 trials.
pat = re.compile(r"^(\S+)\s+seed=(\d+)\s+held=(True|False)\s+slipped=(True|False)\s+dropped_during_formation=(True|False)")
subset = [k for k in A if k[1] <= 2]
held_a = sum(not A[k]["slipped"] and not A[k]["dropped_during_formation"] for k in subset)
drop_a = sum(A[k]["dropped_during_formation"] for k in subset)
print(f"Hand-tuned A on seeds 0-2: n={len(subset)} held_drop_rule={held_a} dropped={drop_a}")
per_a = collections.Counter(k[0] for k in subset if A[k]["dropped_during_formation"])
for label, path in [("RL seed 2", "Data/_seedcheck_eval_output.log"), ("RL seed 3", "Data/_seed3_eval_output.log")]:
    rl = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.match(line.strip())
        if m:
            rl[(m.group(1), int(m.group(2)))] = (m.group(3) == "True", m.group(4) == "True", m.group(5) == "True")
    both = set(rl) & set(subset)
    a_only = sum((not A[k]["slipped"] and not A[k]["dropped_during_formation"]) and not rl[k][0] for k in both)
    r_only = sum(rl[k][0] and not (not A[k]["slipped"] and not A[k]["dropped_during_formation"]) for k in both)
    rl_drop_a_not = sum(rl[k][2] and not A[k]["dropped_during_formation"] for k in both)
    a_drop_rl_not = sum(A[k]["dropped_during_formation"] and not rl[k][2] for k in both)
    print(f"{label}: joined {len(both)} held {sum(v[0] for v in rl.values())} dropped {sum(v[2] for v in rl.values())}; "
          f"held only by hand-tuned {a_only}, only by RL {r_only}; "
          f"dropped only by RL {rl_drop_a_not}, only by hand-tuned {a_drop_rl_not}")
    per_rl = collections.Counter(k[0] for k, v in rl.items() if v[2])
    print("   per-object drops of 3 (object: hand-tuned/RL):",
          {o: f"{per_a[o]}/{per_rl[o]}" for o in sorted(set(k[0] for k in subset))})
