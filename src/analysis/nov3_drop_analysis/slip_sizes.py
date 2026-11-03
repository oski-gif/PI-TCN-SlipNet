"""Read-only: how far past the slip limit do hand-tuned failures go, and how does each object fail?"""
import json
from collections import defaultdict

import numpy as np

for name in ("A", "B"):
    rows = [json.loads(l) for l in open(f"Data/handtuned_model{name}_2300step_results.jsonl") if l.strip()]
    kept = [r for r in rows if r["formation_max_fall"] <= 0.10]
    slips = [r for r in kept if r["slipped"]]
    ratio = np.array([r["max_displacement"] / r["threshold"] for r in slips])
    cm = np.array([100 * r["max_displacement"] for r in slips])
    print(f"Model {name}: stayed within 10 cm {len(kept)}, held {len(kept) - len(slips)}, slipped {len(slips)}")
    print("  slip size vs limit:", {f"{a}-{b}x": int(((ratio > a) & (ratio <= b)).sum()) for a, b in [(1, 1.25), (1.25, 1.5), (1.5, 2), (2, 5), (5, 10), (10, 1000)]})
    print(f"  median slip {np.median(cm):.1f} cm, median limit {100 * np.median([r['threshold'] for r in slips]):.2f} cm")
    # Scoring sensitivity only: what if the slip limit were looser?
    for mult in (1, 1.25, 1.5, 2, 3):
        held = sum(r["max_displacement"] <= mult * r["threshold"] for r in kept)
        print(f"  limit x{mult}: held {held}/{len(rows)} ({100 * held / len(rows):.1f}%)")
    if name == "A":
        per = defaultdict(lambda: [0, 0, 0, 0.0])
        for r in rows:
            p = per[r["object"]]
            p[3] = r["threshold"]
            if r["formation_max_fall"] > 0.10:
                p[0] += 1
            elif r["slipped"]:
                p[1] += 1
            else:
                p[2] += 1
        print("  object: fell over 10 cm, slipped in pull, held, slip limit cm")
        for obj, v in sorted(per.items(), key=lambda x: -x[1][2]):
            print(f"    {obj:<20}{v[0]:>4}{v[1]:>4}{v[2]:>4}{100 * v[3]:>7.2f}")
