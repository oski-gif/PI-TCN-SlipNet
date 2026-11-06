"""Applies the reversal rule fixed in results.md, "Decision: elliptic cone", to the fresh-seed runs.

Default goes back to primary if either:
  - fewer than 3 of the 4 controllers have a 95% Wilson lower bound, at 5 cm, above their locked-set
    default rate
  - solver warnings appear in those runs' logs
Written before any fresh-seed elliptic row existed. Thresholds are the rounded rates the rule states.

Run from the repo root:
    venv/Scripts/python.exe src/analysis/nov3_drop_analysis/elliptic_gate_check.py
"""
import argparse
import json
import math
import os
import re
import sys

THRESHOLDS = {"seed1": 61.2, "seed2": 42.2, "seed3": 15.0, "handA": 18.8}
NEEDED = 3
SOLVER_WARNING = re.compile(r"WARNING|Nan, Inf|Bad QACC|Too many contacts")


def wilson_lower(k, n, z=1.959964):
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (centre - margin) / (1 + z * z / n)


def read_rows(path):
    rows = {}
    duplicates = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["object"], int(r["seed"]))
            duplicates += key in rows
            rows[key] = r
    return rows, duplicates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="Data/diag_fresh_seeds")
    ap.add_argument("--suffix", default="elliptic_impratio10")
    ap.add_argument("--expected-rows", type=int, default=1005)
    args = ap.parse_args()

    above = warnings = 0
    complete = True
    print(f"{'controller':<10} {'rows':>5} {'held':>5} {'rate':>7} {'95% lower':>10} {'default':>8}  above?  warnings")
    for tag, threshold in THRESHOLDS.items():
        path = os.path.join(args.dir, f"{tag}_{args.suffix}.jsonl")
        log = os.path.join(args.dir, f"{tag}_{args.suffix}.log")
        if not os.path.exists(path):
            print(f"{tag:<10} missing {path}")
            complete = False
            continue
        rows, duplicates = read_rows(path)
        n = len(rows)
        held = sum(bool(r["held"]) for r in rows.values())
        lower = wilson_lower(held, n)
        is_above = lower > threshold
        with open(log, encoding="utf-8", errors="replace") as f:
            n_warn = sum(bool(SOLVER_WARNING.search(line)) for line in f)
        above += is_above
        warnings += n_warn
        if n != args.expected_rows or duplicates:
            complete = False
        print(f"{tag:<10} {n:>5} {held:>5} {100 * held / max(n, 1):>6.1f}% {lower:>9.1f}% {threshold:>7.1f}%  "
              f"{'yes' if is_above else 'NO':<6}  {n_warn}" + (f"  ({duplicates} duplicates)" if duplicates else ""))

    print(f"controllers above their default rate: {above} of {len(THRESHOLDS)}, {NEEDED} needed")
    print(f"solver warning lines in the logs: {warnings}")
    if not complete:
        print("RESULT: INCOMPLETE, the rule can't be applied yet")
        sys.exit(2)
    if above < NEEDED or warnings:
        print("RESULT: REVERSE, default goes back to primary and the retrain does not run")
        sys.exit(1)
    print("RESULT: PASS, elliptic stays primary")


if __name__ == "__main__":
    main()
