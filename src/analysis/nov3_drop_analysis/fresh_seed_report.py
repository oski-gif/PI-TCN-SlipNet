"""The pre-registered fresh-seed report (results.md, "Pre-registered, measurement only: noslip pass on
fresh seeds", its elliptic addendum, and the Nov 6 amendment).

For every controller whose fresh-seed run is complete: held at 5 cm and 10 cm with Wilson intervals,
fell and slipped counts, per-object counts, whether the dev-set figure falls inside the fresh-seed
interval, mean and SD across the three RL seeds, a chi-square test across seeds, and paired exact
McNemar tests of each RL seed against hand-tuned A and of the fixed grip against each controller.
A run that isn't complete is only counted, never read. Prints Markdown ready to paste.

Run from the repo root:
    venv/Scripts/python.exe src/analysis/nov3_drop_analysis/fresh_seed_report.py
"""
import json
import math
import os
import statistics as st

from scipy import stats

FRESH = "Data/diag_fresh_seeds"
DEV = "Data/diag_contact_physics"
N = 1005
CONTROLLERS = [("seed1", "RL seed 1"), ("seed2", "RL seed 2"), ("seed3", "RL seed 3"),
               ("handA", "Hand-tuned A"), ("grip_TIRP_fast", "Fixed grip")]
SETTINGS = [("elliptic_impratio10", "elliptic"), ("noslip20", "noslip")]
# Dev-set files for the same controller and setting, where a complete one exists.
DEV_FILES = {
    ("seed1", "elliptic_impratio10"): f"{DEV}/seed1_dev150_elliptic_impratio10.jsonl",
    ("seed2", "elliptic_impratio10"): f"{DEV}/seed2_dev150_elliptic_impratio10.jsonl",
    ("seed1", "noslip20"): f"{DEV}/seed1_dev150_noslip20.jsonl",
    ("seed2", "noslip20"): f"{DEV}/seed2_dev150_noslip20.jsonl",
    ("seed3", "noslip20"): f"{DEV}/seed3_dev150_noslip20.jsonl",
    ("handA", "noslip20"): f"{DEV}/handA_dev150_noslip20.jsonl",
}
DEV_GRIP_NOSLIP = ("Data/diag_scripted_grip/physics_on_grip.jsonl", "TIRP_fast")


def held(r, cm=5):
    return (not r["slipped"]) and r["formation_max_fall"] <= cm / 100.0


def wilson(k, n, z=1.959964):
    p = k / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    return 100 * k / n, 100 * (c - m) / d, 100 * (c + m) / d


def mcnemar(b, c):
    return 1.0 if b + c == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))


def fmt_p(p):
    return "p < 0.0001" if p < 0.0001 else (f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}")


def line_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def load(path, grip=None):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if grip and r.get("grip") != grip:
                    continue
                rows[(r["object"], int(r["seed"]))] = r
    return rows


def main():
    complete = {}
    print("## Fresh-seed results (pre-registered)\n")
    for tag, name in CONTROLLERS:
        for suffix, setting in SETTINGS:
            path = f"{FRESH}/{tag}_{suffix}.jsonl"
            n_lines = line_count(path)
            if n_lines < N:
                print(f"- **{name}, {setting}:** not complete ({n_lines} of {N} rows), not read.")
                continue
            rows = load(path)
            if len(rows) != N:
                print(f"- **{name}, {setting}:** {len(rows)} unique trials, expected {N}; not reported.")
                continue
            complete[(tag, suffix)] = rows
            k5 = sum(held(r) for r in rows.values())
            k10 = sum(held(r, 10) for r in rows.values())
            fell = sum(r["formation_max_fall"] > 0.05 for r in rows.values())
            slipped = sum(r["formation_max_fall"] <= 0.05 and r["slipped"] for r in rows.values())
            p5, lo5, hi5 = wilson(k5, N)
            p10, lo10, hi10 = wilson(k10, N)
            dev_note = ""
            dev_path = DEV_FILES.get((tag, suffix))
            dev_rows = None
            if dev_path and os.path.exists(dev_path) and line_count(dev_path) >= 150:
                dev_rows = load(dev_path)
            elif tag == "grip_TIRP_fast" and suffix == "noslip20":
                dev_rows = load(*DEV_GRIP_NOSLIP)
            if dev_rows and len(dev_rows) == 150:
                d5 = 100 * sum(held(r) for r in dev_rows.values()) / 150
                d10 = 100 * sum(held(r, 10) for r in dev_rows.values()) / 150
                in5 = "inside" if lo5 <= d5 <= hi5 else "OUTSIDE"
                in10 = "inside" if lo10 <= d10 <= hi10 else "OUTSIDE"
                dev_note = (f" Dev figure {d5:.1f}% at 5 cm is {in5} the interval; {d10:.1f}% at 10 cm is {in10} "
                            f"({lo10:.2f}-{hi10:.2f}).")
            else:
                dev_note = " No complete dev-set figure for this setting."
            print(f"- **{name}, {setting}:** {k5} of {N} held at 5 cm ({p5:.1f}%, {lo5:.1f}-{hi5:.1f}), {k10} at 10 cm "
                  f"({p10:.1f}%, {lo10:.1f}-{hi10:.1f}); {fell} fell past 5 cm, {slipped} slipped.{dev_note}")
            objects = sorted({k[0] for k in rows})
            per = ", ".join(f"{o.replace('_', ' ')} {sum(held(rows[k]) for k in rows if k[0] == o)}" for o in objects)
            print(f"  - Held of 67 per object: {per}.")

    for suffix, setting in SETTINGS:
        seeds = [complete.get((f"seed{i}", suffix)) for i in (1, 2, 3)]
        if all(seeds):
            vals = [100 * sum(held(r) for r in s.values()) / N for s in seeds]
            vals10 = [100 * sum(held(r, 10) for r in s.values()) / N for s in seeds]
            chi2, p, _, _ = stats.chi2_contingency([[sum(held(r) for r in s.values()), N - sum(held(r) for r in s.values())]
                                                    for s in seeds])
            print(f"\n- **Three RL seeds, {setting}:** {', '.join(f'{v:.1f}%' for v in vals)} at 5 cm, mean {st.mean(vals):.1f}, "
                  f"SD {st.stdev(vals):.1f}; at 10 cm mean {st.mean(vals10):.1f}, SD {st.stdev(vals10):.1f}. "
                  f"Chi-square across seeds {chi2:.1f}, {fmt_p(p)}.")
        hand = complete.get(("handA", suffix))
        if hand:
            for i in (1, 2, 3):
                s = complete.get((f"seed{i}", suffix))
                if s:
                    keys = sorted(set(s) & set(hand))
                    b = sum(held(s[k]) and not held(hand[k]) for k in keys)
                    c = sum(held(hand[k]) and not held(s[k]) for k in keys)
                    print(f"- {setting}, RL seed {i} vs. hand-tuned A at 5 cm: held only by RL {b}, only by hand-tuned {c}, "
                          f"{fmt_p(mcnemar(b, c))}.")
        grip = complete.get(("grip_TIRP_fast", suffix))
        if grip:
            for tag, name in CONTROLLERS[:4]:
                other = complete.get((tag, suffix))
                if other:
                    keys = sorted(set(grip) & set(other))
                    b = sum(held(grip[k]) and not held(other[k]) for k in keys)
                    c = sum(held(other[k]) and not held(grip[k]) for k in keys)
                    print(f"- {setting}, fixed grip vs. {name} at 5 cm: held only by the grip {b}, only by {name} {c}, "
                          f"{fmt_p(mcnemar(b, c))}.")


if __name__ == "__main__":
    main()
