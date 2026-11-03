"""
Builds the paper's analysis figures from recorded result files. Re-run it
whenever a new result lands:

    python src/analysis/make_figures.py

Writes one PNG plus one CSV of the plotted numbers per figure to
reports/figures/. Plain numpy/json/matplotlib on purpose -- no torch, no
MuJoCo, nothing that touches the GPU -- so it is safe to run during evals.

What this script refuses to do, because each one would quietly misstate a
result:
- Use a file that fails its provenance check. Later batches can overwrite a
  results file, and running batches leave partial ones, so each input is
  checked against the trial count, held count, horizon, and scoring rule it
  should have. A file that fails is left OUT with a printed notice.
- Estimate a missing result. No file means no mark on the chart.
- Compare across horizons. Conditions are grouped by total steps, because the
  same trials score differently at each. A hand-tuned reference line appears
  beside RL only when its horizon matches RL's and it uses RL's drop rule.
- Plot one RL seed as "the RL result". RL is always shown per seed, with the
  mean and SD across seeds.

To add a result when it lands: if its path below is already right, just re-run.
Otherwise edit that entry's "path" in INPUTS, and set expected_held once the
number has been logged.
"""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "figures"

COMPARABILITY_SECTION = ('results.md, section "Baseline comparability check: 40.0% and 58.6% are '
                         'NOT comparable -- the gap is mainly a max_steps difference, not '
                         'sampling noise"')
PREREGISTERED_SECTION = ('results.md, section "Pre-registered evaluation rules for all remaining '
                         'RL comparisons (written before any of these results exist)"')

# ---------------------------------------------------------------------------
# RESULTS.MD-ONLY NUMBERS -- everything here was hand-copied from
# src/training/results.md because no result file records it. Nothing else in
# this script comes from results.md. The rl_seed1 row is drawn with a hollow
# marker so it can't be mistaken for recorded per-trial data, and is excluded
# from the per-object figure.
# ---------------------------------------------------------------------------
BEST_SEED_LABEL = "RL seed 1: best of 3 seeds, deterministic"
RESULTS_MD_ONLY = {
    "rl_seed1": {
        "label": BEST_SEED_LABEL,
        "held": 29,
        "n": 45,
        "slipped": 5,
        "dropped": 11,
        "seeds": "0-2",
        "policy": "Data/ppo_slip_policy_latest.zip (300,224 steps, deterministic)",
        "source": 'results.md, section "THE real RL closed-loop number -- 64.4%, '
                  'with a clean win pattern and one new, concentrated failure mode"',
    },
    "rl_training_launches": {
        "seed 1": 8,
        "seed 2": 2,
        "seed 3": 1,
        "source": 'results.md, section "Seed3 real result: 17.8% -- the 3-seed '
                  "baseline mean+-std is now locked in, and it's nowhere near "
                  '70%", CORRECTION note',
    },
    # Hand-tuned --max-steps counts TOTAL steps; RL's counts steps AFTER settle.
    # Ramp 500 + settle 300 come before monitoring; the pull ramp takes 500 more.
    "horizons": {
        "hand_2500": {"flag": "--max-steps 2500", "total": 2500, "monitored": 1700,
                      "full_pull": 1200, "source": COMPARABILITY_SECTION},
        "hand_1500": {"flag": "--max-steps 1500", "total": 1500, "monitored": 700,
                      "full_pull": 200, "source": COMPARABILITY_SECTION},
        "rl_1500": {"flag": "--max-steps 1500 (after settle)", "total": 2300, "monitored": 1500,
                    "full_pull": 1000, "source": COMPARABILITY_SECTION},
        "hand_2300": {"flag": "--max-steps 2300", "total": 2300, "monitored": 1500,
                      "full_pull": 1000, "source": PREREGISTERED_SECTION},
    },
    # Locked set: seeds 0-2 already helped compare and choose policies, so
    # seeds 3-66 are the primary figure for anything set beside RL.
    "locked_primary_seeds": {"seeds": (3, 66), "source": PREREGISTERED_SECTION},
}
HORIZONS = RESULTS_MD_ONLY["horizons"]
LOCKED_PRIMARY_SEEDS = RESULTS_MD_ONLY["locked_primary_seeds"]["seeds"]
RL_TOTAL_STEPS = HORIZONS["rl_1500"]["total"]

DROP_TAG = {True: "drops count as failures", False: "no formation-drop check"}
DROP_TAG_SHORT = {True: "drops = failure", False: "no drop check"}

# The 2300-step and locked RL entries are slots: those runs haven't happened and
# their paths are placeholders, so point them at the real files when they exist.
# A 2300-step file only counts if every row has dropped_during_formation, the
# same drop test RL uses, because that is what makes it comparable with RL.
INPUTS = [
    {"key": "a2300", "label": "Hand-tuned Model A", "group": "hand_a", "fmt": "jsonl",
     "path": "Data/handtuned_modelA_2300step_results.jsonl",
     "expected_trials": 1005, "expected_held": None, "horizon": "hand_2300", "counts_drops": True,
     "primary_seeds": LOCKED_PRIMARY_SEEDS, "expected_config": {"max_steps": 2300}},
    {"key": "b2300", "label": "Hand-tuned Model B", "group": "hand_b", "fmt": "jsonl",
     "path": "Data/handtuned_modelB_2300step_results.jsonl",
     "expected_trials": 1005, "expected_held": None, "horizon": "hand_2300", "counts_drops": True,
     "primary_seeds": LOCKED_PRIMARY_SEEDS, "expected_config": {"max_steps": 2300}},
    {"key": "rl_seed1_locked", "label": BEST_SEED_LABEL, "group": "rl", "fmt": "jsonl",
     "column_label": "RL seed 1:\nbest of 3 seeds,\ndeterministic",
     "path": "Data/rl_seed1_1005trial_results.jsonl",
     "expected_trials": 1005, "expected_held": None, "horizon": "rl_1500", "counts_drops": True,
     "primary_seeds": LOCKED_PRIMARY_SEEDS,
     "expected_config": {"max_steps": 1500, "deterministic": True}},
    {"key": "rl_seed2", "label": "RL seed 2", "group": "rl", "fmt": "log",
     "path": "Data/_seedcheck_eval_output.log",
     "expected_trials": 45, "expected_held": 16, "horizon": "rl_1500", "counts_drops": True},
    {"key": "rl_seed3", "label": "RL seed 3", "group": "rl", "fmt": "log",
     "path": "Data/_seed3_eval_output.log",
     "expected_trials": 45, "expected_held": 8, "horizon": "rl_1500", "counts_drops": True},
    {"key": "a45", "label": "Hand-tuned Model A", "group": "hand_a", "fmt": "jsonl",
     "path": "Data/closed_loop_scenes/batch_results.jsonl",
     "expected_trials": 45, "expected_held": 18, "horizon": "hand_2500", "counts_drops": False},
    {"key": "b45", "label": "Hand-tuned Model B", "group": "hand_b", "fmt": "jsonl",
     "path": "Data/closed_loop_scenes/batch_results_modelb.jsonl",
     "expected_trials": 45, "expected_held": 15, "horizon": "hand_2500", "counts_drops": False},
    {"key": "a1005", "label": "Hand-tuned Model A", "group": "hand_a", "fmt": "jsonl",
     "path": "Data/baseline_modelA_1000trial_results.jsonl",
     "expected_trials": 1005, "expected_held": 589, "horizon": "hand_1500", "counts_drops": False},
    {"key": "b1005", "label": "Hand-tuned Model B", "group": "hand_b", "fmt": "jsonl",
     "path": "Data/baseline_modelB_1000trial_results.jsonl",
     "expected_trials": 1005, "expected_held": 582, "horizon": "hand_1500", "counts_drops": False},
]

# Conditions by horizon, RL's first. Figures 1 and 3 share this grouping.
HORIZON_GROUPS = [
    (2300, ["a2300", "b2300", "rl_seed1", "rl_seed2", "rl_seed3", "rl_seed1_locked"]),
    (2500, ["a45", "b45"]),
    (1500, ["a1005", "b1005"]),
]
RL_SEED_KEYS = ("rl_seed1", "rl_seed2", "rl_seed3")
RL_REFERENCE_KEYS = ("a2300", "b2300")
HOLDOUT_OBJECTS = {"cone", "cylindrical_prism", "plus_cross", "sphere"}
N_OBJECTS = 15
CORRECTION_PERIOD_STEPS = 5
# Fig 4 slot. Each file: {"model": "A", "split": "test", "seed": 0, "f1": ..., "accuracy": ...}
STATIC_METRICS_GLOB = "Data/static_metrics/*.json"
MIN_STATIC_SEEDS = 2

# Palette checked with the dataviz validator (all pairs, light mode, passes).
# Aqua sits under 3:1 contrast, which is why every RL mark also has a text label.
SERIES_COLOR = {"hand_a": "#2a78d6", "hand_b": "#eb6834", "rl": "#1baf7a"}
SERIES_NAME = {"hand_a": "Hand-tuned Model A", "hand_b": "Hand-tuned Model B", "rl": "RL (PPO)"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

RL_TRIAL_RE = re.compile(
    r"^(\S+)\s+seed=(\d+)\s+held=(True|False)\s+slipped=(True|False)\s+"
    r"dropped_during_formation=(True|False)\s+max_disp=([\d.]+)\s+threshold=([\d.]+)")


# --- Loading and provenance ------------------------------------------------

def read_jsonl(path):
    """Batch or RL eval output. Returns (trials, n_corrupt, n_crashed). A
    half-written last line from a running batch counts as corrupt. A crash row
    only counts if the pair never completed, since resumable evals retry them."""
    trials, n_corrupt, crashed = [], 0, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if "error" in r:
                    crashed.add((r["object"], int(r["seed"])))
                    continue
                slipped = bool(r["slipped"])
                dropped = r.get("dropped_during_formation")
                held = not slipped and not dropped
                if "held" in r and bool(r["held"]) != held:
                    n_corrupt += 1
                    continue
                n_corrections = r.get("n_corrections")
                trials.append({
                    "object": r["object"], "seed": int(r["seed"]), "held": held, "slipped": slipped,
                    "dropped": None if dropped is None else bool(dropped),
                    "threshold": float(r["threshold"]),
                    "n_corrections": None if n_corrections is None else int(n_corrections),
                    # Hand-tuned rows say max_steps_total; RL rows say max_steps_after_settle.
                    "max_steps": r.get("max_steps_total", r.get("max_steps_after_settle")),
                    "deterministic": r.get("deterministic"),
                })
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                n_corrupt += 1
    return trials, n_corrupt, len(crashed - {(t["object"], t["seed"]) for t in trials})


def read_rl_log(path):
    """evaluate_rl_policy.py stdout. Only per-trial lines match the regex; a
    line whose held flag disagrees with its own slipped/dropped flags counts
    as corrupt."""
    trials, n_corrupt = [], 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RL_TRIAL_RE.match(line.strip())
            if not m:
                continue
            held, slipped, dropped = (m.group(i) == "True" for i in (3, 4, 5))
            if held != (not slipped and not dropped):
                n_corrupt += 1
                continue
            trials.append({
                "object": m.group(1), "seed": int(m.group(2)),
                "held": held, "slipped": slipped, "dropped": dropped,
                "threshold": float(m.group(7)), "n_corrections": None,
            })
    return trials, n_corrupt, 0


def check_provenance(spec, trials, n_corrupt, n_crashed):
    """Returns (status, problems, warnings). Only "ok" gets plotted. A batch
    that lost trials to crashes lands in "incomplete" here too; after checking
    it, set expected_trials to its real count so the smaller denominator is a
    decision and not an accident."""
    expected, n = spec["expected_trials"], len(trials)
    if n < expected:
        crash_note = f", {n_crashed} crashed" if n_crashed else ""
        corrupt_note = f", {n_corrupt} corrupt" if n_corrupt else ""
        return "incomplete", [f"{n:,} of {expected:,} rows{crash_note}{corrupt_note}"], []

    problems, warnings = [], []
    if n > expected:
        problems.append(f"{n:,} rows, expected {expected:,}; a later batch may have appended")
    if n_corrupt:
        problems.append(f"{n_corrupt} corrupt rows")

    pair_counts = Counter((t["object"], t["seed"]) for t in trials)
    n_duplicates = sum(1 for c in pair_counts.values() if c > 1)
    if n_duplicates:
        problems.append(f"{n_duplicates} duplicate (object, seed) pairs")

    object_counts = Counter(t["object"] for t in trials)
    if len(object_counts) != N_OBJECTS or len(set(object_counts.values())) != 1:
        problems.append(f"uneven objects: {len(object_counts)} objects, "
                        f"counts {sorted(set(object_counts.values()))}")

    n_held = sum(t["held"] for t in trials)
    if spec["expected_held"] is not None and n_held != spec["expected_held"]:
        problems.append(f"{n_held} held, logged result was {spec['expected_held']}; "
                        "file may have been overwritten")

    # The scoring rule the chart will state has to be the one the file used.
    n_with_drop_field = sum(t["dropped"] is not None for t in trials)
    if n_with_drop_field != (n if spec["counts_drops"] else 0):
        rule = "needs" if spec["counts_drops"] else "registered without"
        problems.append(f"{rule} the formation-drop rule, but {n_with_drop_field} of {n} rows "
                        "have dropped_during_formation")

    for field, want in spec.get("expected_config", {}).items():
        found = {t.get(field) for t in trials} - {None}
        if found - {want}:
            problems.append(f"{field} is {sorted(found, key=str)}, expected {want}")

    # Catches a file overwritten by a batch with a different horizon, which
    # can't be seen from the trial count alone. RL logs have no corrections.
    monitored = HORIZONS[spec["horizon"]]["monitored"]
    corrections = [t["n_corrections"] for t in trials if t["n_corrections"] is not None]
    if corrections:
        implied = max(corrections) * CORRECTION_PERIOD_STEPS
        if implied > monitored:
            problems.append(f"corrections imply at least {implied} monitored steps, "
                            f"horizon has {monitored}")
        elif implied < 0.8 * monitored:
            warnings.append(f"corrections imply about {implied} monitored steps, "
                            f"horizon has {monitored}")

    return ("ok" if not problems else "rejected"), problems, warnings


def new_condition(spec):
    return dict(spec, source=spec.get("path", ""), status="absent", detail="", n=None, held=None,
                slipped=None, dropped=None, seeds_shown="", per_object={}, thresholds={},
                n_all=None, held_all=None, n_primary=None, held_primary=None,
                n_zero_corrections=None, held_zero_corrections=None)


def seed_span(trials):
    seeds = [t["seed"] for t in trials]
    return f"{min(seeds)}-{max(seeds)}"


def summarize(cond, trials):
    """Adds the counts every figure uses. Slots with primary_seeds show those
    seeds only; the all-seed and seeds 3-66 counts stay in the CSV either way."""
    first, last = LOCKED_PRIMARY_SEEDS
    primary = [t for t in trials if first <= t["seed"] <= last]
    cond["n_all"], cond["held_all"] = len(trials), sum(t["held"] for t in trials)
    if primary:
        cond["n_primary"], cond["held_primary"] = len(primary), sum(t["held"] for t in primary)

    shown = primary if cond.get("primary_seeds") else trials
    per_object = defaultdict(lambda: [0, 0])
    for t in shown:
        per_object[t["object"]][0] += t["held"]
        per_object[t["object"]][1] += 1
    cond["per_object"] = {obj: tuple(v) for obj, v in per_object.items()}
    cond["seeds_shown"] = seed_span(shown)
    cond["n"] = len(shown)
    cond["held"] = sum(t["held"] for t in shown)
    cond["slipped"] = sum(t["slipped"] for t in shown)
    if cond["counts_drops"]:
        cond["dropped"] = sum(t["dropped"] for t in shown)
    if all(t["n_corrections"] is not None for t in shown):
        zero = [t for t in shown if t["n_corrections"] == 0]
        cond["n_zero_corrections"] = len(zero)
        cond["held_zero_corrections"] = sum(t["held"] for t in zero)
    cond["thresholds"] = defaultdict(list)
    for t in trials:
        cond["thresholds"][t["object"]].append(t["threshold"])


def load_conditions():
    conds = {}
    for spec in INPUTS:
        cond = new_condition(spec)
        conds[spec["key"]] = cond
        path = REPO_ROOT / spec["path"]
        if not path.exists():
            print(f"[input] {spec['key']:<16} ABSENT    no file at {spec['path']}")
            continue

        reader = read_jsonl if spec["fmt"] == "jsonl" else read_rl_log
        trials, n_corrupt, n_crashed = reader(path)
        status, problems, warnings = check_provenance(spec, trials, n_corrupt, n_crashed)
        cond["status"], cond["detail"] = status, "; ".join(problems)
        if status == "ok":
            summarize(cond, trials)
            print(f"[input] {spec['key']:<16} OK        {cond['held']}/{cond['n']} held, seeds "
                  f"{cond['seeds_shown']}, {horizon_of(cond)['total']} total steps  ({spec['path']})")
        else:
            print(f"[input] {spec['key']:<16} EXCLUDED  {status}: {cond['detail']}  ({spec['path']})")
        for w in warnings:
            print(f"[input] {spec['key']:<16} WARNING   {w}")

    md = RESULTS_MD_ONLY["rl_seed1"]
    conds["rl_seed1"] = dict(
        new_condition({"key": "rl_seed1", "label": md["label"], "group": "rl",
                       "horizon": "rl_1500", "counts_drops": True}),
        status="results_md", n=md["n"], held=md["held"], slipped=md["slipped"],
        dropped=md["dropped"], seeds_shown=md["seeds"], source=md["source"],
        n_all=md["n"], held_all=md["held"])
    print(f"[input] {'rl_seed1':<16} MD ONLY   {md['label']}: {md['held']}/{md['n']} held, no "
          f"per-trial file; from {md['source']}")
    check_thresholds(conds)
    return conds


def check_thresholds(conds):
    """Slip is scored against each object's threshold, so a threshold that
    differs between inputs would mean they aren't scoring the same test."""
    by_object = defaultdict(list)
    for cond in conds.values():
        for obj, values in cond["thresholds"].items():
            by_object[obj].extend(values)
    # 1e-4 because the RL logs print thresholds to 4 decimal places.
    mismatched = [o for o, v in by_object.items() if max(v) - min(v) > 1e-4]
    if mismatched:
        print(f"[input] WARNING: slip thresholds differ across inputs for: {sorted(mismatched)}")
    else:
        print(f"[input] slip thresholds agree across inputs for all {len(by_object)} objects")


def is_plotted(cond):
    return cond["status"] in ("ok", "results_md")


def horizon_of(cond):
    return HORIZONS[cond["horizon"]]


def horizon_summary(total):
    horizon = next(h for h in HORIZONS.values() if h["total"] == total)
    return f"{horizon['monitored']:,} monitored, {horizon['full_pull']:,} at full pull"


def wilson_interval(k, n, z=1.959963985):
    """Wilson 95% binomial interval as (low, high) fractions. Used over the
    normal approximation because that one misbehaves at small n and near
    0%/100%, and most rows here are n=45."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate_stats(cond):
    """(rate, low, high) in percent."""
    lo, hi = wilson_interval(cond["held"], cond["n"])
    return 100 * cond["held"] / cond["n"], 100 * lo, 100 * hi


def absent_text(cond):
    if cond["status"] == "absent":
        return "not run yet"
    return f"{cond['status']}: {cond['detail']}"


def blank_if_none(value, fmt="{:.2f}"):
    return "" if value is None else fmt.format(value)


def share_pct(part, whole):
    return "" if part is None or not whole else f"{100 * part / whole:.1f}"


# --- Drawing helpers -------------------------------------------------------

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "savefig.dpi": 200,
    })


def style_percent_axis(ax, n_rows):
    ax.set_xlim(0, 100)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(0, 101, 20))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", length=0, colors=SECONDARY, labelsize=9)


def draw_interval(ax, y, rate, lo, hi, color, hollow=False, marker="o"):
    ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round",
            zorder=3, clip_on=False)
    ax.plot([rate], [y], marker=marker, markersize=9, zorder=4, clip_on=False,
            markerfacecolor=SURFACE if hollow else color,
            markeredgecolor=color if hollow else SURFACE,
            markeredgewidth=2 if hollow else 1.5)


def draw_row_label(ax, y, title, detail):
    transform = ax.get_yaxis_transform()
    ax.text(-0.02, y, title, transform=transform, ha="right", va="bottom", color=INK)
    ax.text(-0.02, y + 0.04, detail, transform=transform, ha="right", va="top",
            color=SECONDARY, fontsize=8.5)


def draw_row_value(ax, y, value, tag=""):
    """The value, and under it the scoring rule when rows in a panel can differ."""
    transform = ax.get_yaxis_transform()
    if value:
        ax.text(1.02, y, value, transform=transform, ha="left", va="bottom" if tag else "center",
                color=INK, fontsize=9.5)
    if tag:
        ax.text(1.02, y + 0.04, tag, transform=transform, ha="left", va="top",
                color=SECONDARY, fontsize=8.5)


def draw_header(fig, title, subtitle):
    # Spaced in inches so the gaps are the same whatever the figure's height.
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.12 / height, title, fontsize=14, fontweight="semibold", color=INK, va="top")
    fig.text(0.02, 1 - 0.44 / height, subtitle, fontsize=9.5, color=SECONDARY, va="top")


def draw_legend(fig, handles):
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.68 / fig.get_figheight()),
               ncol=len(handles), frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.4,
               labelcolor=SECONDARY)


def draw_footer(fig, text):
    fig.text(0.02, 0.12 / fig.get_figheight(), text, fontsize=8.5, color=SECONDARY, va="bottom",
             linespacing=1.4)


def dot_handle(color, label, hollow=False, marker="o"):
    return Line2D([0], [0], color=color, linewidth=2, marker=marker, markersize=8, label=label,
                  markerfacecolor=SURFACE if hollow else color,
                  markeredgecolor=color if hollow else SURFACE)


def save_figure(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[output] {path.relative_to(REPO_ROOT)}")


def write_csv(name, header, rows):
    path = OUT_DIR / f"{name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[output] {path.relative_to(REPO_ROOT)}")


# --- Figure 1: success by condition ----------------------------------------

def fig1_closed_loop(conds):
    """One panel per horizon, so a rate only sits beside others that faced the
    same test. n still differs within a panel, so every row states it."""
    n_rows = sum(len(keys) for _, keys in HORIZON_GROUPS)
    fig = plt.figure(figsize=(10.5, 0.62 * n_rows + 3.4))
    height = fig.get_figheight()
    grid = fig.add_gridspec(len(HORIZON_GROUPS), 1, height_ratios=[len(k) for _, k in HORIZON_GROUPS],
                            left=0.30, right=0.78, top=1 - 1.35 / height, bottom=0.85 / height,
                            hspace=0.45)
    rows = []
    for i, (total, keys) in enumerate(HORIZON_GROUPS):
        ax = fig.add_subplot(grid[i])
        style_percent_axis(ax, len(keys))
        suffix = "" if total == RL_TOTAL_STEPS else " (does not match RL)"
        ax.set_title(f"{total:,} total steps: {horizon_summary(total)}{suffix}", loc="left",
                     fontsize=10, fontweight="semibold", color=INK, pad=8)
        for y, key in enumerate(keys):
            cond = conds[key]
            tag = DROP_TAG[cond["counts_drops"]]
            if is_plotted(cond):
                rate, lo, hi = rate_stats(cond)
                draw_row_label(ax, y, cond["label"], f"n={cond['n']:,}, seeds {cond['seeds_shown']}")
                draw_interval(ax, y, rate, lo, hi, SERIES_COLOR[cond["group"]],
                              hollow=cond["status"] == "results_md")
                draw_row_value(ax, y, f"{rate:.1f}%  {cond['held']:,}/{cond['n']:,}", tag)
            else:
                rate = lo = hi = None
                first, last = LOCKED_PRIMARY_SEEDS
                planned = f"locked set, seeds {first}-{last}" if cond.get("primary_seeds") else ""
                draw_row_label(ax, y, cond["label"], planned or "no result yet")
                ax.text(1, y, absent_text(cond), color=SECONDARY, va="center", fontsize=9,
                        style="italic")
                draw_row_value(ax, y, "", tag)
            horizon = horizon_of(cond)
            is_hand_tuned = cond["group"] != "rl"
            # Only hand-tuned trials run on after a drop. A default RL trial ends
            # there with slipped=False, so n - slipped would count its drops as holds.
            ignored_drops = (cond["n"] - cond["slipped"]
                             if is_hand_tuned and cond["dropped"] is not None else None)
            n_zero = cond["n_zero_corrections"] if is_hand_tuned else None
            held_zero = cond["held_zero_corrections"] if is_hand_tuned else None
            rows.append([total, cond["label"], cond["status"], cond["detail"],
                         blank_if_none(cond["n"], "{}"), blank_if_none(cond["held"], "{}"),
                         blank_if_none(rate), blank_if_none(lo), blank_if_none(hi),
                         cond["seeds_shown"], blank_if_none(cond["slipped"], "{}"),
                         blank_if_none(cond["dropped"], "{}"), cond["counts_drops"],
                         blank_if_none(ignored_drops, "{}"),
                         blank_if_none(cond["n_all"], "{}"), blank_if_none(cond["held_all"], "{}"),
                         blank_if_none(cond["n_primary"], "{}"),
                         blank_if_none(cond["held_primary"], "{}"),
                         blank_if_none(n_zero, "{}"), share_pct(n_zero, cond["n"]),
                         blank_if_none(held_zero, "{}"), share_pct(held_zero, cond["held"]),
                         horizon["flag"], horizon["total"], horizon["monitored"],
                         horizon["full_pull"], horizon["source"], cond["source"]])

    draw_header(fig, "Closed-loop grasp success by horizon",
                "Dot = share of trials held. Line = Wilson 95% interval. Compare rows only within "
                "a panel: each horizon is a different test.")
    draw_legend(fig, [dot_handle(SERIES_COLOR[g], SERIES_NAME[g]) for g in SERIES_COLOR]
                + [dot_handle(SERIES_COLOR["rl"], "RL, results.md only (no per-trial file)", hollow=True)])
    draw_footer(fig, "Horizons: results.md, \"Baseline comparability check\" and \"Pre-registered "
                     "evaluation rules\". Rows with no mark have no accepted result; nothing is estimated.\n"
                     "Without a formation-drop check, a trial already dropped by the end of settle "
                     "could score as held. This has not been measured.")
    save_figure(fig, "fig1_closed_loop_success")
    write_csv("fig1_closed_loop_success",
              ["total_steps_group", "condition", "status", "status_detail", "n", "held", "rate_pct",
               "wilson_lo_pct", "wilson_hi_pct", "seeds_shown", "slipped",
               "dropped_during_formation", "counts_formation_drops_as_failures",
               "held_if_drops_ignored", "n_all_seeds", "held_all_seeds", "n_seeds_3_66",
               "held_seeds_3_66", "zero_correction_trials", "zero_correction_share_pct",
               "held_zero_correction_trials", "held_zero_correction_share_pct", "max_steps_flag",
               "total_steps", "monitored_steps", "full_pull_steps", "horizon_source", "data_source"],
              rows)


# --- Figure 2: RL seed spread ----------------------------------------------

def fig2_rl_seed_spread(conds):
    """Uses the 45-trial eval of each seed only, so all three seeds and the
    mean sit on one n. A hand-tuned reference line is drawn only when its
    horizon and drop rule match RL's; until those runs exist, there is none."""
    seeds = [conds[k] for k in RL_SEED_KEYS if is_plotted(conds[k])]
    if len(seeds) < 2:
        print(f"[fig2] SKIPPED: needs at least 2 RL seeds, found {len(seeds)}")
        return

    references = sorted(
        (conds[k] for k in RL_REFERENCE_KEYS if conds[k]["status"] == "ok"
         and horizon_of(conds[k])["total"] == RL_TOTAL_STEPS and conds[k]["counts_drops"]),
        key=lambda c: c["held"] / c["n"])
    if not references:
        print("[fig2] No reference line: no hand-tuned run at RL's horizon with the drop rule yet")

    rates = np.array([100 * c["held"] / c["n"] for c in seeds])
    mean, sd = float(rates.mean()), float(rates.std(ddof=1))
    color = SERIES_COLOR["rl"]
    horizon = HORIZONS["rl_1500"]

    fig = plt.figure(figsize=(10.0, 5.0))
    ax = fig.add_axes([0.30, 0.20, 0.50, 0.50 if references else 0.56])
    style_percent_axis(ax, len(seeds) + 1)

    def csv_row(item, kind, cond, rate, lo, hi, sd_pct, source):
        row_horizon = horizon if cond is None else horizon_of(cond)
        return [item, kind, "" if cond is None else cond["n"], "" if cond is None else cond["held"],
                f"{rate:.2f}", blank_if_none(lo), blank_if_none(hi), blank_if_none(sd_pct),
                "" if cond is None else cond["seeds_shown"], True, row_horizon["flag"],
                row_horizon["total"], row_horizon["monitored"], row_horizon["full_pull"],
                row_horizon["source"], source]

    rows = []
    for y, cond in enumerate(seeds):
        rate, lo, hi = rate_stats(cond)
        draw_row_label(ax, y, cond["label"], f"n={cond['n']}, seeds {cond['seeds_shown']}")
        draw_interval(ax, y, rate, lo, hi, color, hollow=cond["status"] == "results_md")
        draw_row_value(ax, y, f"{rate:.1f}%  {cond['held']}/{cond['n']}")
        rows.append(csv_row(cond["label"], "seed", cond, rate, lo, hi, None, cond["source"]))

    y_mean = len(seeds)
    draw_row_label(ax, y_mean, f"Mean of {len(seeds)} seeds", "line = ±1 SD across seeds, not trials")
    draw_interval(ax, y_mean, mean, max(0.0, mean - sd), min(100.0, mean + sd), color, marker="D")
    draw_row_value(ax, y_mean, f"{mean:.1f}%  SD {sd:.1f}")
    rows.append(csv_row(f"Mean of {len(seeds)} seeds", "mean", None, mean, None, None, sd,
                        "computed from the seed rows; SD uses n-1"))

    # Labels fan out away from each other so they can't collide.
    aligns = ["right", "left"] if len(references) == 2 else ["left"]
    for cond, align in zip(references, aligns):
        rate, lo, hi = rate_stats(cond)
        ax.axvline(rate, color=SERIES_COLOR[cond["group"]], linewidth=1.5, zorder=2)
        offset = 0.8 if align == "left" else -0.8
        ax.text(rate + offset, -0.60,
                f"{cond['label']} {rate:.1f}%\nn={cond['n']:,}, seeds {cond['seeds_shown']}, "
                f"{DROP_TAG_SHORT[True]}",
                ha=align, va="bottom", fontsize=8.5, color=INK)
        rows.append(csv_row(cond["label"], "reference", cond, rate, lo, hi, None, cond["source"]))

    launches = RESULTS_MD_ONLY["rl_training_launches"]
    draw_header(fig, "RL success varies widely across training seeds",
                f"{horizon['total']:,} total steps: {horizon_summary(horizon['total'])}. "
                f"{DROP_TAG[True].capitalize()}. Dot = share of 45 trials held; line = Wilson 95% interval.")
    handles = [dot_handle(color, "RL seed"), dot_handle(color, "RL seed, results.md only", hollow=True),
               dot_handle(color, "Mean of seeds", marker="D")]
    handles += [Line2D([0], [0], color=SERIES_COLOR[c["group"]], linewidth=1.5, label=c["label"])
                for c in references]
    draw_legend(fig, handles)
    reference_note = ("Reference lines use the same horizon and drop rule as RL, on more trials."
                      if references else
                      "No reference line: no hand-tuned run exists yet at this horizon with this drop rule.")
    draw_footer(fig, "Same reward, observations, and hyperparameters, but training procedure differed: "
                     f"seeds 1, 2, and 3 took {launches['seed 1']}, {launches['seed 2']}, and "
                     f"{launches['seed 3']} process launches.\n{reference_note}")
    save_figure(fig, "fig2_rl_seed_spread")
    write_csv("fig2_rl_seed_spread",
              ["item", "kind", "n", "held", "rate_pct", "wilson_lo_pct", "wilson_hi_pct", "sd_pct",
               "seeds_shown", "counts_formation_drops_as_failures", "max_steps_flag", "total_steps",
               "monitored_steps", "full_pull_steps", "horizon_source", "data_source"], rows)


# --- Figure 3: per-object heat table ---------------------------------------

def fig3_per_object(conds):
    """Per-trial inputs only, grouped by horizon as in Figure 1. Every cell
    prints its count because cells range from 3 to 67 trials: color alone
    would make 2/3 look as solid as 44/67."""
    groups = [(total, [conds[k] for k in keys if conds[k]["status"] == "ok"])
              for total, keys in HORIZON_GROUPS]
    groups = [(total, group) for total, group in groups if group]
    if not groups:
        print("[fig3] SKIPPED: no per-trial input passed its provenance check")
        return

    # Sort by the largest sample, since 3-trial rates are too coarse to rank on.
    anchor = max((c for _, group in groups for c in group), key=lambda c: c["n"])
    objects = sorted(anchor["per_object"],
                     key=lambda o: (-anchor["per_object"][o][0] / anchor["per_object"][o][1], o))
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)

    n_columns = sum(len(group) for _, group in groups)
    gap = 0.35
    width = n_columns + gap * (len(groups) - 1)
    fig = plt.figure(figsize=(1.05 * width + 3.2, 0.36 * len(objects) + 3.9))
    height = fig.get_figheight()
    bottom, top = 1.20 / height, 1 - 2.25 / height
    ax = fig.add_axes([0.20, bottom, 0.76, top - bottom])
    ax.set_xlim(0, width)
    ax.set_ylim(len(objects), 0)
    ax.axis("off")

    for y, obj in enumerate(objects):
        label = f"{obj} (holdout)" if obj in HOLDOUT_OBJECTS else obj
        ax.text(-0.12, y + 0.5, label, ha="right", va="center", color=INK, fontsize=9.5)

    rows, x = [], 0.0
    for total, group in groups:
        ax.plot([x + 0.04, x + len(group) - 0.04], [-2.45, -2.45], color=BASELINE, linewidth=0.8,
                clip_on=False)
        ax.text(x + len(group) / 2, -2.55, f"{total:,} total steps\n{horizon_summary(total)}",
                ha="center", va="bottom", fontsize=8.5, fontweight="semibold", color=INK,
                linespacing=1.25)
        for cond in group:
            name = cond.get("column_label", cond["label"].replace("Hand-tuned ", "Hand-tuned\n"))
            ax.text(x + 0.5, -0.15,
                    f"{name}\nn={cond['n']:,}\nseeds {cond['seeds_shown']}\n"
                    f"{DROP_TAG_SHORT[cond['counts_drops']]}",
                    ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.25)
            for y, obj in enumerate(objects):
                held, n = cond["per_object"][obj]
                rate = held / n
                fill = cmap(rate)
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor=SURFACE, linewidth=2))
                ax.text(x + 0.5, y + 0.5, f"{held}/{n}", ha="center", va="center", fontsize=9,
                        color=readable_ink(fill))
                rows.append([obj, obj in HOLDOUT_OBJECTS, cond["label"], cond["n"], cond["seeds_shown"],
                             horizon_of(cond)["total"], cond["counts_drops"], held, n, f"{100 * rate:.1f}"])
            x += 1
        x += gap

    draw_header(fig, "Grasp success per object, by horizon",
                "Cells show trials held / trials run, sorted by the largest sample. Compare within "
                "a horizon only. 3-trial cells can only read 0, 33, 67, or 100%.")
    cax = fig.add_axes([0.20, 0.72 / height, 0.24, 0.14 / height])
    colorbar = fig.colorbar(ScalarMappable(Normalize(0, 100), cmap), cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 50, 100])
    colorbar.ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    colorbar.ax.tick_params(length=0, labelsize=8.5, colors=SECONDARY)
    colorbar.outline.set_visible(False)
    fig.text(0.46, 0.79 / height, "share of trials held", fontsize=8.5, color=SECONDARY, va="center")
    draw_footer(fig, "Drops = failure: a formation drop counts against the trial. No drop check: a "
                     "trial already dropped by the end of settle could score as held; not measured.")
    save_figure(fig, "fig3_per_object_success")
    write_csv("fig3_per_object_success",
              ["object", "holdout", "condition", "condition_n", "seeds_shown", "total_steps",
               "counts_formation_drops_as_failures", "held", "trials", "rate_pct"], rows)

    left_out = [k for _, keys in HORIZON_GROUPS for k in keys
                if conds[k]["status"] not in ("ok", "results_md")]
    print(f"[fig3] Not shown, no accepted per-trial data: {left_out}")
    print(f"[fig3] Not shown: {BEST_SEED_LABEL}, n=45. Its only per-object table is in results.md.")


def readable_ink(fill):
    """Whichever of dark ink or white has more contrast on this fill (WCAG)."""
    def luminance(color):
        channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                    for c in to_rgb(color)]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    fill_l, ink_l = luminance(fill), luminance(INK)
    return INK if (fill_l + 0.05) / (ink_l + 0.05) >= 1.05 / (fill_l + 0.05) else "#ffffff"


# --- Figure 4: static metrics slot -----------------------------------------

def load_static_metrics():
    """Returns {(model, split): {seed: {"f1": ..., "accuracy": ...}}}, leaving
    out anything malformed or repeated, with a notice."""
    groups = defaultdict(dict)
    for path in sorted(REPO_ROOT.glob(STATIC_METRICS_GLOB)):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
            key, seed = (str(r["model"]), str(r["split"])), int(r["seed"])
            f1, accuracy = float(r["f1"]), float(r["accuracy"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"[fig4] Ignoring {path.name}: {e!r}")
            continue
        if not (0 <= f1 <= 1 and 0 <= accuracy <= 1):
            print(f"[fig4] Ignoring {path.name}: f1/accuracy outside 0-1")
        elif seed in groups[key]:
            print(f"[fig4] Ignoring {path.name}: repeats seed {seed} for {key}")
        else:
            groups[key][seed] = {"f1": f1, "accuracy": accuracy}
    return groups


def fig4_static_metrics():
    groups = load_static_metrics()
    enough = {k: v for k, v in sorted(groups.items()) if len(v) >= MIN_STATIC_SEEDS}
    for key in sorted(set(groups) - set(enough)):
        print(f"[fig4] Not shown: {key} has {len(groups[key])} seed(s), "
              f"needs {MIN_STATIC_SEEDS} for a mean +/- std")
    if not enough:
        print(f"[fig4] SKIPPED: no per-seed static results at {STATIC_METRICS_GLOB}")
        return

    model_color = {"A": SERIES_COLOR["hand_a"], "B": SERIES_COLOR["hand_b"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 1.1 * len(enough) + 2.8))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.78, bottom=0.12, wspace=0.12)
    rows = []
    for ax, metric, metric_name in zip(axes, ("f1", "accuracy"), ("F1", "Accuracy")):
        all_values = [s[metric] for seeds in enough.values() for s in seeds.values()]
        pad = max(0.01, 0.2 * (max(all_values) - min(all_values)))
        ax.set_xlim(max(0, min(all_values) - pad), min(1, max(all_values) + pad))
        ax.set_ylim(len(enough) - 0.5, -0.5)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="x", length=0, colors=SECONDARY, labelsize=9)
        ax.set_title(metric_name, loc="left", fontsize=10, fontweight="semibold", color=INK)
        for y, ((model, split), seeds) in enumerate(enough.items()):
            values = np.array([s[metric] for s in seeds.values()])
            mean, std = float(values.mean()), float(values.std(ddof=1))
            color = model_color.get(model, MUTED)
            ax.plot([mean - std, mean + std], [y, y], color=color, linewidth=2, zorder=2,
                    solid_capstyle="round")
            ax.scatter(values, [y] * len(values), s=44, color=color, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
            ax.text(mean, y - 0.14, f"{mean:.3f} ± {std:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
            rows.append([model, split, metric, len(values), f"{mean:.4f}", f"{std:.4f}",
                         " ".join(f"{v:.4f}" for v in values)])
            if ax is axes[0]:
                draw_row_label(ax, y, f"Model {model} ({split})", f"{len(values)} seeds")

    draw_header(fig, "Static classification metrics across training seeds",
                "Dot = one seed. Line = ±1 SD (n-1).")
    save_figure(fig, "fig4_static_metrics")
    write_csv("fig4_static_metrics",
              ["model", "split", "metric", "n_seeds", "mean", "std", "per_seed_values"], rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    conds = load_conditions()
    fig1_closed_loop(conds)
    fig2_rl_seed_spread(conds)
    fig3_per_object(conds)
    fig4_static_metrics()


if __name__ == "__main__":
    main()
