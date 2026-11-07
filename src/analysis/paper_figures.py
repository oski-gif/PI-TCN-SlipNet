"""Paper figures and statistics.

Builds every figure in reports/figures/paper/ from stored trial rows, writes a CSV of the plotted
numbers beside each figure, and writes every test result to reports/figures/paper/stats_summary.md.
Re-run it when the fresh-seed runs finish: figures that use them gain those series, and a run that
is missing or incomplete is left out with a printed notice, never estimated.

Run from the repo root:
    venv/Scripts/python.exe src/analysis/paper_figures.py

Only json, numpy, scipy, matplotlib and tensorboard's log reader, so it is safe beside a simulation.

Conventions, the same in every figure:
- Color is the contact setting: blue for elliptic (the primary benchmark), orange for MuJoCo's
  defaults, aqua for the noslip pass. Each setting also has its own marker shape, so the figures
  survive grayscale printing.
- "Held" is RL's rule unless a figure says otherwise: within the slip limit at the end, and no
  fall of more than 5 cm while the grasp forms.
- Intervals are 95% Wilson intervals. Paired comparisons are exact McNemar tests on the same
  trials. A spread across training seeds is an SD, not an interval.
- Locked set: 15 objects x seeds 3-66 (960 trials), default physics, pre-registered. Dev set:
  seeds 1000-1009 (150 trials), looked at many times, so exploratory. Fresh seeds: 2000-2066
  (1,005 trials), untouched before the Nov 4-6 runs.
"""
import csv
import glob
import json
import math
import os
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = ROOT / "reports" / "figures" / "paper"

# Reference palette from the dataviz method, validated on white for these three slots, all pairs.
INK, INK2, MUTED, GRID, AXIS, WASH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
SETTINGS = {
    "elliptic": {"color": "#2a78d6", "marker": "o", "label": "Elliptic cone, impratio 10 (primary)"},
    "default": {"color": "#eb6834", "marker": "s", "label": "MuJoCo default contact settings"},
    "noslip": {"color": "#1baf7a", "marker": "D", "label": "Noslip pass, 20 iterations"},
}
SHORT = {"elliptic": "Elliptic (primary)", "default": "MuJoCo default", "noslip": "Noslip"}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

LOCKED, ALL67, DEV, FRESH = set(range(3, 67)), set(range(0, 67)), set(range(1000, 1010)), set(range(2000, 2067))
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
RL_STEP_S = 0.01
PULL_START, FULL_LOAD, TRIAL_END = 160, 260, 460  # RL steps: settle ends, pull reaches full value, end
GAMMA, HOLD_BONUS = 0.99, 20.0  # SB3's default discount, never overridden; rl_env.HOLD_BONUS

NOTES, STATS, MADE = [], [], []


# ---------------------------------------------------------------------------- data and tests

def read_rows(rel, grip=None):
    path = DATA / rel
    if not path.exists():
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or (grip and r.get("grip") != grip):
                continue
            rows[(r["object"], int(r["seed"]))] = r
    return rows


def held(r, cm=5):
    """cm=None is "no drop rule": only the slip limit counts."""
    if cm is None:
        return not r["slipped"]
    return (not r["slipped"]) and r["formation_max_fall"] <= cm / 100.0


def get(label, rel, seeds, n, held5=None, grip=None):
    """Rows for one controller and setting, or None with a notice if they fail a check."""
    rows = read_rows(rel, grip)
    if rows is None:
        NOTES.append(f"left out: {label}, {rel} does not exist")
        return None
    rows = {k: v for k, v in rows.items() if k[1] in seeds}
    if len(rows) != n:
        NOTES.append(f"left out: {label}, {len(rows)} of {n} trials in {rel} (not complete)")
        return None
    if "formation_max_fall" in next(iter(rows.values())):
        bad = sum(1 for r in rows.values() if "held" in r and bool(r["held"]) != held(r))
        if bad:
            NOTES.append(f"left out: {label}, its stored held flag disagrees with the rule on {bad} trials")
            return None
        h = sum(held(r) for r in rows.values())
        if held5 is not None and h != held5:
            NOTES.append(f"left out: {label}, held {h} at 5 cm, but results.md says {held5}")
            return None
    return rows


def count(rows, cm=5):
    return sum(held(r, cm) for r in rows.values()), len(rows)


def wilson(k, n, z=1.959964):
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    d = 1 + z * z / n
    # 100 * k / n, not 100 * p: 588/960 is exactly 61.25 that way and rounds like results.md does.
    return 100 * k / n, 100 * (centre - margin) / d, 100 * (centre + margin) / d


def mcnemar(b, c):
    return 1.0 if b + c == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))


def paired(ra, rb, cm=5):
    """Held only by a, held only by b, exact McNemar p, over the trials both ran."""
    keys = sorted(set(ra) & set(rb))
    b = sum(held(ra[k], cm) and not held(rb[k], cm) for k in keys)
    c = sum(held(rb[k], cm) and not held(ra[k], cm) for k in keys)
    return b, c, mcnemar(b, c), len(keys)


def cmh(tables):
    """Cochran-Mantel-Haenszel test over 2x2 tables (a, b, c, d) = (high held, high not, low held,
    low not). Returns the common odds ratio of holding at high vs. low friction, and p."""
    num = den = var = dev = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n < 2:
            continue
        num += a * d / n
        den += b * c / n
        dev += a - (a + b) * (a + c) / n
        var += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if var == 0:
        return float("nan"), 1.0
    chi2 = (abs(dev) - 0.5) ** 2 / var
    return (num / den if den else float("inf")), float(stats.chi2.sf(chi2, 1))


def fmt_p(p):
    if p < 0.0001:
        return "p < 0.0001"
    if p < 0.01:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}" if p < 0.2 else f"p = {p:.2f}"


def pct(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.1f}% [{lo:.1f}, {hi:.1f}]"


def nice(obj):
    return obj.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------- drawing

def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelsize": 8, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "pdf.fonttype": 42, "text.color": INK,
    })


def forest_axes(ax, labels, xlabel, xlim=(0, 100)):
    ys = list(range(len(labels)))[::-1]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return ys


def draw_rate(ax, y, k, n, setting, filled=True, ms=5.5):
    p, lo, hi = wilson(k, n)
    s = SETTINGS[setting]
    ax.plot([lo, hi], [y, y], color=s["color"], lw=1.5, zorder=2)
    ax.plot([p], [y], ls="none", marker=s["marker"], ms=ms, zorder=3, mfc=s["color"] if filled else "white",
            mec="white" if filled else s["color"], mew=0.9 if filled else 1.3)
    return p, lo, hi


def handle(setting, filled=True, label=None, color=None):
    s = SETTINGS[setting]
    c = color or s["color"]
    return Line2D([0], [0], color=c, lw=1.5, marker=s["marker"], ms=5.5, mfc=c if filled else "white",
                  mec="white" if filled else c, mew=0.9 if filled else 1.3, label=label or s["label"])


def target_line(ax, x=70, text="70% target"):
    ax.axvline(x, color=MUTED, lw=0.8, zorder=1)
    ax.text(x, 1.0, " " + text, transform=ax.get_xaxis_transform(), ha="left", va="bottom", color=INK2, fontsize=7)


def right_text(ax, y, text, x=1.01):
    ax.text(x, y, text, transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK2, fontsize=7)


def save(fig, name, header, rows):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp, so unchanged figures stay byte-identical
    plt.close(fig)
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    MADE.append(name)


# ---------------------------------------------------------------------------- loading

def load_all():
    D = {}
    rl = {1: ("rl_seed1", 588, "ppo_slip_policy_latest"), 2: ("rl_seed2", 405, "ppo_slip_policy_seedcheck"),
          3: ("rl_seed3", 144, "ppo_slip_policy_seed3")}
    for s, (stem, h, _) in rl.items():
        D[("locked", "default", f"RL seed {s}")] = get(f"RL seed {s}, locked", f"{stem}_1005trial_results.jsonl", LOCKED, 960, h)
    for m, h in (("A", 180), ("B", 185)):
        D[("locked", "default", f"Hand-tuned {m}")] = get(f"Hand-tuned {m}, locked", f"handtuned_model{m}_2300step_results.jsonl", LOCKED, 960, h)
        D[("all67", "default", f"Hand-tuned {m}")] = get(f"Hand-tuned {m}, 1,005", f"handtuned_model{m}_2300step_results.jsonl", ALL67, 1005)
        D[("all67_1500", "default", f"Hand-tuned {m}")] = get(f"Hand-tuned {m}, 1,500 steps", f"baseline_model{m}_1000trial_results.jsonl", ALL67, 1005)
    for s in (1, 2, 3):
        D[("all67", "default", f"RL seed {s}")] = get(f"RL seed {s}, 1,005", f"rl_seed{s}_1005trial_results.jsonl", ALL67, 1005)

    dev = {
        ("default", "RL seed 1"): ("rl_seed1_dev_deterministic.jsonl", 86),
        ("default", "RL seed 2"): ("rl_seed2_dev_deterministic.jsonl", 56),
        ("default", "RL seed 3"): ("rl_seed3_dev_deterministic.jsonl", 19),
        ("noslip", "RL seed 1"): ("diag_contact_physics/seed1_dev150_noslip20.jsonl", 122),
        ("noslip", "RL seed 2"): ("diag_contact_physics/seed2_dev150_noslip20.jsonl", 101),
        ("noslip", "RL seed 3"): ("diag_contact_physics/seed3_dev150_noslip20.jsonl", 61),
        ("elliptic", "RL seed 1"): ("diag_contact_physics/seed1_dev150_elliptic_impratio10.jsonl", 118),
        ("elliptic", "RL seed 2"): ("diag_contact_physics/seed2_dev150_elliptic_impratio10.jsonl", 87),
        ("elliptic", "RL seed 3"): ("diag_contact_physics/seed3_dev150_elliptic_impratio10.jsonl", None),
        ("default", "Hand-tuned A"): ("diag_contact_physics/handA_dev150_default.jsonl", 35),
        ("noslip", "Hand-tuned A"): ("diag_contact_physics/handA_dev150_noslip20.jsonl", 66),
        ("default", "RL seed 1, traced"): ("diag_contact_physics/seed1_dev150_default.jsonl", 86),
        ("default", "RL seed 1, no TCN"): ("diag_contact_physics/seed1_dev150_noTCN.jsonl", 86),
        ("default", "RL seed 2, no TCN"): ("diag_contact_physics/seed2_dev150_noTCN.jsonl", 56),
        ("default", "RL seed 3, no TCN"): ("diag_contact_physics/seed3_dev150_noTCN.jsonl", 19),
        ("noslip", "RL seed 1, no TCN"): ("diag_contact_physics/seed1_dev150_noslip20_noTCN.jsonl", 122),
        ("noslip", "RL seed 2, no TCN"): ("diag_contact_physics/seed2_dev150_noslip20_noTCN.jsonl", 99),
        ("noslip", "RL seed 3, no TCN"): ("diag_contact_physics/seed3_dev150_noslip20_noTCN.jsonl", 60),
        ("default", "Hand-tuned A, no TCN"): ("diag_contact_physics/handA_dev150_default_noTCN.jsonl", 38),
        ("noslip", "Hand-tuned A, no TCN"): ("diag_contact_physics/handA_dev150_noslip20_noTCN.jsonl", 71),
    }
    for (setting, name), (rel, h) in dev.items():
        D[("dev", setting, name)] = get(f"{name}, dev, {setting}", rel, DEV, 150, h)
    for grip in ("TIRP_fast", "seed1_like", "TIR_fast", "TIMRP_fast", "hand_tuned_like"):
        D[("dev", "default", grip)] = get(f"{grip}, dev, default", "diag_scripted_grip/dev_first_grips.jsonl", DEV, 150, grip=grip)
        D[("dev", "noslip", grip)] = get(f"{grip}, dev, noslip", "diag_scripted_grip/physics_on_grip.jsonl", DEV, 150, grip=grip)

    for tag, name in (("seed1", "RL seed 1"), ("seed2", "RL seed 2"), ("seed3", "RL seed 3"), ("handA", "Hand-tuned A")):
        for setting, suffix in (("elliptic", "elliptic_impratio10"), ("noslip", "noslip20")):
            D[("fresh", setting, name)] = get(f"{name}, fresh, {setting}", f"diag_fresh_seeds/{tag}_{suffix}.jsonl", FRESH, 1005)
    D[("fresh", "elliptic", "Fixed grip")] = get("Fixed grip, fresh, elliptic",
                                                 "diag_fresh_seeds/grip_TIRP_fast_elliptic_impratio10.jsonl", FRESH, 1005)
    return D


# ---------------------------------------------------------------------------- figures

def fig_success_definition(D):
    cuts = list(range(1, 26))
    none_x = 28
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    out = []
    panels = [
        ("a  Hand-tuned controller, 1,005 trials", [("Model A", D[("all67", "default", "Hand-tuned A")], "-"),
                                                    ("Model B", D[("all67", "default", "Hand-tuned B")], (0, (3, 2)))]),
        ("b  RL, three training seeds, locked set (960)", [(f"Seed {s}", D[("locked", "default", f"RL seed {s}")], "-")
                                                           for s in (1, 2, 3)]),
    ]
    color = SETTINGS["default"]["color"]
    for ax, (title, series) in zip(axes, panels):
        for name, rows, ls in series:
            if rows is None:
                continue
            n = len(rows)
            ys = [100 * count(rows, c)[0] / n for c in cuts]
            none = 100 * count(rows, None)[0] / n
            ax.plot(cuts, ys, color=color, ls=ls, lw=1.6)
            ax.plot([5, 10, 15], [ys[4], ys[9], ys[14]], ls="none", marker="s", ms=4.5, mfc=color, mec="white", mew=0.8)
            ax.plot([none_x], [none], ls="none", marker="s", ms=5, mfc="white", mec=color, mew=1.3)
            out += [[title[3:], name, c, n, count(rows, c)[0], round(y, 2)] for c, y in zip(cuts, ys)]
            out.append([title[3:], name, "no drop rule", n, count(rows, None)[0], round(none, 2)])
            if "Seed" in name:
                ax.text(12.5, ys[11] + 1.6, name, va="bottom", ha="center", color=INK2, fontsize=7)
        if "Hand" in title:
            rows = series[0][1]
            if rows is not None:
                y12 = 100 * count(rows, 12)[0] / len(rows)
                ax.text(12.5, y12 + 1.6, "Models A and B (lines overlap)", va="bottom", ha="center", color=INK2, fontsize=7)
        for x, lab in ((5, "5 cm, RL's rule"), (15, "15 cm")):
            ax.axvline(x, color=AXIS, lw=1.0, zorder=1)
            ax.text(x, 1.0, " " + lab, transform=ax.get_xaxis_transform(), ha="left", va="bottom", color=INK2, fontsize=7)
        ax.set_xticks([1, 5, 10, 15, 20, 25, none_x])
        ax.set_xticklabels(["1", "5", "10", "15", "20", "25", "none"])
        ax.set_xlim(0, 30.5)
        ax.set_xlabel("Largest fall allowed while the grasp forms (cm)")
        ax.set_title(title, pad=16)
    axes[0].set_ylabel("Trials scored as held (%)")
    axes[0].set_ylim(0, 70)
    save(fig, "fig01_success_definition", ["panel", "controller", "cutoff_cm", "n", "held", "held_pct"], out)
    for key in (("all67", "default", "Hand-tuned A"), ("all67", "default", "Hand-tuned B")):
        rows = D[key]
        if rows:
            STATS.append(f"- {key[2]}, 1,005 trials, 2,300 steps: no drop rule {pct(*count(rows, None))}; "
                         f"15 cm {pct(*count(rows, 15))}; 10 cm {pct(*count(rows, 10))}; 5 cm {pct(*count(rows, 5))}.")


def fig_formation_falls(D):
    panels = [("Hand-tuned A, 1,005", D[("all67", "default", "Hand-tuned A")])] + \
             [(f"RL seed {s}, 960", D[("locked", "default", f"RL seed {s}")]) for s in (1, 2, 3)]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1), sharey=True)
    out = []
    bins = np.arange(0, 26, 1.0)
    for ax, (title, rows) in zip(axes, panels):
        if rows is None:
            continue
        falls = np.clip([100 * r["formation_max_fall"] for r in rows.values()], 0, 24.999)
        counts, _ = np.histogram(falls, bins=bins)
        ax.bar(bins[:-1] + 0.5, counts, width=0.78, color=SETTINGS["default"]["color"], zorder=2)
        for x in (5, 10):
            ax.axvline(x, color=AXIS, lw=1.0, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 1500)
        ax.set_xlim(0, 25)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        over5 = sum(f > 5 for f in falls)
        band = sum(5 < f <= 10 for f in falls)
        ax.text(0.62, 0.66, f"{over5} fell > 5 cm,\n{band} of them\n5-10 cm", transform=ax.transAxes, ha="center",
                va="center", fontsize=6.8, color=INK2)
        out += [[title, int(lo), int(lo + 1), int(c)] for lo, c in zip(bins[:-1], counts)]
    axes[0].set_ylabel("Trials (log scale)")
    fig.supxlabel("Largest fall while the grasp forms (cm); lines at 5 and 10 cm", fontsize=8, color=INK2, y=-0.06)
    save(fig, "fig02_formation_falls", ["controller", "bin_low_cm", "bin_high_cm", "trials"], out)


def fig_detector_vs_closed_loop(D):
    rowsA = {"1500": D[("all67_1500", "default", "Hand-tuned A")], "2300": D[("all67", "default", "Hand-tuned A")]}
    rowsB = {"1500": D[("all67_1500", "default", "Hand-tuned B")], "2300": D[("all67", "default", "Hand-tuned B")]}
    spec = [("1,500 steps, no drop rule", "1500", None), ("2,300 steps, no drop rule", "2300", None),
            ("2,300 steps, 15 cm rule", "2300", 15), ("2,300 steps, 10 cm rule", "2300", 10),
            ("2,300 steps, 5 cm rule", "2300", 5)]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ys = forest_axes(ax, [s[0] for s in spec], "Trials held (%), same 1,005 trials for both detectors", (0, 75))
    out = []
    for y, (label, horizon, cm) in zip(ys, spec):
        a, b = rowsA[horizon], rowsB[horizon]
        if a is None or b is None:
            continue
        ka, n = count(a, cm)
        kb, _ = count(b, cm)
        draw_rate(ax, y + 0.14, ka, n, "default", True)
        draw_rate(ax, y - 0.14, kb, n, "default", False)
        only_a, only_b, p, _ = paired(a, b, cm)
        right_text(ax, y, f"A {100 * ka / n:.1f}%  B {100 * kb / n:.1f}%   only A {only_a}, only B {only_b}, {fmt_p(p)}")
        out.append([label, n, ka, kb, only_a, only_b, round(p, 4)])
        STATS.append(f"- Detector comparison, {label}: Model A {pct(ka, n)}, Model B {pct(kb, n)}; "
                     f"held only with A {only_a}, only with B {only_b}, exact McNemar {fmt_p(p)}.")
    ax.legend(handles=[handle("default", True, "Model A, static F1 0.691"), handle("default", False, "Model B, static F1 0.799")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "fig03_detector_vs_closed_loop", ["scoring", "n", "held_A", "held_B", "only_A", "only_B", "mcnemar_p"], out)


def fig_tcn_removal(D):
    spec = [("RL seed 1, default", "default", "RL seed 1", "RL seed 1, no TCN"),
            ("RL seed 2, default", "default", "RL seed 2", "RL seed 2, no TCN"),
            ("RL seed 3, default", "default", "RL seed 3", "RL seed 3, no TCN"),
            ("RL seed 1, noslip", "noslip", "RL seed 1", "RL seed 1, no TCN"),
            ("RL seed 2, noslip", "noslip", "RL seed 2", "RL seed 2, no TCN"),
            ("RL seed 3, noslip", "noslip", "RL seed 3", "RL seed 3, no TCN"),
            ("Hand-tuned A, default", "default", "Hand-tuned A", "Hand-tuned A, no TCN"),
            ("Hand-tuned A, noslip", "noslip", "Hand-tuned A", "Hand-tuned A, no TCN")]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ys = forest_axes(ax, [s[0] for s in spec], "Trials held (%), dev set, same 150 trials with and without the TCN")
    out = []
    tot_b = tot_c = 0
    for y, (label, setting, with_name, without_name) in zip(ys, spec):
        a, b = D[("dev", setting, with_name)], D[("dev", setting, without_name)]
        if a is None or b is None:
            continue
        ka, n = count(a)
        kb, _ = count(b)
        draw_rate(ax, y + 0.14, ka, n, setting, True)
        draw_rate(ax, y - 0.14, kb, n, setting, False)
        only_a, only_b, p, _ = paired(a, b)
        tot_b, tot_c = tot_b + only_a, tot_c + only_b
        right_text(ax, y, f"{ka} vs {kb}   only with {only_a}, only without {only_b}, {fmt_p(p)}")
        out.append([label, n, ka, kb, only_a, only_b, round(p, 4)])
    ax.legend(handles=[handle("default", True, "Default, with TCN"), handle("default", False, "Default, without"),
                       handle("noslip", True, "Noslip, with TCN"), handle("noslip", False, "Noslip, without")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    save(fig, "fig04_tcn_removal", ["controller_setting", "n", "held_with", "held_without", "only_with", "only_without", "mcnemar_p"], out)
    STATS.append(f"- TCN removal, dev set, all 8 pairs: {tot_b} trials held only with the TCN, {tot_c} only without "
                 f"({tot_b + tot_c} of {150 * len(out)} outcomes change).")


def rate_rows(D, names, settings, sets):
    return [(name, [(s, D.get((sets[s], s, name))) for s in settings]) for name in names]


def fig_rl_seed_spread(D):
    names = ["RL seed 1", "RL seed 2", "RL seed 3", "Mean of 3 seeds", "Hand-tuned A", "Hand-tuned B", "Fixed grip"]
    sources = {"default": "locked", "elliptic": "fresh", "noslip": "fresh"}
    present = [s for s in ("elliptic", "default", "noslip")
               if any(D.get((sources[s], s, n)) for n in names)]
    has_grip = D.get(("fresh", "elliptic", "Fixed grip")) is not None
    names = [n for n in names if n != "Fixed grip" or has_grip]
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(names) + 1.1))
    labels = [n if n != "Mean of 3 seeds" else "Mean of 3 seeds (± 1 SD)" for n in names]
    labels = [lab if lab != "Fixed grip" else "Fixed grip, no learning" for lab in labels]
    ys = forest_axes(ax, labels, "Trials held at 5 cm (%)")
    out = []
    for y, name in zip(ys, names):
        if name == "Mean of 3 seeds":
            here = [s for s in present if all(D.get((sources[s], s, f"RL seed {i}")) for i in (1, 2, 3))]
        else:
            here = [s for s in present if D.get((sources[s], s, name)) is not None]
        spread = 0.14 * (len(here) - 1)
        offsets = dict(zip(here, np.linspace(spread, -spread, len(here)) if len(here) > 1 else [0.0]))
        for s in here:
            src = sources[s]
            yy = y + offsets[s]
            if name == "Mean of 3 seeds":
                seeds = [D.get((src, s, f"RL seed {i}")) for i in (1, 2, 3)]
                if any(r is None for r in seeds):
                    continue
                vals = [100 * count(r)[0] / len(r) for r in seeds]
                m, sd = st.mean(vals), st.stdev(vals)
                c = SETTINGS[s]["color"]
                ax.plot([m - sd, m + sd], [yy, yy], color=c, lw=1.5, alpha=0.55)
                ax.plot([m], [yy], ls="none", marker=SETTINGS[s]["marker"], ms=7, mfc=c, mec="white", mew=0.9)
                right_text(ax, yy, f"{m:.1f}% ± {sd:.1f} SD")
                out.append([name, s, src, "", "", round(m, 2), "", "", round(sd, 2)])
                chi2, p, _, _ = stats.chi2_contingency([[count(r)[0], len(r) - count(r)[0]] for r in seeds])
                STATS.append(f"- RL seeds, {s} ({src} set): {', '.join(f'{v:.1f}%' for v in vals)}; mean {m:.1f}%, SD {sd:.1f}. "
                             f"Chi-square test that the three seeds hold at the same rate: chi2 = {chi2:.1f}, {fmt_p(p)}. "
                             f"Sampling SE per seed is about {100 * math.sqrt(0.25 / len(seeds[0])):.1f} points at most.")
                continue
            rows = D.get((src, s, name))
            if rows is None:
                continue
            k, n = count(rows)
            p, lo, hi = draw_rate(ax, yy, k, n, s)
            right_text(ax, yy, f"{p:.1f}% [{lo:.1f}, {hi:.1f}]  {k}/{n}")
            out.append([name, s, src, k, n, round(p, 2), round(lo, 2), round(hi, 2), ""])
    target_line(ax)
    ax.legend(handles=[handle(s, label=SETTINGS[s]["label"] + (", locked set" if s == "default" else ", fresh seeds"))
                       for s in present], loc="lower left", bbox_to_anchor=(0, 1.06), ncol=len(present))
    save(fig, "fig05_rl_seed_spread", ["controller", "setting", "set", "held", "n", "held_pct", "ci_low", "ci_high", "sd_across_seeds"], out)
    for s in (1, 2, 3):
        a, b = D[("locked", "default", f"RL seed {s}")], D[("locked", "default", "Hand-tuned A")]
        if a and b:
            only_rl, only_h, p, n = paired(a, b)
            STATS.append(f"- Locked set, default: RL seed {s} vs hand-tuned A, held only by RL {only_rl}, only by hand-tuned {only_h}, {fmt_p(p)}.")


def fig_fixed_grips(D):
    grips = [("TIRP_fast", "Fixed: thumb, index, ring, pinky, full speed"),
             ("seed1_like", "Fixed: same fingers at seed 1's speeds"),
             ("TIR_fast", "Fixed: thumb, index, ring, full speed"),
             ("TIMRP_fast", "Fixed: all five fingers, full speed"),
             ("hand_tuned_like", "Fixed: all five, hand-tuned ramp")]
    pols = [("RL seed 1", "RL seed 1"), ("RL seed 2", "RL seed 2"), ("RL seed 3", "RL seed 3"), ("Hand-tuned A", "Hand-tuned A")]
    spec = grips + pols
    fresh_grip = D.get(("fresh", "elliptic", "Fixed grip"))
    two = fresh_grip is not None
    if two:  # stacked, so panel b's labels and title stay clear of panel a
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), gridspec_kw={"height_ratios": [9, 5], "hspace": 0.55})
    else:
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ys = forest_axes(ax, [s[1] for s in spec], "Trials held at 5 cm (%), dev set, 150 trials")
    out = []
    for y, (key, label) in zip(ys, spec):
        for setting, off in (("default", 0.14), ("noslip", -0.14)):
            rows = D.get(("dev", setting, key))
            if rows is None:
                continue
            k, n = count(rows)
            p, lo, hi = draw_rate(ax, y + off, k, n, setting)
            out.append(["dev", label, setting, k, n, round(p, 2), round(lo, 2), round(hi, 2)])
    ax.axhline(len(pols) - 0.5, color=GRID, lw=0.8)
    if two:
        ax.set_title("a  Dev set: fixed grips against the controllers", pad=22)
    ax.legend(handles=[handle("default"), handle("noslip")], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    for setting in ("default", "noslip"):
        g, r = D.get(("dev", setting, "TIRP_fast")), D.get(("dev", setting, "RL seed 1"))
        if g and r:
            og, orl, p, n = paired(g, r)
            STATS.append(f"- Dev set, {setting}: fixed grip TIRP {pct(*count(g))} vs RL seed 1 {pct(*count(r))}; "
                         f"held only by the grip {og}, only by RL {orl}, {fmt_p(p)}.")
    if two:
        names = ["Fixed grip", "RL seed 1", "RL seed 2", "RL seed 3", "Hand-tuned A"]
        ys2 = forest_axes(ax2, [grips[0][1]] + names[1:], "Trials held at 5 cm (%), fresh seeds, 1,005 trials")
        for y, name in zip(ys2, names):
            rows = D.get(("fresh", "elliptic", name))
            if rows is None:
                continue
            k, n = count(rows)
            p, lo, hi = draw_rate(ax2, y, k, n, "elliptic")
            if name != "Fixed grip":
                og, oc, pv, _ = paired(fresh_grip, rows)
                right_text(ax2, y, f"grip-only {og}, {name.split(',')[0]}-only {oc}, {fmt_p(pv)}")
                STATS.append(f"- Fresh seeds, elliptic: fixed grip {pct(*count(fresh_grip))} vs {name} {pct(k, n)}; "
                             f"held only by the grip {og}, only by {name} {oc}, {fmt_p(pv)}.")
            out.append(["fresh", name, "elliptic", k, n, round(p, 2), round(lo, 2), round(hi, 2)])
        ax2.set_title("b  Fresh seeds, elliptic: the fixed grip against each controller", pad=14)
        target_line(ax2)
    save(fig, "fig06_fixed_grips", ["set", "controller", "setting", "held", "n", "held_pct", "ci_low", "ci_high"], out)


def fig_contact_settings_dev(D):
    names = ["RL seed 1", "RL seed 2", "RL seed 3", "Hand-tuned A", "TIRP_fast"]
    labels = ["RL seed 1", "RL seed 2", "RL seed 3", "Hand-tuned A", "Fixed grip (thumb, index, ring, pinky)"]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ys = forest_axes(ax, labels, "Trials held at 5 cm (%), dev set, same 150 trials under each setting")
    out = []
    for y, name in zip(ys, names):
        base = D.get(("dev", "default", name))
        for setting, off in (("default", 0.2), ("noslip", 0.0), ("elliptic", -0.2)):
            rows = D.get(("dev", setting, name))
            if rows is None:
                continue
            k, n = count(rows)
            p, lo, hi = draw_rate(ax, y + off, k, n, setting)
            row = [name, setting, k, n, round(p, 2), round(lo, 2), round(hi, 2), "", "", ""]
            if setting != "default" and base is not None:
                gain, loss, pv, _ = paired(rows, base)
                row[7:] = [gain, loss, round(pv, 6)]
                STATS.append(f"- Dev set, {name}: {setting} {pct(k, n)} vs default {pct(*count(base))}; held only under {setting} "
                             f"{gain}, only under default {loss}, {fmt_p(pv)}.")
            out.append(row)
    ax.legend(handles=[handle("default"), handle("noslip"), handle("elliptic")], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    a, b = D.get(("dev", "noslip", "RL seed 1")), D.get(("dev", "elliptic", "RL seed 1"))
    if a and b:
        oa, ob, p, _ = paired(a, b)
        STATS.append(f"- Dev set, RL seed 1, noslip vs elliptic: held only under noslip {oa}, only under elliptic {ob}, {fmt_p(p)}.")
    save(fig, "fig07_contact_settings_dev", ["controller", "setting", "held", "n", "held_pct", "ci_low", "ci_high",
                                             "only_this_setting_vs_default", "only_default", "mcnemar_p"], out)


def fig_per_object(D):
    cols = [("locked", "default", "Hand-tuned A"), ("locked", "default", "Hand-tuned B"),
            ("locked", "default", "RL seed 1"), ("locked", "default", "RL seed 2"), ("locked", "default", "RL seed 3")]
    cols += [c for c in [("fresh", "elliptic", "Hand-tuned A"), ("fresh", "elliptic", "RL seed 1"),
                         ("fresh", "elliptic", "RL seed 2"), ("fresh", "elliptic", "RL seed 3"),
                         ("fresh", "elliptic", "Fixed grip")] if D.get(c) is not None]
    cols = [c for c in cols if D.get(c) is not None]
    objects = sorted({k[0] for k in D[cols[0]]})
    M = np.array([[100 * np.mean([held(r) for kk, r in D[c].items() if kk[0] == o]) for c in cols] for o in objects])
    order = np.argsort(-M[:, :5].mean(axis=1))
    M, objects = M[order], [objects[i] for i in order]
    fig, ax = plt.subplots(figsize=(1.3 + 0.62 * len(cols), 4.6))
    im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if M[i, j] > 55 else INK)
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels([nice(o) for o in objects])
    short = {"Hand-tuned A": "Hand A", "Hand-tuned B": "Hand B", "RL seed 1": "RL 1", "RL seed 2": "RL 2",
             "RL seed 3": "RL 3", "Fixed grip": "Grip"}
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([short[c[2]] for c in cols])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_def = 5
    ax.text((n_def - 1) / 2, -1.6, "Default, locked set (64 per object)", ha="center", va="bottom", fontsize=7, color=INK2)
    if len(cols) > n_def:
        ax.axvline(n_def - 0.5, color="white", lw=3)
        ax.text(n_def + (len(cols) - n_def - 1) / 2, -1.6, "Elliptic, fresh seeds (67 per object)", ha="center",
                va="bottom", fontsize=7, color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Trials held at 5 cm (%)", color=INK2)
    cb.outline.set_visible(False)
    out = [[nice(o)] + [round(v, 1) for v in M[i]] for i, o in enumerate(objects)]
    save(fig, "fig08_per_object", ["object"] + [f"{c[2]} ({c[1]}, {c[0]})" for c in cols], out)


def survival(rows):
    inhand = [r for r in rows.values() if r["formation_max_fall"] <= 0.05]
    slip_t = np.array([r["ep_len"] if r["slipped"] else np.inf for r in inhand])
    t = np.arange(PULL_START, TRIAL_END + 1)
    return t, np.array([100 * np.mean(slip_t > x) for x in t]), slip_t, len(inhand)


def fig_survival(D):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    out = []
    for ax, s in zip(axes, (1, 2, 3)):
        name = f"RL seed {s}"
        ax.axvspan(0, (FULL_LOAD - PULL_START) * RL_STEP_S, color=WASH, zorder=0, lw=0)
        for setting, src in (("default", "locked"), ("noslip", "fresh"), ("elliptic", "fresh")):
            rows = D.get((src, setting, name))
            if rows is None:
                continue
            t, S, slip_t, n = survival(rows)
            x = (t - PULL_START) * RL_STEP_S
            ax.plot(x, S, color=SETTINGS[setting]["color"], lw=1.8)
            ax.plot([x[-1]], [S[-1]], ls="none", marker=SETTINGS[setting]["marker"], ms=5, mfc=SETTINGS[setting]["color"], mec="white", mew=0.8)
            out += [[name, setting, src, n, round(xx, 2), round(ss, 2)] for xx, ss in zip(x[::10], S[::10])]
            ramp = int(np.sum((slip_t > PULL_START) & (slip_t <= FULL_LOAD)))
            first = int(np.sum((slip_t > FULL_LOAD) & (slip_t <= 360)))
            second = int(np.sum((slip_t > 360) & (slip_t <= TRIAL_END)))
            STATS.append(f"- Slip timing, {name}, {setting} ({src} set, {n} trials in hand when the pull starts): "
                         f"{ramp} slip while the pull ramps up, {first} in the first second at full load, {second} in the "
                         f"second; {S[-1]:.1f}% still held at the end.")
        ax.set_title(name)
        ax.set_xlim(0, 3.08)
        ax.set_ylim(0, 102)
        ax.text(0.5, 3, "pull ramps up", ha="center", va="bottom", color=INK2, fontsize=6.8)
        ax.text(2.0, 3, "full pull, 1.5x weight", ha="center", va="bottom", color=INK2, fontsize=6.8)
    axes[0].set_ylabel("In-hand trials not yet slipped (%)")
    fig.supxlabel("Time since the pull started (s)", fontsize=8, color=INK2, y=-0.04)
    present = [s for s in ("default", "noslip", "elliptic") if any(r[1] == s for r in out)]
    axes[0].legend(handles=[handle(s, label=SHORT[s] + (", locked set" if s == "default" else ", fresh seeds"))
                            for s in present], loc="lower left", bbox_to_anchor=(0, 1.12), ncol=len(present))
    save(fig, "fig09_slip_timing", ["controller", "setting", "set", "n_in_hand", "time_s", "pct_not_slipped"], out)
    for m in ("A", "B"):
        short, long_ = D[("all67_1500", "default", f"Hand-tuned {m}")], D[("all67", "default", f"Hand-tuned {m}")]
        if short and long_:
            keys = sorted(set(short) & set(long_))
            ok_short = sum(not short[k]["slipped"] for k in keys)
            lost = sum((not short[k]["slipped"]) and long_[k]["slipped"] for k in keys)
            STATS.append(f"- Hand-tuned {m}, no drop rule: {ok_short} of {len(keys)} trials not slipped after 200 steps at full pull "
                         f"(1,500-step runs); {lost} of those slip by 1,000 steps at full pull (2,300-step runs), at constant load.")


def full_load_ok(r):
    fl = r.get("full_load")
    return bool(fl) and fl.get("friction_used") is not None and fl.get("n_tips_touching", 0) >= 1 \
        and "world" not in fl.get("bodies_touching", [])


def fig_creep(D):
    sets = [("default", "RL seed 1, traced"), ("noslip", "RL seed 1"), ("elliptic", "RL seed 1")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.3})
    out = []
    base = D.get(("dev", "default", "RL seed 1, traced"))
    for setting, name in sets:
        rows = D.get(("dev", setting, name))
        if rows is None:
            continue
        keys = [k for k, r in rows.items() if full_load_ok(r)]
        if not keys:
            continue
        fl = {k: rows[k]["full_load"] for k in keys}
        x = np.array([max(fl[k]["friction_used"], 1e-3) for k in keys])
        y = np.array([max(fl[k]["object_speed_mm_s"], 1e-2) for k in keys])
        cap = np.array([fl[k]["capacity_over_load"] for k in keys])
        s = SETTINGS[setting]
        short = s["label"].split(" (")[0]
        ax.plot(x, y, ls="none", marker=s["marker"], ms=4, mfc=s["color"], mec="white", mew=0.5, alpha=0.85,
                label=f"{short}, median {np.median(y):.2f} mm/s")
        out += [["all trials in hand", setting, nice(k[0]), k[1], round(fl[k]["friction_used"], 5),
                 round(fl[k]["object_speed_mm_s"], 4), "", round(fl[k]["capacity_over_load"], 2)] for k in keys]
        STATS.append(f"- Full load, RL seed 1, dev set, {setting}: {len(keys)} trials with the object in the hand; median object "
                     f"speed {np.median(y):.2f} mm/s (90th percentile {np.percentile(y, 90):.2f}); median friction used "
                     f"{100 * np.median(x):.1f}% of the limit; median capacity {np.median(cap):.0f}x the load.")
        if setting != "default" and base is not None:
            both = [k for k in keys if k in base and full_load_ok(base[k])]
            d = np.array([base[k]["full_load"]["object_speed_mm_s"] for k in both])
            e = np.array([rows[k]["full_load"]["object_speed_mm_s"] for k in both])
            slower = int(np.sum(e < d))
            ax2.plot(np.maximum(d, 1e-2), np.maximum(e, 1e-2), ls="none", marker=s["marker"], ms=4, mfc=s["color"],
                     mec="white", mew=0.5, alpha=0.85, label=f"{short.split(',')[0]}: slower on {slower} of {len(both)}")
            w = stats.wilcoxon(d, e)
            STATS.append(f"- Same trials in the hand at full load under default and {setting} (RL seed 1, dev, n = {len(both)}): "
                         f"median speed {np.median(d):.2f} vs. {np.median(e):.2f} mm/s; slower under {setting} on {slower}; "
                         f"Wilcoxon signed-rank {fmt_p(w.pvalue)}.")
            out += [["same trials", setting, nice(k[0]), k[1], "", round(float(ee), 4), round(float(dd), 4), ""]
                    for k, dd, ee in zip(both, d, e)]
    ax.axvline(1.0, color=MUTED, lw=0.8)
    ax.text(0.93, 0.03, "friction\nlimit", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 1.5)
    ax.set_ylim(1e-2, 100)
    ax.set_xlabel("Friction force used, as a fraction of the limit")
    ax.set_ylabel("Object speed at full load (mm/s)")
    ax.set_title("a  Objects move with contacts well inside the limit", pad=44)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.6, handletextpad=0.2, borderaxespad=0.2)
    lim = (1e-2, 100)
    ax2.plot(lim, lim, color=INK2, lw=0.9)
    ax2.text(0.97, 0.04, "below the line:\nslower than under default", transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=6.8, color=INK2)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(*lim)
    ax2.set_ylim(*lim)
    ax2.set_xlabel("Same trial under default settings (mm/s)")
    ax2.set_ylabel("Under the stricter setting (mm/s)")
    ax2.set_title("b  The same trials, default against stricter", pad=44)
    ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.6, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig10_creep_at_full_load", ["subset", "setting", "object", "seed", "friction_used", "speed_mm_s",
                                           "default_speed_mm_s_same_trial", "capacity_over_load"], out)


def fig_finger_closure(D):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.0), sharex=True)
    steps = np.arange(10, TRIAL_END + 1, 10)
    out = []
    im = None
    for ax, s in zip(axes, (1, 2, 3)):
        rows = D.get(("dev", "noslip", f"RL seed {s}"))
        if rows is None:
            continue
        by_step = {t: [] for t in steps}
        for r in rows.values():
            for smp in r.get("samples", []):
                if smp["rl_step"] in by_step:
                    by_step[smp["rl_step"]].append(smp["fraction"])
        M = np.full((5, len(steps)), np.nan)
        for j, t in enumerate(steps):
            if by_step[t]:
                M[:, j] = np.median(np.array(by_step[t]), axis=0)
                out += [[f"RL seed {s}", int(t), FINGERS[f], round(float(M[f, j]), 3), len(by_step[t])] for f in range(5)]
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto", extent=(5, TRIAL_END + 5, 4.5, -0.5), interpolation="nearest")
        ax.set_yticks(range(5))
        ax.set_yticklabels(FINGERS)
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for x in (PULL_START, FULL_LOAD):
            ax.axvline(x, color="white", lw=1.5)
        ax.set_title(f"RL seed {s}", pad=3)
    axes[-1].set_xlabel("RL step (0.01 s each); white lines: pull starts (160) and reaches full value (260)")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
        cb.set_label("Median closure (fraction of range)", color=INK2)
        cb.outline.set_visible(False)
    save(fig, "fig11_finger_closure", ["controller", "rl_step", "finger", "median_closure", "trials_still_running"], out)
    rows = D.get(("dev", "noslip", "RL seed 1"))
    if rows:
        mids = [smp["fraction"][2] for r in rows.values() for smp in r["samples"] if smp["rl_step"] == 50]
        STATS.append(f"- Seed 1's middle finger at RL step 50 (dev, noslip): median closure {np.median(mids):.2f}, "
                     f"{sum(m > 0.05 for m in mids)} of {len(mids)} trials close it at all.")


def fig_reward(D):
    rows = D[("locked", "default", "RL seed 1")]
    groups = {"Held": [], "Slipped": [], "Fell": []}
    for r in rows.values():
        g = "Fell" if r["formation_max_fall"] > 0.05 else ("Slipped" if r["slipped"] else "Held")
        groups[g].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1, 1.15]})
    out = []
    names = list(groups)
    col = SETTINGS["default"]["color"]
    rng = np.random.default_rng(0)
    for ax, field, title in ((axes[0], "ep_reward", "a  Total reward per trial"),
                             (axes[1], "phase2_reward", "b  Reward during the pull")):
        data = [[r[field] for r in groups[g]] for g in names]
        ax.boxplot(data, tick_labels=[f"{g}\nn = {len(groups[g])}" for g in names], widths=0.5, showfliers=False,
                   medianprops={"color": col, "lw": 1.8}, boxprops={"color": AXIS}, whiskerprops={"color": AXIS},
                   capprops={"color": AXIS})
        for i, d in enumerate(data):
            ax.plot(i + 1 + rng.uniform(-0.16, 0.16, len(d)), d, ls="none", marker="o", ms=1.5, color=col, alpha=0.22, zorder=1)
            ax.text(i + 1.3, np.median(d), f"{np.median(d):.0f}", va="center", ha="left", fontsize=6.8, color=INK2)
            out.append([field, names[i], len(d), round(float(np.median(d)), 2), round(float(np.percentile(d, 25)), 2),
                        round(float(np.percentile(d, 75)), 2)])
        ax.set_title(title)
        ax.set_ylabel("Reward")
        ax.grid(axis="x", visible=False)
        STATS.append(f"- Reward, RL seed 1, locked set, {field}: median "
                     + "; ".join(f"{g.lower()} {np.median(d):.1f} (n = {len(d)})" for g, d in zip(names, data)) + ".")
        for other in ("Slipped", "Fell"):
            u = stats.mannwhitneyu(data[0], data[names.index(other)])
            STATS.append(f"  - {field}, held vs. {other.lower()}: Mann-Whitney {fmt_p(u.pvalue)}.")
    axes[0].set_ylim(0, 900)
    ax = axes[2]
    t = np.arange(0, TRIAL_END + 1)
    ax.plot(t, HOLD_BONUS * GAMMA ** (TRIAL_END - t), color=col, lw=1.8)
    for x, txt, tx, ty in ((0, "trial start", 10, 3.0), (PULL_START, "pull start", 120, 8.0)):
        v = HOLD_BONUS * GAMMA ** (TRIAL_END - x)
        ax.plot([x], [v], ls="none", marker="s", ms=5, mfc=col, mec="white", mew=0.8)
        ax.text(tx, ty, f"{v:.2f} at {txt}", fontsize=6.8, color=INK2, va="bottom")
    ax.set_xlim(-5, TRIAL_END + 5)
    ax.set_ylim(0, 21)
    ax.set_xlabel("RL step")
    ax.set_ylabel("Value of the +20 hold bonus")
    ax.set_title("c  The hold bonus, discounted")
    out += [["hold_bonus_present_value", int(x), "", round(float(HOLD_BONUS * GAMMA ** (TRIAL_END - x)), 4), "", ""]
            for x in (0, 80, 160, 260, 360, 460)]
    save(fig, "fig12_reward_structure", ["quantity", "group_or_step", "n", "median_or_value", "q25", "q75"], out)


def fig_pad_friction(D):
    fr = read_rows("diag_fresh_seeds/trial_friction.jsonl")
    if fr is None:
        NOTES.append("left out: pad friction figure, trial_friction.jsonl missing")
        return
    names = ["RL seed 1", "RL seed 2", "RL seed 3", "Hand-tuned A"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), sharey=True)
    out = []
    for ax, name in zip(axes, names):
        for setting, src in (("default", "locked"), ("noslip", "fresh"), ("elliptic", "fresh")):
            rows = D.get((src, setting, name))
            if rows is None:
                continue
            bins = {q: [0, 0] for q in range(5)}
            tables = []
            for obj in sorted({k[0] for k in rows}):
                keys = sorted((k for k in rows if k[0] == obj), key=lambda k: fr[k]["tip_mu_mean"])
                for i, k in enumerate(keys):
                    q = min(4, int(5 * i / len(keys)))
                    bins[q][0] += held(rows[k])
                    bins[q][1] += 1
                med = st.median(fr[k]["tip_mu_mean"] for k in keys)
                hi = [held(rows[k]) for k in keys if fr[k]["tip_mu_mean"] > med]
                lo = [held(rows[k]) for k in keys if fr[k]["tip_mu_mean"] <= med]
                tables.append((sum(hi), len(hi) - sum(hi), sum(lo), len(lo) - sum(lo)))
            xs = np.arange(1, 6)
            ys = [100 * bins[q][0] / bins[q][1] for q in range(5)]
            s = SETTINGS[setting]
            ax.plot(xs, ys, color=s["color"], lw=1.6)
            ax.plot(xs, ys, ls="none", marker=s["marker"], ms=4.5, mfc=s["color"], mec="white", mew=0.8)
            orr, p = cmh(tables)
            STATS.append(f"- Pad friction, {name}, {setting} ({src} set): held by within-object friction quintile, low to high: "
                         f"{', '.join(f'{y:.0f}%' for y in ys)}. Mantel-Haenszel odds ratio, above vs. below each object's median "
                         f"friction: {orr:.2f}, {fmt_p(p)}.")
            out += [[name, setting, src, q + 1, bins[q][0], bins[q][1], round(ys[q], 2)] for q in range(5)] + \
                   [[name, setting, src, "MH odds ratio high vs low", round(orr, 3), "", round(p, 4)]]
        ax.set_title(name)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["low", "2", "3", "4", "high"])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Trials held at 5 cm (%)")
    fig.supxlabel("Pad friction, quintile within each object", fontsize=8, color=INK2, y=-0.05)
    present = [s for s in ("default", "noslip", "elliptic") if any(r[1] == s for r in out)]
    axes[0].legend(handles=[handle(s, label=SHORT[s] + (", locked set" if s == "default" else ", fresh seeds"))
                            for s in present], loc="lower left", bbox_to_anchor=(0, 1.12), ncol=len(present))
    save(fig, "fig13_pad_friction", ["controller", "setting", "set", "quintile", "held", "n", "held_pct"], out)


def fig_training_curves(D):
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        NOTES.append("left out: training curves, tensorboard is not installed")
        return
    runs = [(1, "rl_logs_production/PPO_2", 300224), (2, "rl_logs_seedcheck/PPO_1", 300408), (3, "rl_logs_seed3/PPO_1", 300032)]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True, sharey="row")
    out = []
    finals = {}
    for j, (s, rel, last) in enumerate(runs):
        path = DATA / rel
        if not path.exists():
            NOTES.append(f"left out: training curve for seed {s}, {rel} missing")
            continue
        ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
        ea.Reload()
        rows = D[("locked", "default", f"RL seed {s}")]
        k, n = count(rows) if rows else (0, 1)
        for i, (tag, ylabel) in enumerate((("rollout/ep_rew_mean", "Mean episode reward"),
                                           ("rollout/ep_len_mean", "Mean episode length (RL steps)"))):
            # A run that crashed and resumed from a checkpoint logs some steps twice. Keep the resumed
            # segment: walk in wall-clock order and drop anything at or past the step it restarted from.
            pts = []
            for e in sorted(ea.Scalars(tag), key=lambda e: e.wall_time):
                while pts and pts[-1][0] >= e.step:
                    pts.pop()
                pts.append((e.step, e.value))
            pts = [p for p in pts if p[0] <= last]
            x, y = zip(*pts)
            ax = axes[i][j]
            ax.plot(np.array(x) / 1000, y, color=SETTINGS["default"]["color"], lw=1.4)
            if j == 0:
                ax.set_ylabel(ylabel)
            out += [[s, tag, int(xx), round(float(yy), 3)] for xx, yy in pts]
            if i == 0:
                ax.set_title(f"Seed {s}: {100 * k / n:.1f}% held, locked set", fontsize=8)
                finals[s] = (y[-1], 100 * k / n)
    for ax in axes[1]:
        ax.set_xlabel("Training steps (thousands)")
    save(fig, "fig14_training_curves", ["seed", "tag", "step", "value"], out)
    if len(finals) == 3:
        by_reward = sorted(finals, key=lambda s: -finals[s][0])
        by_held = sorted(finals, key=lambda s: -finals[s][1])
        STATS.append("- Final mean training reward: " + ", ".join(f"seed {s} {finals[s][0]:.0f}" for s in (1, 2, 3))
                     + f". Ranked by training reward: seeds {', '.join(map(str, by_reward))}; ranked by held on the locked set: "
                     f"seeds {', '.join(map(str, by_held))}.")


def fig_sensor_channel(D):
    rows = D.get(("dev", "default", "RL seed 1, traced"))
    if rows is None:
        return
    pairs = [(t, abs(nv)) for r in rows.values() for smp in r["samples"]
             for t, nv in zip(smp["tip_touch"], smp["tip_normal"]) if t > 0.5]
    t, nv = map(np.array, zip(*pairs))
    ratio = nv / t
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(t, np.maximum(nv, 1e-3), ls="none", marker="o", ms=2.2, color=SETTINGS["default"]["color"], alpha=0.3, mec="none")
    lim = (0.4, 150)
    ax.plot(lim, lim, color=INK2, lw=0.9)
    ax.text(0.45, 9.0, "y = x: where a true\nnormal-force channel would lie", fontsize=6.6, color=INK2, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(1e-3, 150)
    ax.set_xlabel("Touch sensor: actual normal force (N)")
    ax.set_ylabel("Channel named \"normal\", absolute value (N)")
    ax.text(0.03, 0.97, f"{len(t):,} samples, median ratio {100 * np.median(ratio):.0f}%\n"
                        f"10th-90th percentile {100 * np.percentile(ratio, 10):.0f}-{100 * np.percentile(ratio, 90):.0f}%",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=INK2)
    STATS.append(f"- Sensor channel, RL seed 1 dev traces: over {len(t):,} fingertip samples with touch above 0.5 N, the channel "
                 f"named normal reads a median {100 * np.median(ratio):.0f}% of the touch sensor (10th-90th percentile "
                 f"{100 * np.percentile(ratio, 10):.0f}-{100 * np.percentile(ratio, 90):.0f}%).")
    save(fig, "fig15_sensor_channel", ["touch_N", "abs_normal_channel_N"], [[round(a, 4), round(b, 4)] for a, b in zip(t, nv)])


def agreement_counts(rows):
    """Over the trials every controller ran: how many controllers hold each trial, and the held count if each
    object got its best controller. That best is picked on the same trials, so it is an optimistic ceiling."""
    keys = sorted(set.intersection(*[set(r) for r in rows]))
    counts = [sum(held(r[k]) for r in rows) for k in keys]
    hist = [sum(c == i for c in counts) for i in range(len(rows) + 1)]
    objects = sorted({k[0] for k in keys})
    best = sum(max(sum(held(r[k]) for k in keys if k[0] == o) for r in rows) for o in objects)
    return len(keys), hist, best


def fig_agreement(D):
    """How many controllers hold each trial: the locked set under default, then fresh seeds under elliptic
    once those runs are complete. Bars are shares of trials so the two sets compare; labels are counts."""
    panels = [("locked", "default", ["Hand-tuned A", "Hand-tuned B", "RL seed 1", "RL seed 2", "RL seed 3"],
               "a  Locked set, default", "Hand-tuned A and B, RL seeds 1-3"),
              ("fresh", "elliptic", ["Hand-tuned A", "RL seed 1", "RL seed 2", "RL seed 3", "Fixed grip"],
               "b  Fresh seeds, elliptic", "Hand-tuned A, RL seeds 1-3, fixed grip")]
    panels = [p for p in panels if all(D.get((p[0], p[1], nm)) is not None for nm in p[2])]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.8), sharey=True, squeeze=False)
    out, top = [], 0
    for ax, (set_, setting, names, title, who) in zip(axes[0], panels):
        n, hist, best = agreement_counts([D[(set_, setting, nm)] for nm in names])
        pcts = [100 * h / n for h in hist]
        top = max(top, max(pcts))
        ax.bar(range(len(hist)), pcts, width=0.6, color=SETTINGS[setting]["color"], zorder=2)
        for i, (h, p) in enumerate(zip(hist, pcts)):
            ax.text(i, p + 0.6, f"{h}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(range(len(hist)))
        ax.set_xlabel(f"Controllers that hold the trial (of {len(names)})")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title} ({n:,} trials)", pad=16)
        ax.text(0, 1.03, who, transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK2)
        out.extend([title[3:], i, h, round(p, 2)] for i, (h, p) in enumerate(zip(hist, pcts)))
        STATS.append(f"- {title[3:]}: {n - hist[0]} of {n} trials ({100 * (n - hist[0]) / n:.1f}%) are held by at least one of "
                     f"the {len(names)} controllers ({who}); {hist[0]} ({100 * hist[0] / n:.1f}%) by none; {hist[-1]} by all "
                     f"{len(names)}. Choosing the best controller for each object would hold {best} ({100 * best / n:.1f}%), "
                     f"picked on the same trials, so optimistic.")
    axes[0][0].set_ylabel("Share of trials (%)")
    axes[0][0].set_ylim(0, top * 1.15)
    if len(panels) == 2:  # like for like: the four controllers run on both sets
        common = ["Hand-tuned A", "RL seed 1", "RL seed 2", "RL seed 3"]
        for set_, setting in (("locked", "default"), ("fresh", "elliptic")):
            n, hist, best = agreement_counts([D[(set_, setting, nm)] for nm in common])
            STATS.append(f"- Same four controllers (hand-tuned A, RL seeds 1-3), {set_} set, {setting}: {hist[0]} of {n} "
                         f"({100 * hist[0] / n:.1f}%) held by none; best controller per object {best} ({100 * best / n:.1f}%).")
    save(fig, "fig16_controller_agreement", ["panel", "controllers_holding", "trials", "share_pct"], out)


def outcome_split(rows):
    fell = sum(r["formation_max_fall"] > 0.05 for r in rows.values())
    slipped = sum(r["formation_max_fall"] <= 0.05 and r["slipped"] for r in rows.values())
    return len(rows) - fell - slipped, slipped, fell


def fig_outcomes(D):
    """Where each setting's gain comes from: held, slipped after the grasp formed, or fell while it formed."""
    shades = [("Held", "#52514e", "white"), ("Slipped after the grasp formed", "#898781", "white"),
              ("Fell while the grasp formed", "#c3c2b7", INK)]
    names = [("RL seed 1", "RL seed 1"), ("RL seed 2", "RL seed 2"), ("RL seed 3", "RL seed 3"),
             ("Hand-tuned A", "Hand-tuned A"), ("TIRP_fast", "Fixed grip")]
    dev_rows = [(lab, s, "dev", D.get(("dev", s, key))) for key, lab in names for s in ("default", "noslip", "elliptic")]
    dev_rows = [r for r in dev_rows if r[3] is not None]
    fresh_rows = []
    for key, lab in names:
        fkey = "Fixed grip" if key == "TIRP_fast" else key
        for s, src in (("default", "locked"), ("elliptic", "fresh")):
            rows = D.get((src, s, fkey))
            if rows is not None:
                fresh_rows.append((lab, s, src, rows))
    panels = [("a  Dev set, the same 150 trials under each setting", dev_rows)]
    if any(r[2] == "fresh" for r in fresh_rows):
        panels.append(("b  Locked set under default against fresh seeds under elliptic", fresh_rows))
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 0.24 * sum(len(p[1]) for p in panels) + 1.3 * len(panels)),
                             squeeze=False, gridspec_kw={"height_ratios": [len(p[1]) for p in panels]})
    out = []
    for ax, (title, rows_list) in zip(axes[:, 0], panels):
        ys = list(range(len(rows_list)))[::-1]
        labels = []
        prev = None
        for y, (lab, setting, src, rows) in zip(ys, rows_list):
            held_n, slip_n, fell_n = outcome_split(rows)
            n = len(rows)
            left = 0.0
            for (name, color, ink), v in zip(shades, (held_n, slip_n, fell_n)):
                w = 100 * v / n
                ax.barh(y, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0, zorder=2)
                if w >= 7:
                    ax.text(left + w / 2, y, f"{100 * v / n:.0f}%", ha="center", va="center", fontsize=6.4, color=ink, zorder=3)
                left += w
            st_ = SETTINGS[setting]
            ax.plot([-2.2], [y], ls="none", marker=st_["marker"], ms=5, mfc=st_["color"], mec="white", mew=0.8, clip_on=False)
            labels.append(f"{lab}, {setting}" if lab != prev else setting)
            prev = lab
            out.append([title[3:], lab, setting, src, n, held_n, slip_n, fell_n])
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows_list) - 0.4)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, pad=4)
    axes[-1, 0].set_xlabel("Share of trials (%)")
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[Patch(facecolor=c, edgecolor="white", label=nm) for nm, c, _ in shades]
                      + [handle(s) for s in ("default", "noslip", "elliptic")],
                      loc="lower left", bbox_to_anchor=(0, 1.06), ncol=3, fontsize=6.8)
    save(fig, "fig17_outcome_breakdown", ["panel", "controller", "setting", "set", "n", "held", "slipped_after_forming",
                                          "fell_while_forming"], out)
    for key, lab in names:
        base = D.get(("dev", "default", key))
        for s in ("noslip", "elliptic"):
            rows = D.get(("dev", s, key))
            if base is None or rows is None:
                continue
            keys = sorted(set(base) & set(rows))
            only_def = sum(base[k]["formation_max_fall"] > 0.05 and rows[k]["formation_max_fall"] <= 0.05 for k in keys)
            only_new = sum(rows[k]["formation_max_fall"] > 0.05 and base[k]["formation_max_fall"] <= 0.05 for k in keys)
            b, sl, f = outcome_split(base)
            b2, sl2, f2 = outcome_split(rows)
            STATS.append(f"- Dev set, {lab}, default to {s}: fell while forming {f} to {f2} (fell only under default {only_def}, "
                         f"only under {s} {only_new}, {fmt_p(mcnemar(only_def, only_new))}); slipped after forming {sl} to {sl2}; "
                         f"held {b} to {b2}.")


def paired_diff(ra, rb, cm_a=5, cm_b=5, reps=4000):
    """Held-rate difference a minus b in points over the trials both ran, with a 95% paired
    bootstrap interval (trials resampled together, so the pairing is kept)."""
    keys = sorted(set(ra) & set(rb))
    a = np.array([held(ra[k], cm_a) for k in keys], dtype=float)
    b = np.array([held(rb[k], cm_b) for k in keys], dtype=float)
    d = a - b
    idx = np.random.default_rng(0).integers(0, len(d), (reps, len(d)))
    boots = d[idx].mean(axis=1)
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), len(keys)


def fig_factor_summary(D):
    """One figure for the paper's argument: how much each factor moves the held rate. Where the fresh
    seeds give the same paired comparison under the primary elliptic settings, it sits on the same row
    in blue, so the reader can see whether the ranking holds there. Elliptic-minus-default rows are
    neutral gray, since blue means "under elliptic" everywhere else in the paper."""
    L, A, V, F, E = "locked", "all67", "dev", "fresh", "elliptic"
    spec = [  # label, kind, a, b, cm for a, cm for b, set, the same comparison on fresh elliptic seeds (or None)
        ("RL training seed: seed 1 vs. seed 3", "default", (L, "default", "RL seed 1"), (L, "default", "RL seed 3"), 5, 5,
         "locked set", ((F, E, "RL seed 1"), (F, E, "RL seed 3"))),
        ("Controller: best RL seed vs. hand-tuned A", "default", (L, "default", "RL seed 1"), (L, "default", "Hand-tuned A"),
         5, 5, "locked set", ((F, E, "RL seed 1"), (F, E, "Hand-tuned A"))),
        ("Grip: four fingers vs. all five (fixed grips)", "default", (V, "default", "TIRP_fast"), (V, "default", "TIMRP_fast"),
         5, 5, "dev set", None),
        ("Success definition: no drop rule vs. 5 cm rule", "default", (A, "default", "Hand-tuned A"),
         (A, "default", "Hand-tuned A"), None, 5, "seeds 0-66", ((F, E, "Hand-tuned A"), (F, E, "Hand-tuned A"))),
        ("Contact settings: elliptic vs. default, RL seed 1", "between", (V, E, "RL seed 1"), (V, "default", "RL seed 1"),
         5, 5, "dev set", None),
        ("Contact settings: elliptic vs. default, RL seed 2", "between", (V, E, "RL seed 2"), (V, "default", "RL seed 2"),
         5, 5, "dev set", None),
        ("Learning: best RL seed vs. fixed grip", "default", (V, "default", "RL seed 1"), (V, "default", "TIRP_fast"), 5, 5,
         "dev set", ((F, E, "RL seed 1"), (F, E, "Fixed grip"))),
        ("Detector quality: F1 0.799 vs. 0.691", "default", (A, "default", "Hand-tuned B"), (A, "default", "Hand-tuned A"),
         5, 5, "seeds 0-66", None),
        ("Detector: with vs. without TCN, hand-tuned A", "default", (V, "default", "Hand-tuned A"),
         (V, "default", "Hand-tuned A, no TCN"), 5, 5, "dev set", None),
        ("Detector: with vs. without signal, RL seed 1", "default", (V, "default", "RL seed 1"),
         (V, "default", "RL seed 1, no TCN"), 5, 5, "dev set", None),
    ]
    look = {"default": (SETTINGS["default"]["color"], SETTINGS["default"]["marker"]),
            "elliptic": (SETTINGS["elliptic"]["color"], SETTINGS["elliptic"]["marker"]),
            "between": (INK2, "^")}
    rows = []
    for label, kind, ka, kb, cma, cmb, where, fresh in spec:
        ra, rb = D.get(ka), D.get(kb)
        if ra is None or rb is None:
            NOTES.append(f"left out of the factor summary: {label}")
            continue
        ests = [(kind, *paired_diff(ra, rb, cma, cmb), where)]
        if fresh is not None and D.get(fresh[0]) is not None and D.get(fresh[1]) is not None:
            ests.append(("elliptic", *paired_diff(D[fresh[0]], D[fresh[1]], cma, cmb), "fresh seeds, elliptic"))
        rows.append((label, ests))
    rows.sort(key=lambda r: -abs(r[1][0][1]))
    two = any(len(e) > 1 for _, e in rows)
    fig, ax = plt.subplots(figsize=(7.2, (0.48 if two else 0.36) * len(rows) + 0.9))
    ys = forest_axes(ax, [r[0] for r in rows], "Change in trials held at 5 cm (percentage points), same trials on both sides",
                     xlim=(-10, 60))
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    out = []
    for y, (label, ests) in zip(ys, rows):
        for off, (kind, m, lo, hi, n, where) in zip([0.0] if len(ests) == 1 else [0.17, -0.17], ests):
            c, mk = look[kind]
            ax.plot([lo, hi], [y + off, y + off], color=c, lw=1.5, zorder=2)
            ax.plot([m], [y + off], ls="none", marker=mk, ms=5.5, mfc=c, mec="white", mew=0.9, zorder=3)
            right_text(ax, y + off, f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]   {where}, n = {n:,}")
            out.append([label, kind, round(m, 2), round(lo, 2), round(hi, 2), n, where])
            STATS.append(f"- {label} ({where}): {m:+.1f} points (95% paired bootstrap interval {lo:+.1f} to {hi:+.1f}), "
                         f"{n} paired trials.")
    names = (("default", "Under MuJoCo default settings"), ("elliptic", "Under elliptic settings, fresh seeds"),
             ("between", "Elliptic minus default, same trials"))
    ax.legend(handles=[Line2D([0], [0], color=look[k][0], lw=1.5, marker=look[k][1], ms=5.5, mfc=look[k][0], mec="white",
                              mew=0.9, label=t) for k, t in names if two or k != "elliptic"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    save(fig, "fig00_factor_summary", ["comparison", "kind", "difference_points", "ci_low", "ci_high", "n_pairs", "set"], out)


def fig_dev_vs_fresh(D):
    """The pre-registered check: does each dev-set figure fall inside its fresh-seed 95% interval?
    Drawn as dev minus fresh, so the fresh interval is a band around zero."""
    names = ["RL seed 1", "RL seed 2", "RL seed 3", "Hand-tuned A"]
    pairs = [(n, s) for s in ("elliptic", "noslip") for n in names
             if D.get(("dev", s, n)) is not None and D.get(("fresh", s, n)) is not None]
    if not pairs:
        NOTES.append("left out: dev-vs-fresh figure, no controller has both a dev and a fresh run yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(pairs) + 1.3), sharey=True)
    out = []
    for ax, cm in zip(axes, (5, 10)):
        ys = forest_axes(ax, [f"{n}, {s}" for n, s in pairs], f"Dev-set figure minus fresh-seed figure, {cm} cm (points)",
                         xlim=(-10, 10))
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for y, (n, s) in zip(ys, pairs):
            dev, fresh = D[("dev", s, n)], D[("fresh", s, n)]
            kf, nf = count(fresh, cm)
            kd, nd = count(dev, cm)
            pf, lo, hi = wilson(kf, nf)
            pd = 100 * kd / nd
            c = SETTINGS[s]["color"]
            ax.plot([lo - pf, hi - pf], [y, y], color=c, lw=7, alpha=0.3, solid_capstyle="butt", zorder=2)
            ax.plot([pd - pf], [y], ls="none", marker=SETTINGS[s]["marker"], ms=6.5, mfc="white", mec=c, mew=1.4, zorder=4)
            inside = lo <= pd <= hi
            right_text(ax, y, f"{pd - pf:+.1f}, {'inside' if inside else 'outside'}")
            out.append([n, s, cm, kf, nf, round(pf, 2), round(lo, 2), round(hi, 2), kd, nd, round(pd, 2), inside])
            STATS.append(f"- {n}, {s}, {cm} cm: fresh {kf}/{nf} = {pf:.2f}% [{lo:.2f}, {hi:.2f}]; dev {kd}/{nd} = {pd:.2f}%, "
                         f"{'inside' if inside else 'OUTSIDE'} the fresh interval.")
        ax.set_title(f"{'a' if cm == 5 else 'b'}  {cm} cm rule", pad=22)
    axes[0].legend(handles=[Line2D([0], [0], color=INK2, lw=7, alpha=0.3, label="Fresh seeds, 95% interval"),
                            Line2D([0], [0], ls="none", marker="o", ms=6.5, mfc="white", mec=INK2, mew=1.4,
                                   label="Dev set")],
                   loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "fig18_dev_vs_fresh", ["controller", "setting", "cutoff_cm", "fresh_held", "fresh_n", "fresh_pct",
                                     "fresh_ci_low", "fresh_ci_high", "dev_held", "dev_n", "dev_pct", "dev_inside"], out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    D = load_all()
    STATS.append("# Statistics behind the paper figures\n")
    STATS.append("Generated by `src/analysis/paper_figures.py`. Rates are k/n with 95% Wilson intervals. Paired tests are exact "
                 "McNemar on the same trials. Locked set = seeds 3-66 (960 trials); dev = seeds 1000-1009 (150); fresh = seeds "
                 "2000-2066 (1,005).\n")
    for f in (fig_factor_summary, fig_success_definition, fig_formation_falls, fig_detector_vs_closed_loop, fig_tcn_removal,
              fig_rl_seed_spread, fig_fixed_grips, fig_contact_settings_dev, fig_per_object, fig_survival,
              fig_creep, fig_finger_closure, fig_reward, fig_pad_friction, fig_training_curves,
              fig_sensor_channel, fig_agreement, fig_outcomes, fig_dev_vs_fresh):
        STATS.append(f"\n## {f.__name__[4:].replace('_', ' ').capitalize()}\n")
        f(D)
    if NOTES:
        STATS.append("\n## Left out\n")
        STATS.extend(f"- {n}" for n in NOTES)
    (OUT / "stats_summary.md").write_text("\n".join(s for s in STATS if s is not None) + "\n", encoding="utf-8")
    print(f"made {len(MADE)} figures in {OUT}")
    for n in NOTES:
        print("NOTE", n)


if __name__ == "__main__":
    main()
