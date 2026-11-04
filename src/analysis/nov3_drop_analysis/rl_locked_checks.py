"""Read-only: applies the pre-registered checks to the finished RL evaluation runs.
Covers the locked set, the 2.2% eval-path check and deterministic vs. stochastic on
the dev set. Rules: results.md, "Pre-registered evaluation rules". Run from the repo root."""
import json
import math
import statistics as st

HOLDOUT = {"cone", "cylindrical_prism", "plus_cross", "sphere"}
CUTOFFS = (0.05, 0.10)
SEEDS = ("seed1", "seed2", "seed3")
PRIMARY = lambda k: k[1] >= 3  # seeds 0-2 were already used to compare and choose policies
EVERY = lambda k: True


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load(path):
    return {(r["object"], int(r["seed"])): r for r in read_rows(path) if "error" not in r}


def held(r, cutoff):
    return (not r["slipped"]) and r["formation_max_fall"] <= cutoff


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def t2_p(t):
    # Two-sided p for Student's t with 2 degrees of freedom, which has a closed form.
    return 1 - abs(t) / math.sqrt(t * t + 2)


def pct(D, cutoff, keep):
    ks = [k for k in D if keep(k)]
    return 100 * sum(held(D[k], cutoff) for k in ks) / len(ks)


A = load("Data/handtuned_modelA_2300step_results.jsonl")
B = load("Data/handtuned_modelB_2300step_results.jsonl")
L = {s: load(f"Data/rl_{s}_1005trial_results.jsonl") for s in SEEDS}

print("Integrity")
for s in SEEDS:
    rows = read_rows(f"Data/rl_{s}_1005trial_results.jsonl")
    ok = [r for r in rows if "error" not in r]
    print(f"  {s}: rows {len(rows)}, errors {len(rows) - len(ok)}, unique {len({(r['object'], r['seed']) for r in ok})}, "
          f"policy {ok[0]['policy_sha256'][:8]}, TCN {ok[0]['checkpoint_sha256'][:8]}, deterministic {ok[0]['deterministic']}, "
          f"code hash sets {len({tuple(r['code_sha256']) for r in ok})}, "
          f"stored held vs 5 cm recompute mismatches {sum(bool(r['held']) != held(r, 0.05) for r in ok)}")

print("Locked set")
for cutoff in CUTOFFS:
    for label, keep in (("seeds 3-66", PRIMARY), ("all 1,005", EVERY)):
        vals = []
        for s, D in L.items():
            ks = [k for k in D if keep(k)]
            h = sum(held(D[k], cutoff) for k in ks)
            lo, hi = wilson(h, len(ks))
            vals.append(100 * h / len(ks))
            print(f"  {s} {int(cutoff * 100)} cm {label}: {h}/{len(ks)} {100 * h / len(ks):.1f}% CI {lo:.1f}-{hi:.1f}")
        print(f"    mean {st.mean(vals):.1f}, SD {st.stdev(vals):.1f}")
        for name, D in (("A", A), ("B", B)):
            ks = [k for k in D if keep(k)]
            h = sum(held(D[k], cutoff) for k in ks)
            print(f"    hand-tuned {name}: {h}/{len(ks)} {100 * h / len(ks):.1f}%")

print("Paired with hand-tuned on the same trials (RL only, hand-tuned only, exact McNemar p)")
for cutoff in CUTOFFS:
    for label, keep in (("seeds 3-66", PRIMARY), ("all 1,005", EVERY)):
        for s, D in L.items():
            for name, H in (("A", A), ("B", B)):
                ks = [k for k in D if keep(k)]
                b = sum(held(D[k], cutoff) and not held(H[k], cutoff) for k in ks)
                c = sum(held(H[k], cutoff) and not held(D[k], cutoff) for k in ks)
                print(f"  {int(cutoff * 100)} cm {label} {s} vs {name}: {b}, {c}, p={mcnemar_exact(b, c):.2e}")

print("Three-repeat mean vs. hand-tuned A, seeds 3-66, one-sample t, df 2 (not pre-registered)")
for cutoff in CUTOFFS:
    vals = [pct(D, cutoff, PRIMARY) for D in L.values()]
    base = pct(A, cutoff, PRIMARY)
    m, sd = st.mean(vals), st.stdev(vals)
    t = (m - base) / (sd / math.sqrt(len(vals)))
    print(f"  {int(cutoff * 100)} cm: mean {m:.1f}, SD {sd:.1f}, hand-tuned {base:.1f}, difference {m - base:.1f}, t={t:.2f}, p={t2_p(t):.2f}")

print("Outcomes at 5 cm, all 1,005")
for s, D in L.items():
    print(f"  {s}: held {sum(held(r, .05) for r in D.values())}, "
          f"fell past 5 cm {sum(r['formation_max_fall'] > .05 for r in D.values())}, "
          f"slipped without falling {sum(r['slipped'] and r['formation_max_fall'] <= .05 for r in D.values())}")

print("Training vs. held-out objects at 5 cm, seeds 3-66")
for name, D in (*L.items(), ("hand-tuned A", A)):
    for group, test in (("training", lambda o: o not in HOLDOUT), ("held-out", lambda o: o in HOLDOUT)):
        ks = [k for k in D if k[1] >= 3 and test(k[0])]
        h = sum(held(D[k], .05) for k in ks)
        print(f"  {name} {group}: {h}/{len(ks)} {100 * h / len(ks):.1f}%")

print("Held per object at 5 cm, all seeds, of 67")
print(f"  {'object':<20}" + "".join(f"{s:>7}" for s in L) + f"{'A':>6}")
for obj in sorted({k[0] for k in A}):
    counts = [sum(held(r, .05) for k, r in D.items() if k[0] == obj) for D in (*L.values(), A)]
    print(f"  {obj:<20}" + "".join(f"{c:>7}" for c in counts[:-1]) + f"{counts[-1]:>6}")

print("2.2% policy eval-path check, original 45 trials")
direct, wrapped, sampled = (load(f"Data/evalpath_{x}.jsonl") for x in ("direct", "wrapped", "wrapped_stochastic"))
print(f"  held: direct {sum(r['held'] for r in direct.values())}, wrapped {sum(r['held'] for r in wrapped.values())}, "
      f"wrapped stochastic {sum(r['held'] for r in sampled.values())}")
agree = sum(direct[k]["held"] == wrapped[k]["held"] for k in direct)
print(f"  direct and wrapped agree on {agree}/{len(direct)} trials (rule: at least 43), "
      f"largest reward difference {max(abs(direct[k]['ep_reward'] - wrapped[k]['ep_reward']) for k in direct):.1e}")
train_rows = [r for k, r in sampled.items() if k[0] not in HOLDOUT]
rew, length = st.mean(r["ep_reward"] for r in train_rows), st.mean(r["ep_len"] for r in train_rows)
print(f"  stochastic wrapper, training objects: reward {rew:.1f} vs 334 ({100 * (rew / 334 - 1):+.1f}%), "
      f"length {length:.1f} vs 162 ({100 * (length / 162 - 1):+.1f}%) (rule: within 25%)")

print("Deterministic vs. stochastic, dev set, seeds 1000-1009")
dev = {}
for s in SEEDS:
    det, sto = load(f"Data/rl_{s}_dev_deterministic.jsonl"), load(f"Data/rl_{s}_dev_stochastic.jsonl")
    dev[s] = det
    same = all(det[k]["object_pos"] == sto[k]["object_pos"] and det[k]["squeeze_intensity"] == sto[k]["squeeze_intensity"] for k in det)
    print(f"  {s}: {len(det)} and {len(sto)} trials, same keys {set(det) == set(sto)}, same start conditions {same}")
    for cutoff in CUTOFFS:
        hd = sum(held(det[k], cutoff) for k in det)
        hs = sum(held(sto[k], cutoff) for k in det)
        b = sum(held(det[k], cutoff) and not held(sto[k], cutoff) for k in det)
        c = sum(held(sto[k], cutoff) and not held(det[k], cutoff) for k in det)
        n = len(det)
        diff, limit = abs(hd - hs) / n, 2 * math.sqrt(b + c) / n
        print(f"    {int(cutoff * 100)} cm: deterministic {hd}/{n} {100 * hd / n:.1f}%, stochastic {hs}/{n} {100 * hs / n:.1f}%, "
              f"only det {b}, only sto {c}, difference {100 * diff:.1f}, limit {100 * limit:.1f}, "
              f"{'TRIPS' if diff > limit else 'ok'}, p={mcnemar_exact(b, c):.3f}")

print("Baseline dev scores for the longer-rollout verdict, deterministic")
for cutoff in CUTOFFS:
    vals = [100 * sum(held(r, cutoff) for r in D.values()) / len(D) for D in dev.values()]
    print(f"  {int(cutoff * 100)} cm: {', '.join(f'{v:.1f}' for v in vals)}; mean {st.mean(vals):.1f}, SD {st.stdev(vals):.1f}")
