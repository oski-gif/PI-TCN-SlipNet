"""Read-only: summarizes the output of diagnostic_contact_physics.py. Pass its .jsonl path.
Descriptive only: nothing here picks a policy or changes a benchmark. Run from the repo root."""
import json
import statistics as st
import sys
from collections import defaultdict

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
by_variant = defaultdict(list)
for r in rows:
    by_variant[r["variant"]].append(r)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def med(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else float("nan")


print("Per variant: trials that reached full pull with enough time to measure")
print(f"{'variant':<22}{'held':>7}{'fell>5cm':>9}{'slipped':>8}{'n':>5}{'speed mm/s':>12}{'tips mm':>9}{'friction':>10}"
      f"{'cap/load':>10}{'contacts':>10}{'force balance':>15}")
for variant, group in by_variant.items():
    fl = [r["full_load"] for r in group if r["full_load"]]
    balance = [(f["up_normal"] + f["up_friction"]) / f["load_expected"] for f in fl]
    print(f"{variant:<22}{sum(r['held'] for r in group):>4}/{len(group):<2}{sum(r['dropped_during_formation'] for r in group):>9}"
          f"{sum(r['slipped'] for r in group):>8}{len(fl):>5}{med([f['object_speed_mm_s'] for f in fl]):>12.2f}"
          f"{med([f['tip_move_mm_median'] for f in fl]):>9.2f}{med([f['friction_used'] for f in fl]):>10.3f}"
          f"{med([f['capacity_over_load'] for f in fl]):>10.1f}{med([f['n_contacts'] for f in fl]):>10.1f}{med(balance):>15.2f}")

if "default" not in by_variant:
    sys.exit()
default = by_variant["default"]

print("\nDefault physics, per trial, measured from full pull to the end of the trial")
print(f"{'object':<19}{'held':<6}{'fall mm':>8}{'disp mm':>8}{'limit':>6}{'mm/s':>7}{'tips mm':>8}{'used':>6}{'cap/load':>9}"
      f"{'contacts':>9}{'friction carries':>17}  tips touching")
for r in default:
    f = r["full_load"]
    base = f"{r['object']:<19}{str(r['held']):<6}{1000 * r['formation_max_fall']:>8.1f}{1000 * r['max_displacement']:>8.1f}{1000 * r['threshold']:>6.1f}"
    if f is None:
        print(base + "   ended before full pull could be measured")
        continue
    total = f["up_friction"] + f["up_normal"]
    share = f["up_friction"] / total if total else float("nan")
    names = f["bodies_touching"]
    floor = "  ON FLOOR" if "world" in names else ""
    print(base + f"{f['object_speed_mm_s']:>7.2f}{(f['tip_move_mm_median'] or 0):>8.2f}{(f['friction_used'] or 0):>6.2f}"
          f"{f['capacity_over_load']:>9.1f}{f['n_contacts']:>9.1f}{100 * share:>16.0f}%  {f['n_tips_touching']} of {len(names)} bodies{floor}")

print("\nDefault physics: commanded closure per finger (0 open, 1 fully closed), mean across trials")
marks = (10, 20, 50, 100, 160, 260, 360, 460)
print(f"{'RL step':>8}" + "".join(f"{n:>9}" for n in FINGERS))
for mark in marks:
    values = [next(s for s in r["samples"] if s["rl_step"] >= mark)["fraction"]
              for r in default if any(s["rl_step"] >= mark for s in r["samples"])]
    if values:
        print(f"{mark:>8}" + "".join(f"{st.mean(v[i] for v in values):>9.2f}" for i in range(5)))
spread = {i: [] for i in range(5)}
for r in default:
    s = next((s for s in r["samples"] if s["rl_step"] >= 160), None)
    if s:
        for i in range(5):
            spread[i].append(s["fraction"][i])
print("  at settle end, min and max across objects: " + ", ".join(
    f"{FINGERS[i]} {min(spread[i]):.2f}-{max(spread[i]):.2f}" for i in range(5)))

print("\nDefault physics: raw policy mean action per finger, before clipping to [-1, 1]")
for label, keep in (("closing, RL steps up to 50", lambda s: s["rl_step"] <= 50),
                    ("rest of formation", lambda s: 50 < s["rl_step"] <= 160),
                    ("pull", lambda s: s["rl_step"] > 160)):
    values = [s["raw_mean_action"] for r in default for s in r["samples"] if keep(s)]
    print(f"  {label:<28}" + "  ".join(
        f"{FINGERS[i]} {st.mean(v[i] for v in values):+.2f} ({min(v[i] for v in values):+.2f},{max(v[i] for v in values):+.2f})"
        for i in range(5)))

print("\nDefault physics: does anything change once the pull starts?")
moved, total = 0, 0
for r in default:
    pull = [s for s in r["samples"] if s["rl_step"] >= 160]
    if len(pull) >= 2:
        total += 1
        moved += any(abs(a - b) > 0.01 for x, y in zip(pull, pull[1:]) for a, b in zip(x["fraction"], y["fraction"]))
print(f"  trials where any finger's commanded closure changed by more than 0.01 during the pull: {moved}/{total}")
probs = [s["slip_probability"] for r in default for s in r["samples"] if s["rl_step"] > 160]
print(f"  TCN slip probability during the pull: median {med(probs):.2f}, min {min(probs):.2f}, max {max(probs):.2f}")

print("\nDefault physics: the grip-force reward's reading at settle end, in multiples of object weight")
readings, negatives = [], 0
for r in default:
    s = next((s for s in r["samples"] if s["rl_step"] >= 160), None)
    if s:
        readings.append(s["grip_force_sum"] / r["object_weight"])
        negatives += s["grip_force_sum"] < 0
print(f"  median {med(readings):.1f}, min {min(readings):.1f}, max {max(readings):.1f}; negative in {negatives}/{len(readings)}; "
      f"the reward caps this at 2.0 and has no lower bound")
sample = next((s for r in default if r["held"] for s in r["samples"] if s["rl_step"] >= 160), None)
if sample:
    print(f"  one held trial at settle end, per fingertip: touch {sample['tip_touch']}, normal {sample['tip_normal']}, "
          f"contact force by body {sample['contact']['fn_by_body']}, actuator force {sample['actuator_force']}")
