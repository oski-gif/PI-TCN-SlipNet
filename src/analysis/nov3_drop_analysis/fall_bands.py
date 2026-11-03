"""Read-only: in recorded training trials, what is the object doing at the end of settle, by fall size?"""
import json
import os
import random
from collections import defaultdict

import numpy as np

items = [i for i in json.load(open("Data/manifest_full_merged.json")) if i["ever_contacted"]]
random.seed(2)
sample = random.sample(items, 1500)

names = None
bands = [(0, .05), (.05, .07), (.07, .15), (.15, 1)]
out = defaultdict(lambda: defaultdict(list))
for it in sample:
    p = it["data_path"]
    if not os.path.exists(p):
        p = os.path.join("Data", os.path.basename(os.path.dirname(p)), os.path.basename(p))
    z = np.load(p, allow_pickle=True)
    if names is None:
        names = [str(n) for n in z["feature_names"]]
        print("features:", names)
        force_cols = [i for i, n in enumerate(names) if "force" in n.lower() or "touch" in n.lower()]
        print("force columns:", [names[i] for i in force_cols])
    fm = z["feature_matrix"]
    oz, ph = fm[:, names.index("obj_z")], fm[:, names.index("phase")]
    formation = np.where(ph < 2)[0]
    if len(formation) < 20:
        continue
    fall = float(oz[0] - oz[formation].min())
    end = formation[-1]
    speed_col = next((i for i, n in enumerate(names) if "speed" in n.lower()), None)
    b = next(x for x in bands if x[0] < fall <= x[1]) if fall > 0 else bands[0]
    rec = out[b]
    rec["fall"].append(fall)
    rec["label"].append(int(z["trial_slip_label"]))
    rec["squeeze"].append(float(it["squeeze_intensity"]))
    # Vertical drift over the last 50 formation steps: near 0 means the object has stopped.
    rec["z_drift_last50"].append(abs(float(oz[end] - oz[max(end - 50, 0)])))
    # Fingers pressing on the object at the end of settle.
    forces = np.abs(fm[end, force_cols])
    rec["n_force_gt_0.01"].append(int((forces > 0.01).sum()))
    rec["sum_force"].append(float(forces.sum()))
    rec["z_end_minus_start"].append(float(oz[end] - oz[0]))

nos={b:sum(1 for l in out[b]["label"] if l==0) for b in bands}; tot=sum(nos.values()); print("no-slip by band:",{f"{100*b[0]:.0f}-{100*b[1]:.0f}":(nos[b],round(100*nos[b]/tot,1)) for b in bands},"total",tot); slp={b:sum(out[b]["label"]) for b in bands}; print("slip by band:",{f"{100*b[0]:.0f}-{100*b[1]:.0f}":slp[b] for b in bands})
print(f"{'band_cm':<10}{'n':>5}{'slip%':>7}{'med_squeeze':>12}{'med_drift_cm':>13}{'med_contacts':>13}{'frac_0_contacts':>16}{'med_force':>10}")
for b in bands:
    r = out[b]
    n = len(r["fall"])
    if not n:
        continue
    contacts = np.array(r["n_force_gt_0.01"])
    print(f"{100*b[0]:.0f}-{100*b[1]:<7.0f}{n:>5}{100*np.mean(r['label']):>7.1f}{np.median(r['squeeze']):>12.2f}"
          f"{100*np.median(r['z_drift_last50']):>13.3f}{np.median(contacts):>13.1f}{np.mean(contacts == 0):>16.2f}"
          f"{np.median(r['sum_force']):>10.3f}")
