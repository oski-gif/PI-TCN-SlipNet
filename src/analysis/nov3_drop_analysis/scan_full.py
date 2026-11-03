import json, random, os, collections, zipfile
import numpy as np
zf = zipfile.ZipFile('Data/pitcn_dataset_full_random_split.npz'); print('dataset arrays', [n[:-4] for n in zf.namelist()][:30])
items = json.load(open('Data/manifest_full_merged.json'))
print('full manifest', len(items), 'labels', dict(collections.Counter(i['slip_label'] for i in items)), 'ever_contacted', dict(collections.Counter(i['ever_contacted'] for i in items)))
items = [i for i in items if i['ever_contacted']]
random.seed(1)
sample = random.sample(items, 1000)
res = collections.Counter(); falls = {0: [], 1: []}; hi = collections.Counter()
for it in sample:
    p = it['data_path']
    if not os.path.exists(p):
        p = os.path.join('Data', os.path.basename(os.path.dirname(p)), os.path.basename(p))
    z = np.load(p, allow_pickle=True)
    names = [str(n) for n in z['feature_names']]
    fm = z['feature_matrix']
    oz, ph = fm[:, names.index('obj_z')], fm[:, names.index('phase')]
    fall = float(oz[0] - oz[ph < 2].min())
    label = int(z['trial_slip_label']); d = fall > 0.05
    res[(label, d)] += 1; falls[label].append(fall)
    if it['squeeze_intensity'] >= 0.9: hi[(label, d)] += 1
for lab in (0, 1):
    n = res[(lab, True)] + res[(lab, False)]
    print(f'contact-only label={lab} n={n} dropped={res[(lab, True)]} ({100*res[(lab, True)]/max(n,1):.1f}%) median_fall={np.median(falls[lab]):.4f}')
print('squeeze>=0.9:', dict(hi))
