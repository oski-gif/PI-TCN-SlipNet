import json, re, math, collections
import numpy as np
def load(p):
    d={}
    for l in open(p,encoding='utf-8'):
        if l.strip():
            r=json.loads(l)
            if 'error' not in r: d[(r['object'],int(r['seed']))]=r
    return d
A=load('Data/handtuned_modelA_2300step_results.jsonl'); B=load('Data/handtuned_modelB_2300step_results.jsonl')
fall={k:A[k]['formation_max_fall'] for k in A}
print('fall bands, trials: <=5',sum(v<=.05 for v in fall.values()),'5-15',sum(.05<v<=.15 for v in fall.values()),'>15',sum(v>.15 for v in fall.values()))
def held(D,k,T): return (not D[k]['slipped']) and (T is None or fall[k]<=T)
def line(name,D,keys):
    keys=list(keys); n=len(keys)
    print(f"{name:<26} n={n:>4} none={sum(held(D,k,None) for k in keys):>4} ({100*sum(held(D,k,None) for k in keys)/n:5.1f}%)  15cm={sum(held(D,k,.15) for k in keys):>4} ({100*sum(held(D,k,.15) for k in keys)/n:5.1f}%)  10cm={sum(held(D,k,.10) for k in keys):>4}  5cm={sum(held(D,k,.05) for k in keys):>4} ({100*sum(held(D,k,.05) for k in keys)/n:5.1f}%)")
line('A 2300 all',A,A); line('B 2300 all',B,B)
line('A 2300 seeds 3-66',A,[k for k in A if k[1]>=3]); line('B 2300 seeds 3-66',B,[k for k in B if k[1]>=3])
line('A 2300 seeds 0-2',A,[k for k in A if k[1]<=2]); line('B 2300 seeds 0-2',B,[k for k in B if k[1]<=2])
for name,p in [('A 1500','Data/baseline_modelA_1000trial_results.jsonl'),('B 1500','Data/baseline_modelB_1000trial_results.jsonl'),('A 2500 45','Data/closed_loop_scenes/batch_results.jsonl'),('B 2500 45','Data/closed_loop_scenes/batch_results_modelb.jsonl')]:
    D=load(p); line(name,D,D)
for T,lab in [(.05,'5cm'),(.15,'15cm')]:
    ao=sum(held(A,k,T) and not held(B,k,T) for k in A); bo=sum(held(B,k,T) and not held(A,k,T) for k in A)
    print('2300 paired',lab,'A only',ao,'B only',bo)
print('band: held (A) of trials',[(lab,sum(held(A,k,None) for k in fall if lo<fall[k]<=hi),sum(lo<fall[k]<=hi for k in fall)) for lab,lo,hi in [('<=5',-1,.05),('5-15',.05,.15),('>15',.15,9)]])
per=collections.defaultdict(lambda:[0,0,0,0,0])
for k in A:
    p=per[k[0]]; p[0]+=1; p[1]+=fall[k]>.05; p[2]+=fall[k]>.15; p[3]+=held(A,k,.05); p[4]+=held(A,k,.15)
print('object: n, >5cm, >15cm, held5, held15')
for o,v in sorted(per.items(),key=lambda x:-x[1][4]): print(' ',o,v)
# F1 check of worker output
j=open('Data/_tcn_f1_subsets.log').read()
for m in re.finditer(r'^\s+(all trials|no drop|drop)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+(\S+)\s+(\S+) .*\s(0\.\d+) \(\S+\)$',j,re.M):
    s,tp,fp,tn,fn=m.group(1),*map(int,m.group(2,3,4,5))
    print(s,'recomputed f1',round(2*tp/(2*tp+fp+fn),3),'logged',m.group(8))
