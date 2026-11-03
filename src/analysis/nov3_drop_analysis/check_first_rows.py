"""Read-only: do the first locked-set rows match the same trials in the 45-trial re-score?"""
import json

old = {}
for line in open("Data/rescore_45_seed1.jsonl", encoding="utf-8"):
    r = json.loads(line)
    old[(r["object"], r["seed"])] = r
rows = [json.loads(line) for line in open("Data/rl_seed1_1005trial_results.jsonl", encoding="utf-8") if line.strip()]
print("rows", len(rows), "errors", sum("error" in r for r in rows))
good = [r for r in rows if "error" not in r]
shared = [r for r in good if (r["object"], r["seed"]) in old]
fields = ("held", "slipped", "dropped_during_formation", "formation_max_fall", "max_displacement", "ep_len")
bad = [(r["object"], r["seed"], f) for r in shared for f in fields if r[f] != old[(r["object"], r["seed"])][f]]
print("shared trials", len(shared), "field mismatches", bad)
if good:
    print("config:", {k: good[0][k] for k in ("deterministic", "continue_past_formation_drop", "seed_start", "n_trials_per_object", "total_steps", "checkpoint_sha256")})
    print("policy sha", good[0]["policy_sha256"][:8], "| held so far", sum(r["held"] for r in good), "of", len(good))
