"""Adds --n-steps to train_rl_policy.py, default 256 (the baseline). Held back until
the evaluation chain ends, because evaluate_rl_policy.py imports that file and
live jobs' scripts don't get edited. Run from the Simulation directory."""
import os

PATH = "train_rl_policy.py"
with open(PATH, encoding="utf-8", newline="") as f:
    text = f.read()

if '"--n-steps"' in text:
    print("Already applied")
    raise SystemExit(0)

eol = "\r\n" if "\r\n" in text else "\n"
new_arg = eol.join([
    '    ap.add_argument("--n-envs", type=int, default=6)',
    '    ap.add_argument("--n-steps", type=int, default=256,',
    '                    help="Rollout length per env, fresh runs only (a --resume-from "',
    '                         "checkpoint keeps its own). 256 is the baseline; a held "',
    '                         "trial lasts up to 460 env steps.")',
])
replacements = [
    ('    ap.add_argument("--n-envs", type=int, default=6)', new_arg),
    ("n_steps=256, batch_size=256, learning_rate=3e-4,",
     "n_steps=args.n_steps, batch_size=256, learning_rate=3e-4,"),
]
for old, new in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"Expected exactly one match for: {old}")
    text = text.replace(old, new)

compile(text, PATH, "exec")
with open(PATH + ".tmp", "w", encoding="utf-8", newline="") as f:
    f.write(text)
os.replace(PATH + ".tmp", PATH)
print("Applied")
