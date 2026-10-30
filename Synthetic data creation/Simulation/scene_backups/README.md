# Scene file backups -- NOT the live copies

`scene_slip.xml` and `scene_flat.xml` here are safety backups of the files
actually used at `third_party/orcahand_description/v2/scene_slip.xml` /
`scene_flat.xml`. Those live files were never tracked in git anywhere --
not even locally -- because they sit inside the orcahand_description
nested repo, which has its own `.git` and (as of 2026-10-30) is registered
as a proper submodule in the outer repo. A submodule boundary means the
outer repo cannot track a file path nested inside it, so these two files
had no backup at all until now.

**Do not point any script at these backup copies.** Their mesh references
(`models/assets/left/...`, relative paths with no `meshdir` override) only
resolve correctly from inside `third_party/orcahand_description/v2/`,
which is where `generate_trials.py`/`closed_loop_sim.py` still expect to
load them from via `--mjcf`. These backups exist purely so the files
aren't at risk of being lost to a local-machine failure -- actually
relocating the live pipeline to use a tracked path would require adding
an explicit `meshdir` compiler attribute and verifying mesh resolution
still works, which hasn't been done here.
