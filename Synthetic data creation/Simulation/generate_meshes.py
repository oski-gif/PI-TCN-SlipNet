"""
Generate unit meshes (.obj) for shapes MuJoCo has no native primitive for:
cone, square pyramid, triangular prism, hexagonal prism.

One function covers all of them: a cone is just an n-gon pyramid with a lot
of sides (n_sides=32); a pyramid is an n-gon base + apex; a prism is an
n-gon extruded from z=0 to z=1 with top+bottom caps.

All shapes are generated as UNIT solids: base vertices at radius 1 in the
xy-plane, height 1 along z. generate_trials.py scales each per-trial via
the MJCF <mesh scale="rx ry rz"> attribute -- rx=ry control the base size,
rz controls height -- so you don't need to regenerate mesh files to change
object dimensions.

Usage:
    python generate_meshes.py --out-dir assets
    (generates assets/cone.obj, assets/square_pyramid.obj,
     assets/triangular_prism.obj, assets/hexagonal_prism.obj)
"""
import argparse
import math
import os


def write_ngon_solid(out_path, n_sides, apex):
    """
    apex=True  -> pyramid/cone: n-gon base at z=0 (radius 1), single apex at
                  (0,0,1), no top cap.
    apex=False -> prism: n-gon base at z=0 AND top at z=1 (both radius 1),
                  connected by rectangular side faces, both ends capped.
    """
    verts = []
    faces = []

    for i in range(n_sides):
        theta = 2 * math.pi * i / n_sides
        verts.append((math.cos(theta), math.sin(theta), 0.0))
    base_start = 1  # 1-based OBJ indexing

    base_center_idx = len(verts) + 1
    verts.append((0.0, 0.0, 0.0))

    if apex:
        apex_idx = len(verts) + 1
        verts.append((0.0, 0.0, 1.0))

        # Bottom cap (fan from base center)
        for i in range(n_sides):
            v1 = base_start + i
            v2 = base_start + (i + 1) % n_sides
            faces.append((base_center_idx, v2, v1))

        # Side faces (fan from apex)
        for i in range(n_sides):
            v1 = base_start + i
            v2 = base_start + (i + 1) % n_sides
            faces.append((v1, v2, apex_idx))

    else:
        top_start = len(verts) + 1
        for i in range(n_sides):
            theta = 2 * math.pi * i / n_sides
            verts.append((math.cos(theta), math.sin(theta), 1.0))
        top_center_idx = len(verts) + 1
        verts.append((0.0, 0.0, 1.0))

        # Bottom cap (fan from base center)
        for i in range(n_sides):
            v1 = base_start + i
            v2 = base_start + (i + 1) % n_sides
            faces.append((base_center_idx, v2, v1))

        # Top cap (fan from top center)
        for i in range(n_sides):
            v1 = top_start + i
            v2 = top_start + (i + 1) % n_sides
            faces.append((top_center_idx, v1, v2))

        # Side walls (quad per edge, split into 2 triangles)
        for i in range(n_sides):
            b1 = base_start + i
            b2 = base_start + (i + 1) % n_sides
            t1 = top_start + i
            t2 = top_start + (i + 1) % n_sides
            faces.append((b1, b2, t2))
            faces.append((b1, t2, t1))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# unit {'pyramid' if apex else 'prism'}, n_sides={n_sides}, "
                f"base radius 1, height 1\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Wrote {out_path}  (n_sides={n_sides}, apex={apex}, "
          f"{len(verts)} verts, {len(faces)} faces)")


SHAPES = {
    "cone":              {"n_sides": 32, "apex": True},
    "square_pyramid":    {"n_sides": 4,  "apex": True},
    "triangular_prism":  {"n_sides": 3,  "apex": False},
    "hexagonal_prism":   {"n_sides": 6,  "apex": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    for name, params in SHAPES.items():
        out_path = os.path.join(args.out_dir, f"{name}.obj")
        write_ngon_solid(out_path, params["n_sides"], params["apex"])


if __name__ == "__main__":
    main()