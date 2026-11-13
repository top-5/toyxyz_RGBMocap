"""Re-export a GEM-X result with a different SOMA body shape (identity coefficients + bone scales), same motion.

Usage: gemx_reshape_export.py <gemx result dir> <shape.npz with identity_coeffs (1,128), scale_params (1,57)> <out dir>
       [--keep-floor] [--transl-world <npz with transl_world (T,3)>]

--keep-floor: keep the SOURCE result's floor offset (soma-result.json vertical_offset_m) instead of recomputing it for
the new body, so the body stays where it was fitted in the source camera. --transl-world: per-frame world-space root
offsets added to GEM-X's global translation (a camera-frame refit residual rotated into world by the caller).

Reads <gemx>/soma-parameters.npz (GEM-X's global SOMA parameters), replaces identity_coeffs / scale_params on every
frame, and writes the files tools/gemx_to_glb.py consumes into <out>: soma-rig.npz, soma-world.npy, soma-vertices.npy
(via mocap.gemx_pipeline.evaluate, the same code gemx_headless uses), soma-bind.npz, soma-parameters.npz and
project.json (copied). Rotations, translations and the clip's floor handling are GEM-X's; nothing else changes.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("shape", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--keep-floor", action="store_true")
    ap.add_argument("--transl-world", type=Path)
    a = ap.parse_args()
    src, shape_file, out = a.src, a.shape, a.out
    out.mkdir(parents=True, exist_ok=False)
    import gemx_headless as gh
    from mocap.gemx_pipeline import evaluate

    z = np.load(src / "soma-parameters.npz")
    params = {k: torch.as_tensor(z[k]) for k in z.files}
    n = params["global_orient"].shape[0]
    shp = np.load(shape_file)
    for key, width in (("identity_coeffs", 128), ("scale_params", 57)):
        v = torch.as_tensor(np.asarray(shp[key], np.float32).reshape(1, width))
        if not torch.isfinite(v).all():
            raise ValueError(f"non-finite {key}")
        params[key] = v.expand(n, width).contiguous()
    if a.transl_world is not None:
        tw = torch.as_tensor(np.load(a.transl_world)["transl_world"], dtype=params["transl"].dtype)
        if tw.shape != params["transl"].shape or not torch.isfinite(tw).all():
            raise ValueError(f"transl_world {tuple(tw.shape)} != {tuple(params['transl'].shape)}")
        params["transl"] = params["transl"] + tw
    np.savez(out / "soma-parameters.npz", **{k: v.numpy() for k, v in params.items()})
    with gh.seed_adapters(None):
        from gem.motion.geometry import MotionGeometry

        engine = MotionGeometry(str(gh.MODELS), device="cuda", chunk_size=16)
        offset = evaluate(engine, params, out, torch, np)
    if a.keep_floor:
        src_offset = float(json.loads((src / "soma-result.json").read_text())["vertical_offset_m"])
        dy = src_offset - offset
        world = np.load(out / "soma-world.npy", mmap_mode="r+")
        world[:, :, 1, 3] += dy
        world.flush()
        verts = np.load(out / "soma-vertices.npy", mmap_mode="r+")
        verts[:, :, 1] += dy
        verts.flush()
        offset = src_offset
    with gh.seed_adapters(None):
        neutral = {k: (torch.zeros_like(v[:1]) if k in ("global_orient", "body_pose", "transl") else v[:1]).to(
            device="cuda", dtype=torch.float32) for k, v in params.items()}
        with torch.no_grad():
            bind = engine.body_model(**neutral, pose2rot=True, apply_correctives=True,
                                     repose_to_bind_pose=True)["vertices"][0].cpu().numpy()
        bind[:, 1] -= bind[:, 1].min()
        weights = engine.body_model.runtime.public_skinning_weights()[:, 1:].cpu().numpy()
    if bind.shape != (18056, 3) or not np.isfinite(bind).all() or not np.allclose(weights.sum(1), 1, atol=1e-5):
        raise ValueError("invalid bind geometry or weights")
    np.savez(out / "soma-bind.npz", bind_vertices=bind, skin_weights=weights)
    shutil.copy2(src / "project.json", out / "project.json")
    (out / "reshape.json").write_text(json.dumps(dict(source=str(src), shape=str(shape_file), frames=n,
                                                      vertical_offset_m=offset), indent=2))
    print(json.dumps(dict(out=str(out), frames=n, vertical_offset_m=offset)))


if __name__ == "__main__":
    main()
