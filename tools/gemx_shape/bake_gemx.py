"""Deliverable A: seed's texture bake (1280's options) on GEM-X's exact incam mesh + K via gather_texture's
geometry_provider. Usage: bake_gemx.py <tag> [--verts V.npy] [--low-confidence iou.json --min-iou 0.80]"""
import argparse, json, re, shutil, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/seed/src")
from seed.body_model.soma_texture import gather_texture
from seed.body_model.soma_pack import pack_arrays
from seed.body_model.soma import shared_cpu_body
from seed.pipeline.plan_video_bake import texture_options
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--verts"); ap.add_argument("--low-confidence")
ap.add_argument("--min-iou", type=float, default=0.80); ap.add_argument("--arm-ownership", action="store_true"); ap.add_argument("--face", action="store_true"); a = ap.parse_args()
G = Path("/workspace/scratch/rgbmocap-test/headless-full-07")
project = Path("/workspace/seed/projects/bald-european-video"); ref = next((project / "runs").glob("1280_*"))
faces_seed = np.asarray(pack_arrays(shared_cpu_body(), correctives=True)["faces"], np.int32)
faces_gem = np.load(G / "soma-rig.npz")["faces"].astype(np.int32)
assert faces_seed.shape == faces_gem.shape and (faces_seed == faces_gem).all(), "SOMA topology differs"
V = torch.as_tensor(np.load(a.verts or G / "soma-incam-vertices.npy"), device="cuda", dtype=torch.float32)
KG = np.load(G / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3).astype(np.float32); SRC_W, SRC_H = 704, 1254
calls = {}
fj = json.loads((ref / "capture" / "frames.json").read_text()); RW, RH = fj["resizedSize"]; assert fj["padLeftTop"] == [0, 0]
def axes(H, W):
    if (H, W) == (SRC_H, SRC_W): return 1.0, 1.0
    if H == RH and W in (RW, RW + 1): return RW / SRC_W, RH / SRC_H        # aspect-preserving resize, pad right
    raise ValueError(f"unknown bake grid {W}x{H}")
class Provider:
    """GEM-X exact incam vertices + GEM-X K, both mapped to the requested grid in continuous pixel coordinates."""
    def intrinsics(self, k, H, W):
        sx, sy = axes(H, W); kg = KG[k]
        return (kg[0, 0] * sx, kg[1, 1] * sy, kg[0, 2] * sx, kg[1, 2] * sy)  # continuous coords (native zbuffer: pixel i centre i+.5; GEM-X cx = W/2), impulse-checked
    def __call__(self, k, K, H, W):
        fx, fy, cx, cy = self.intrinsics(k, H, W)
        if not np.allclose(np.asarray(K, np.float64).reshape(4), (fx, fy, cx, cy), rtol=1e-5, atol=1e-3):
            raise ValueError(f"frame {k}: bake K {K} is not the provider's {(fx, fy, cx, cy)}")
        v = V[k]; calls[(H, W)] = calls.get((H, W), 0) + 1
        assert torch.isfinite(v).all() and (v[:, 2] > 0).all()
        return torch.stack([fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy, v[:, 2]], 1)
provider = Provider()
assert re.fullmatch(r"[A-Za-z0-9._-]+", a.tag), "tag must be a plain local name"
fit = ref / "fit" / f"bake-gemx-{a.tag}"
if fit.exists(): raise SystemExit(f"{fit} exists; pick a new tag")
shutil.copytree(ref / "fit" / "body", fit, ignore=shutil.ignore_patterns("texture-bake", "texture-bake-*", "hair*"))
o = texture_options(json.loads((ref / "preparation" / "video-fit.json").read_text()))
o["arm_ownership"] = a.arm_ownership
if not a.face:
    o["face_landmark_warp"] = False
    for key in ("face_reference_frames", "face_landmark_asset", "face_max_applied_median", "face_max_applied_p90"): o.pop(key, None)
if a.low_confidence:
    rows = json.loads(Path(a.low_confidence).read_text())["frames"]
    o["low_confidence_frames"] = tuple(r["frame"] for r in rows if r["iou"] < a.min_iou)
    print("low-confidence frames", len(o["low_confidence_frames"]))
t = time.time()
gather_texture(project, str((ref / "capture").relative_to(project)), fit.name, run_directory=fit, geometry_provider=provider, **o)
print("T gemx %s %.1f s grids %s" % (a.tag, time.time() - t, calls), flush=True)
