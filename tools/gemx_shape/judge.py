"""One judge for seed fits and GEM-X meshes: seed's own fit masks (capture_observations, full-person-minus-hair incl.
material exclusion) on the score grid; pixels that are person-labelled but outside the fit mask are don't-care for
both. Seed: native forward + fitted fields + fit camera/K (as the bake). GEM-X: given incam vertices + GEM-X K,
per-axis mapped to the score grid. Usage: judge.py <seed fit dir> <gemx run> [name=verts.npy ...]"""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/workspace/seed/src")
from seed.capture.gpu_shape_fit import capture_observations, load_repose_pack
from seed.capture.saved_fit_camera import resolve_fit_intrinsics
from seed.body_model.soma import shared_cpu_body
from seed.body_model.soma_pack import pack_arrays
from seed.body_model.soma_native import SomaNative
from seed.body_model.soma_texture import _set_fitted_fields
fit = Path(sys.argv[1]); G = Path(sys.argv[2]); extra = [a.split("=", 1) for a in sys.argv[3:]]
project = Path("/workspace/seed/projects/bald-european-video")
spec = json.loads((fit / "soma-direct-spec.json").read_text()); params = np.load(fit / "soma-direct-params.npz")
capture = Path(spec["capture"]); capture = capture if capture.is_absolute() else project / capture
frames = [int(f) for f in params["frames"]]; prep = json.loads((capture / "frames.json").read_text())
H, W = np.asarray(Image.open(capture / prep["frames"][frames[0]]["rgb"])).shape[:2]
meta, arrays = load_repose_pack(capture / spec["pack_json"], capture / spec["source_glb"])
obs = capture_observations(capture, meta, arrays, capture / spec["source_glb"], frames, motion=spec["motion"], size=(H, W),
                           materials=spec["materials"], mask_policy="full-person-minus-hair")
Kfit = resolve_fit_intrinsics(spec, frames, obs["K"], size=(H, W))
pack = pack_arrays(shared_cpu_body(), correctives=True); fwd = SomaNative(pack); _set_fitted_fields(fwd, params)
faces = torch.as_tensor(np.asarray(pack["faces"], np.int32), device="cuda")
ident = torch.as_tensor(params["identity"][None].astype(np.float32), device="cuda")
poses = torch.as_tensor(params["poses"].astype(np.float32), device="cuda"); roots = torch.as_tensor(params["roots"].astype(np.float32), device="cuda")
scale = torch.tensor([float(params["scale"])], device="cuda")
RW, RH = prep["resizedSize"]; SW, SH = prep["sourceSize"]; sx, sy = RW / SW, RH / SH
KG = np.load(G / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3)
banks = {"seed-fit": None}; banks.update({n: torch.as_tensor(np.load(p), device="cuda") for n, p in extra})
res = {n: [] for n in banks}
for i, f in enumerate(frames):
    m = torch.as_tensor(np.asarray(obs["mask"][i], bool), device="cuda")
    lab = torch.as_tensor(np.load(capture / f"labels/{f:04d}.npy"), device="cuda")[:H, :W]
    care = torch.as_tensor(np.asarray(obs["valid"][i], bool), device="cuda")                # seed fit judge support, used for intersection and union
    for n, V in banks.items():
        if V is None:
            fwd.forward_device(ident, poses[i:i + 1]); pr = fwd.project_device(1, obs["camera"][i], Kfit[i], roots[i:i + 1], scale)[0]
        else:
            v = V[f]; k = KG[f]; fx, fy = k[0, 0] * sx, k[1, 1] * sy; cx, cy = k[0, 2] * sx, k[1, 2] * sy  # continuous coords
            pr = torch.stack([fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy, v[:, 2]], 1)
        _, face, _ = fwd.zbuffer(pr.contiguous(), faces, H, W); sil = (face >= 0).reshape(H, W)
        res[n].append(float((sil & m & care).sum() / (((sil | m) & care).sum()).clamp_min(1)))
shots = np.array([prep["frames"][f]["shot"] for f in frames])
out = {n: {"mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "min": float(np.min(v)),
           "perShot": {str(s): float(np.mean(np.array(v)[shots == s])) for s in sorted(set(shots))}} for n, v in res.items()}
out["grid"] = [W, H]; out["frames"] = len(frames); out["fit"] = str(fit)
print(json.dumps(out, indent=1)); json.dump({"summary": out, "perFrame": res}, open(Path("/workspace/scratch/rgbmocap-test/shape") / f"judge-{fit.parent.parent.name[:4]}.json", "w"))
