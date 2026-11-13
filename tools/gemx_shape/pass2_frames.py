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
from PIL import ImageDraw
pick = [int(x) for x in "115,135,160,265,280".split(",")]
rows = []
for f in pick:
    i = frames.index(f)
    m = torch.as_tensor(np.asarray(obs["mask"][i], bool), device="cuda")
    lab = torch.as_tensor(np.load(capture / f"labels/{f:04d}.npy"), device="cuda")[:H, :W]
    rgb = np.asarray(Image.open(capture / prep["frames"][f]["rgb"]).convert("RGB"))[:H, :W]
    tiles = [rgb]
    for n, V in banks.items():
        if V is None:
            fwd.forward_device(ident, poses[i:i + 1]); pr = fwd.project_device(1, obs["camera"][i], Kfit[i], roots[i:i + 1], scale)[0]
        else:
            v = V[f]; k = KG[f]; fx, fy = k[0, 0] * sx, k[1, 1] * sy; cx, cy = (k[0, 2] + .5) * sx - .5, (k[1, 2] + .5) * sy - .5
            pr = torch.stack([fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy, v[:, 2]], 1)
        _, face, _ = fwd.zbuffer(pr.contiguous(), faces, H, W); sil = (face >= 0).reshape(H, W).cpu().numpy(); mm = m.cpu().numpy()
        img = (rgb * 0.45).astype(np.uint8); img[sil & mm] = (205, 205, 205); img[mm & ~sil] = (255, 60, 60); img[sil & ~mm] = (60, 120, 255)
        care = ~(((lab != 0) & ~m).cpu().numpy()); iou = (sil & mm).sum() / ((sil & care) | mm).sum()
        im = Image.fromarray(img); ImageDraw.Draw(im).rectangle((0, 0, W, 34), fill=(20, 20, 20)); ImageDraw.Draw(im).text((8, 8), f"{n} f{f} IoU {iou:.3f}", fill=(240, 240, 240))
        tiles.append(np.asarray(im))
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).save("/workspace/scratch/rgbmocap-test/shape/pass2-seed-wins.png"); print("ok")
