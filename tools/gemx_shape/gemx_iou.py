"""Per-frame IoU of GEM-X's exact incam mesh (bone scales + correctives) vs seed's capture Sapiens2 person masks.
Usage: gemx_iou.py <gemx run dir> <seed capture dir> <out.json> [--verts other.npy]
Projection: x = K V (OpenCV camera metres, GEM-X K on the 704x1254 source grid), scaled to the label grid."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/workspace/seed/src")
from seed.render import drtk_ops
from seed import sapiens_classes as sc

def project_clip(V, K, s, W, H, near=0.05, far=50.0):
    x = (K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2]) * s; y = (K[1, 1] * V[:, 1] / V[:, 2] + K[1, 2]) * s; z = V[:, 2]
    return torch.stack([((x + 0.5) / W * 2 - 1) * z, (1 - (y + 0.5) / H * 2) * z,
                        (far + near) / (far - near) * z - 2 * far * near / (far - near), z], 1)[None]

def raster(V, K, faces, s, W, H):
    r = drtk_ops.rasterize(project_clip(V, K, s, W, H), faces, (H, W))
    return r.valid[0].flip(0), r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--verts"); a = ap.parse_args()
    R, C = Path(a.run), Path(a.capture)
    fj = json.loads((C / "frames.json").read_text()); src_w, src_h = fj["sourceSize"]
    V = torch.as_tensor(np.load(a.verts or R / "soma-incam-vertices.npy"), device="cuda")
    K = torch.as_tensor(np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3), device="cuda", dtype=torch.float32)
    faces = torch.as_tensor(np.load(R / "soma-rig.npz")["faces"], device="cuda", dtype=torch.int32)
    T = V.shape[0]; assert T == len(fj["frames"]) == K.shape[0], (T, len(fj["frames"]), K.shape)
    # frame clock check: GEM-X frame t vs capture frame t (and +-1) on the label grid
    gem = sorted((R / "frames").glob("*.png"))
    lab0 = np.load(C / "labels/0000.npy"); H, W = lab0.shape; s = fj["resizedSize"][1] / src_h
    def small(p): return np.asarray(Image.open(p).convert("L").resize((W, H)), np.float32)
    clock = {}
    for t in (5, 120, 250):
        g = small(gem[t]); clock[t] = {d: float(np.abs(g - small(C / fj["frames"][t + d]["rgb"])).mean()) for d in (-1, 0, 1)}
    rows = []
    with torch.no_grad():
        for t in range(T):
            sil, _ = raster(V[t], K[t], faces, s, W, H); sil = sil.cpu().numpy()
            lab = np.load(C / f"labels/{t:04d}.npy")
            person = lab != sc.BACKGROUND; nohair = person & (lab != sc.HAIR)
            care = lab != sc.HAIR  # hair pixels are don't-care (mask policy full-person-minus-hair)
            inter = (sil & nohair).sum(); uni = ((sil & care) | nohair).sum()
            mask_only = nohair & ~sil; pred_only = sil & ~person
            parts = np.bincount(lab[mask_only], minlength=29)
            rows.append({"frame": t, "shot": fj["frames"][t]["shot"], "iou": float(inter / max(uni, 1)),
                         "predPx": int(sil.sum()), "maskPx": int(nohair.sum()), "maskOnlyPx": int(mask_only.sum()),
                         "predOnlyPx": int(pred_only.sum()), "maskOnlyByLabel": {int(i): int(c) for i, c in enumerate(parts) if c}})
    ious = np.array([r["iou"] for r in rows]); tot = {}
    for r in rows:
        for k, c in r["maskOnlyByLabel"].items(): tot[k] = tot.get(k, 0) + c
    names = {int(v): k for k, v in vars(sc).items() if k.isupper() and isinstance(v, int)}
    summary = {"frames": T, "grid": [W, H], "scale": s, "frameClockMeanAbsDiff": clock,
               "iouMean": float(ious.mean()), "iouMedian": float(np.median(ious)), "iouP10": float(np.percentile(ious, 10)),
               "iouMin": float(ious.min()), "worstFrames": [int(i) for i in np.argsort(ious)[:10]],
               "perShot": {str(sh): float(ious[[r["shot"] == sh for r in rows]].mean()) for sh in sorted({r["shot"] for r in rows})},
               "maskOnlyPxByLabel": {names.get(k, str(k)): v for k, v in sorted(tot.items(), key=lambda kv: -kv[1])},
               "predOnlyPxTotal": int(sum(r["predOnlyPx"] for r in rows)), "maskOnlyPxTotal": int(sum(r["maskOnlyPx"] for r in rows))}
    Path(a.out).write_text(json.dumps({"summary": summary, "frames": rows}, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
