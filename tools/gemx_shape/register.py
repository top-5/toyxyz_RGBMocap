"""Per-frame rigid registration of GEM-X's exact incam mesh to the capture person mask: camera-space translation
(3 dof; depth covers image scale) by contour ICP (symmetric 2D chamfer between the rendered mesh outline and the mask
outline, hair-adjacent outline excluded), all CUDA. Writes registered vertices + per-frame/per-part contour stats.
Usage: register.py <gemx run> <capture> <out dir> [--iters 25] [--smooth 2]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, "/workspace/scratch/rgbmocap-test/shape"); sys.path.insert(0, "/workspace/seed/src")
from gemx_iou import raster
from seed import sapiens_classes as sc
PARTS = {"head": ["Head", "Neck", "Jaw", "Eye"], "torso": ["Spine", "Chest", "Hips", "Shoulder", "Clavicle", "Root"],
         "arm": ["Arm", "ForeArm"], "hand": ["Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger"],
         "leg": ["UpLeg", "Leg", "Shin", "Thigh"], "foot": ["Foot", "Toe", "Ankle"]}

def part_of(name):
    side = "L" if name.startswith("Left") else "R" if name.startswith("Right") else ""
    for part, keys in (("hand", PARTS["hand"]), ("foot", PARTS["foot"]), ("head", PARTS["head"]), ("arm", PARTS["arm"]),
                       ("leg", PARTS["leg"]), ("torso", PARTS["torso"])):
        if any(k.lower() in name.lower() for k in keys): return (side + part) if part not in ("head", "torso") else part
    return "torso"

def outline(binary):
    b = binary.float()[None, None]; er = -Fn.max_pool2d(-b, 3, 1, 1)
    return (b[0, 0] > 0) & (er[0, 0] == 0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=25); ap.add_argument("--smooth", type=float, default=2.0); a = ap.parse_args()
    R, C, O = Path(a.run), Path(a.capture), Path(a.out); O.mkdir(parents=True, exist_ok=True)
    fj = json.loads((C / "frames.json").read_text()); s = fj["resizedSize"][1] / fj["sourceSize"][1]
    V = torch.as_tensor(np.load(R / "soma-incam-vertices.npy"), device="cuda")
    K = torch.as_tensor(np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3), device="cuda", dtype=torch.float32)
    rig = np.load(R / "soma-rig.npz"); F = torch.as_tensor(rig["faces"], device="cuda", dtype=torch.int32)
    W8 = np.load(R / "soma-bind.npz")["skin_weights"]; names = [str(n) for n in rig["names"]]
    vpart_names = sorted({part_of(n) for n in names}); jpart = np.array([vpart_names.index(part_of(n)) for n in names])
    vpart = torch.as_tensor(jpart[W8.argmax(1)], device="cuda")
    T = V.shape[0]; trans = torch.zeros(T, 3, device="cuda"); rows = []
    for t in range(T):
        lab = torch.as_tensor(np.load(C / f"labels/{t:04d}.npy"), device="cuda"); H, W = lab.shape
        mask = (lab != sc.BACKGROUND) & (lab != sc.HAIR); hair = lab == sc.HAIR
        near_hair = Fn.max_pool2d(hair.float()[None, None], 7, 1, 3)[0, 0] > 0
        edge = (H - 1, W - 1)
        mo = outline(mask) & ~near_hair; mo[0, :] = mo[-1, :] = False; mo[:, 0] = mo[:, -1] = False  # frame border is not outline
        q = torch.nonzero(mo).flip(1).float()                              # (M,2) x,y mask outline
        tr = (trans[t - 1].clone() if t and fj["frames"][t]["shot"] == fj["frames"][t - 1]["shot"] else torch.zeros(3, device="cuda"))
        kk = K[t]
        for it in range(a.iters):
            sil, r = raster(V[t] + tr, kk, F, s, W, H)
            po = outline(sil) & ~near_hair; po[0, :] = po[-1, :] = False; po[:, 0] = po[:, -1] = False
            pix = torch.nonzero(po)                                        # (N,2) y,x (rows flipped back below)
            tri = r.index_img[0].flip(0)[pix[:, 0], pix[:, 1]].long() if hasattr(r, "index_img") else None
            # outline vertices: project all, take vertices of outline triangles (fallback: nearest projected vertex)
            v = V[t] + tr; x = (kk[0, 0] * v[:, 0] / v[:, 2] + kk[0, 2]) * s; y = (kk[1, 1] * v[:, 1] / v[:, 2] + kk[1, 2]) * s
            P = torch.stack([x, y], 1); pxy = pix.flip(1).float()
            vid = torch.cdist(pxy, P).argmin(1).unique()                   # outline vertices
            if len(vid) < 20 or len(q) < 20: break
            tvar = tr.clone().requires_grad_(True)
            vv = V[t][vid] + tvar
            Pv = torch.stack([(kk[0, 0] * vv[:, 0] / vv[:, 2] + kk[0, 2]) * s, (kk[1, 1] * vv[:, 1] / vv[:, 2] + kk[1, 2]) * s], 1)
            d1 = torch.cdist(Pv, q); n1 = d1.argmin(1); d2 = torch.cdist(q, Pv.detach()); n2 = d2.argmin(1)
            # symmetric chamfer, robust (Huber 6 px), GN on 3 params via autograd normal equations
            r1 = (Pv - q[n1]); r2 = (q - Pv[n2])
            res = torch.cat([r1.reshape(-1), -r2.reshape(-1)])
            w = torch.cat([torch.clamp(6 / r1.norm(dim=1).clamp_min(1e-6), max=1).repeat_interleave(2),
                           torch.clamp(6 / r2.norm(dim=1).clamp_min(1e-6), max=1).repeat_interleave(2)])
            # Jacobian of projected points wrt translation is analytic: d(x)/d(t) = s*fx*[1/z, 0, -X/z^2]
            z = vv[:, 2].detach(); X = vv[:, 0].detach(); Y = vv[:, 1].detach(); fx, fy = kk[0, 0] * s, kk[1, 1] * s
            jx = torch.stack([fx / z, torch.zeros_like(z), -fx * X / z**2], 1); jy = torch.stack([torch.zeros_like(z), fy / z, -fy * Y / z**2], 1)
            J1 = torch.stack([jx, jy], 1).reshape(-1, 3)
            J2 = torch.stack([jx[n2], jy[n2]], 1).reshape(-1, 3)
            Jall = torch.cat([J1, J2]); rr = res.detach()
            A = (Jall * w[:, None]).T @ Jall + 1e-2 * torch.eye(3, device="cuda") * (Jall**2).sum(0).mean()
            step = -torch.linalg.solve(A, (Jall * w[:, None]).T @ rr)
            tr = (tr + step).detach()
            if step.norm() < 1e-4: break
        sil, _ = raster(V[t] + tr, kk, F, s, W, H)
        care = ~hair; inter = (sil & mask).sum(); uni = ((sil & care) | mask).sum(); iou = float(inter / uni.clamp_min(1))
        # per-part contour distance of the registered mesh outline (mesh part -> median px to mask outline)
        po = outline(sil) & ~near_hair; po[0, :] = po[-1, :] = False; po[:, 0] = po[:, -1] = False
        v = V[t] + tr; P = torch.stack([(kk[0, 0] * v[:, 0] / v[:, 2] + kk[0, 2]) * s, (kk[1, 1] * v[:, 1] / v[:, 2] + kk[1, 2]) * s], 1)
        pxy = torch.nonzero(po).flip(1).float(); vid = torch.cdist(pxy, P).argmin(1)
        dist = torch.cdist(pxy, q).min(1).values if len(q) else torch.full((len(pxy),), 1e9, device="cuda")
        parts = {}
        for pi, pn in enumerate(vpart_names):
            m = vpart[vid] == pi
            if m.sum() >= 10: parts[pn] = {"medianPx": float(dist[m].median()), "p90Px": float(torch.quantile(dist[m], 0.9)), "n": int(m.sum())}
        trans[t] = tr.detach(); rows.append({"frame": t, "iou": iou, "translation": tr.tolist(), "parts": parts})
        if t % 40 == 0: print(t, round(iou, 4), [round(x, 4) for x in tr.tolist()], flush=True)
    # temporal smoothing of translation within shots (Gaussian, sigma frames), then final IoU
    shots = np.array([f["shot"] for f in fj["frames"]]); tn = trans.cpu().numpy(); sm = tn.copy()
    if a.smooth > 0:
        from math import exp
        for t in range(T):
            idx = np.where(shots == shots[t])[0]; w = np.exp(-0.5 * ((idx - t) / a.smooth) ** 2); sm[t] = (w[:, None] * tn[idx]).sum(0) / w.sum()
    Vr = (V + torch.as_tensor(sm, device="cuda")[:, None]).cpu().numpy().astype(np.float32)
    np.save(O / "registered-vertices.npy", Vr)
    Vs = torch.as_tensor(Vr, device="cuda")
    for t in range(T):                                                                 # gate stats on the FINAL (smoothed) output
        lab = torch.as_tensor(np.load(C / f"labels/{t:04d}.npy"), device="cuda"); H, W = lab.shape
        mask = (lab != sc.BACKGROUND) & (lab != sc.HAIR); hair = lab == sc.HAIR
        near_hair = Fn.max_pool2d(hair.float()[None, None], 7, 1, 3)[0, 0] > 0
        mo = outline(mask) & ~near_hair; mo[[0, -1], :] = False; mo[:, [0, -1]] = False; q = torch.nonzero(mo).flip(1).float()
        kk = K[t]; sil, _ = raster(Vs[t], kk, F, s, W, H); care = ~hair
        rows[t]["iou"] = float((sil & mask).sum() / ((sil & care) | mask).sum().clamp_min(1))
        po = outline(sil) & ~near_hair; po[[0, -1], :] = False; po[:, [0, -1]] = False
        v = Vs[t]; P = torch.stack([(kk[0, 0] * v[:, 0] / v[:, 2] + kk[0, 2]) * s, (kk[1, 1] * v[:, 1] / v[:, 2] + kk[1, 2]) * s], 1)
        pxy = torch.nonzero(po).flip(1).float(); vid = torch.cdist(pxy, P).argmin(1)
        dist = torch.cdist(pxy, q).min(1).values if len(q) else torch.full((len(pxy),), 1e9, device="cuda")
        parts = {}
        for pi, pn in enumerate(vpart_names):
            m = vpart[vid] == pi
            if m.sum() >= 10: parts[pn] = {"medianPx": float(dist[m].median()), "p90Px": float(torch.quantile(dist[m], 0.9)), "n": int(m.sum())}
        rows[t]["parts"] = parts
    json.dump({"translationSmoothSigmaFrames": a.smooth, "partNames": vpart_names, "frames": rows,
               "translationSmoothed": sm.tolist()}, open(O / "registration.json", "w"), indent=1)
    print("mean iou (final smoothed)", np.mean([r["iou"] for r in rows]))

if __name__ == "__main__":
    main()
