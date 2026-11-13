"""Face-local shape (shared over frames) from source face landmarks: 3D vector Wendland patches on the bind face
(centres = landmark-anchor vertices), posed by GEM-X's Head rotation (camera frame), fitted by damped GN to the
TRAIN landmarks (feature id % 4 != 0) of all frames on the head-registered mesh; held-out landmarks reported.
Usage: face_shape.py <gemx run> <capture> <head-registered verts.npy> <out dir> [--radius 0.03 --lam 0.05]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/seed/src"); sys.path.insert(0, "/workspace/scratch/rgbmocap-test/shape")
from seed.body_model.soma_texture_face_evidence import load_face_warp_evidence
from fit_shape import geodesic, procrustes_R
dev = "cuda"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("capture"); ap.add_argument("verts"); ap.add_argument("out")
ap.add_argument("--radius", type=float, default=0.03); ap.add_argument("--lam", type=float, default=0.05); ap.add_argument("--cap", type=float, default=0.012)
a = ap.parse_args(); R, C, O = Path(a.run), Path(a.capture), Path(a.out); O.mkdir(parents=True, exist_ok=True)
rig = np.load(R / "soma-rig.npz"); names = [str(n) for n in rig["names"]]; F = torch.as_tensor(rig["faces"], device=dev).long()
B = torch.as_tensor(np.load(R / "soma-bind.npz")["bind_vertices"], device=dev)
V = torch.as_tensor(np.load(a.verts), device=dev); Vg = torch.as_tensor(np.load(R / "soma-vertices.npy"), device=dev)
world = torch.as_tensor(np.load(R / "soma-world.npy"), device=dev, dtype=torch.float32); rest = torch.as_tensor(rig["rest"], device=dev, dtype=torch.float32)
K = torch.as_tensor(np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3), device=dev, dtype=torch.float32)
ev = load_face_warp_evidence(C, rig["faces"].astype(np.int32), C / "face-texture-calibration/proposed-core-asset.json")["frames"]
T = V.shape[0]; hj = names.index("Head")
Rh = torch.stack([procrustes_R(Vg[t], V[t]) for t in range(T)]) @ (world[:, hj] @ torch.linalg.inv(rest[hj]))[:, :3, :3]
e0 = ev[0]; ids = torch.as_tensor(np.asarray(e0["vertex_ids"]), device=dev).long()
anchor = (ids if ids.ndim == 1 else ids.reshape(-1)).unique()
D = geodesic(anchor, B, F, a.radius); r = (D / a.radius).clamp(max=1)
Phi = torch.where(torch.isfinite(D), (1 - r) ** 4 * (4 * r + 1), torch.zeros_like(D)); Phi = Phi / Phi.sum(0, keepdim=True).clamp_min(1)
P = Phi.shape[0]; ov = ((Phi > 0).float() @ (Phi > 0).float().T) > 0; ov.fill_diagonal_(False); Lap = torch.diag(ov.sum(1).float()) - ov.float()
print(f"face basis {P} centres x 3, radius {a.radius}", flush=True)
def lm(v, e):
    i = torch.as_tensor(np.asarray(e["vertex_ids"]), device=dev).long(); p = v[i]
    if i.ndim == 2: p = (p * torch.as_tensor(e["barycentric"], device=dev, dtype=p.dtype)[..., None]).sum(1)
    return p
def lmPhi(e):
    i = torch.as_tensor(np.asarray(e["vertex_ids"]), device=dev).long(); ph = Phi[:, i].permute(1, 0, *range(2, i.ndim + 1)) if i.ndim == 1 else Phi[:, i]
    if i.ndim == 1: return Phi[:, i].T                                                    # (L,P)
    return (Phi[:, i] * torch.as_tensor(e["barycentric"], device=dev, dtype=Phi.dtype)[None]).sum(-1).T
def posed(c, t):                                                                          # c (P,3) bind-space vectors
    return V[t] + ((Phi.T @ c) @ Rh[t].T)
def stats(c, held):
    out = []
    for t in range(T):
        e = ev[t]; fid = np.asarray(e["feature_ids"]); w = (np.asarray(e["weights"]) > 0) & (((fid % 4) == 0) if held else ((fid % 4) != 0))
        if w.sum() < 2: continue
        k = K[t]; p = lm(posed(c, t), e)[torch.as_tensor(w, device=dev)]
        xy = torch.stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]], 1)
        d = (xy - torch.as_tensor(np.asarray(e["source_xy"])[w], device=dev, dtype=torch.float32)).norm(dim=1)
        out.append((float(d.median()), float(torch.quantile(d, 0.9))))
    o = np.array(out); return {"median": float(np.median(o[:, 0])), "p90med": float(np.median(o[:, 1])), "framesP90le8": int((o[:, 1] <= 8).sum()), "frames": len(o)}
c = torch.zeros(P, 3, device=dev)
print("start train", stats(c, False), "held", stats(c, True), flush=True)
for it in range(6):
    JTJ = torch.zeros(3 * P, 3 * P, device=dev, dtype=torch.float64); JTr = torch.zeros(3 * P, device=dev, dtype=torch.float64)
    for t in range(T):
        e = ev[t]; fid = np.asarray(e["feature_ids"]); w = (np.asarray(e["weights"]) > 0) & ((fid % 4) != 0)
        if w.sum() < 6: continue
        wt = torch.as_tensor(w, device=dev); k = K[t]; p = lm(posed(c, t), e)[wt]; ph = lmPhi(e)[wt]           # (L,3),(L,P)
        z = p[:, 2]; xy = torch.stack([k[0, 0] * p[:, 0] / z + k[0, 2], k[1, 1] * p[:, 1] / z + k[1, 2]], 1)
        res = (xy - torch.as_tensor(np.asarray(e["source_xy"])[w], device=dev, dtype=torch.float32)).reshape(-1)
        Jp = torch.zeros(len(p), 2, 3, device=dev); Jp[:, 0, 0] = k[0, 0] / z; Jp[:, 0, 2] = -k[0, 0] * p[:, 0] / z**2; Jp[:, 1, 1] = k[1, 1] / z; Jp[:, 1, 2] = -k[1, 1] * p[:, 1] / z**2
        JR = Jp @ Rh[t]                                                                    # (L,2,3) d pixel / d bind vector
        J = (JR[:, :, None, :] * ph[:, None, :, None]).reshape(2 * len(p), 3 * P)          # rows (L,2), cols (P,3)
        hub = torch.clamp(4.0 / res.abs().clamp_min(1e-6), max=1.0)
        JTJ += ((J * hub[:, None]).T @ J).double(); JTr += ((J * hub[:, None]).T @ res).double()
    Lk = torch.kron(Lap @ Lap + 1e-3 * torch.eye(P, device=dev), torch.eye(3, device=dev)).double()
    reg = a.lam * float(torch.diagonal(JTJ).mean())
    step = -torch.linalg.solve(JTJ + reg * Lk, JTr + reg * (Lk @ c.reshape(-1).double())).float().reshape(P, 3)
    c = (c + step); n = c.norm(dim=1, keepdim=True); c = c * torch.clamp(a.cap / n.clamp_min(1e-9), max=1.0)
    print(f"it {it}: |step| {step.norm(dim=1).max():.4f} train {stats(c, False)} held {stats(c, True)}", flush=True)
out = torch.stack([posed(c, t) for t in range(T)]).cpu().numpy().astype(np.float32); np.save(O / "face-shaped-vertices.npy", out)
disp = (Phi.T @ c).norm(dim=1)
rep = {"centres": P, "radius": a.radius, "lam": a.lam, "train": stats(c, False), "heldOut": stats(c, True),
       "dispMm": {"max": float(disp.max()) * 1000, "p95": float(torch.quantile(disp[disp > 0], 0.95)) * 1000}}
json.dump(rep, open(O / "face-shape.json", "w"), indent=1); np.savez(O / "face-basis.npz", anchor=anchor.cpu().numpy(), coef=c.cpu().numpy()); print(json.dumps(rep))
