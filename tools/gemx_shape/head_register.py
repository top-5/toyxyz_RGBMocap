"""Per-frame head/neck correction of a GEM-X incam mesh to seed's source face landmarks: rotation (3 dof) about
the first Neck joint, blended by the skin weight of the Neck subtree; damped GN with a rotation prior, Huber,
temporal smoothing within shots. All CUDA. Usage: head_register.py <gemx run> <capture> <verts.npy> <out dir>"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/workspace/seed/src")
from seed.body_model.soma_texture_face_evidence import load_face_warp_evidence
dev = "cuda"
R, C, VIN, O = map(Path, sys.argv[1:5]); O.mkdir(parents=True, exist_ok=True)
rig = np.load(R / "soma-rig.npz"); names = [str(n) for n in rig["names"]]; parents = rig["parents"]
Wt = torch.as_tensor(np.load(R / "soma-bind.npz")["skin_weights"], device=dev)
neck = next(i for i, n in enumerate(names) if "Neck" in n)
sub = {neck}
for j in range(len(names)):
    p = j
    while p >= 0:
        if p in sub: sub.add(j); break
        p = parents[p]
wh = Wt[:, sorted(sub)].sum(1)                                                        # blend weight per vertex
V = torch.as_tensor(np.load(VIN), device=dev); Vg = torch.as_tensor(np.load(R / "soma-vertices.npy"), device=dev)
world = np.load(R / "soma-world.npy"); K = np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3)
ev = load_face_warp_evidence(C, rig["faces"].astype(np.int32), C / "face-texture-calibration/proposed-core-asset.json")["frames"]
T = V.shape[0]; shots = [ev[t]["shot"] for t in range(T)]
def skew(w):
    z = torch.zeros((), device=dev); return torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])])
def rodrigues(w):
    th = w.norm().clamp_min(1e-9); k = skew(w / th); return torch.eye(3, device=dev) + torch.sin(th) * k + (1 - torch.cos(th)) * k @ k
def landmarks(v, e):
    ids = torch.as_tensor(e["vertex_ids"], device=dev, dtype=torch.long); pts = v[ids]
    if ids.ndim == 2: pts = (pts * torch.as_tensor(e["barycentric"], device=dev, dtype=pts.dtype)[..., None]).sum(1)
    return pts
omegas = torch.zeros(T, 6, device=dev); before, after, held_before, held_after = [], [], [], []; excluded = set()
for t in range(T):
    e = ev[t]; fid = torch.as_tensor(np.asarray(e["feature_ids"]), device=dev)
    hold = (fid % 4) == 0                                                                # outer held-out landmark subset (stable feature ids)
    w = (torch.as_tensor(e["weights"], device=dev) > 0) & ~hold
    if w.sum() < 6: before.append(np.nan); excluded.add(t); continue
    # neck pivot in the incam frame: rigid global->incam map from the meshes (exact, see register)
    a = Vg[t] - Vg[t].mean(0); b = V[t] - V[t].mean(0); U, S_, Vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(U @ Vt)); Rc = (U @ torch.diag(torch.tensor([1.0, 1.0, float(d)], device=dev)) @ Vt).T
    piv = Rc @ (torch.as_tensor(world[t, neck, :3, 3], device=dev, dtype=torch.float32) - Vg[t].mean(0)) + V[t].mean(0)
    k = torch.as_tensor(K[t], device=dev, dtype=torch.float32); src = torch.as_tensor(e["source_xy"], device=dev, dtype=torch.float32)[w]
    def proj(p): return torch.stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]], 1)
    def apply(x, v):
        Rm = rodrigues(x[:3]); return v + wh[:, None] * ((v - piv) @ Rm.T - (v - piv) + x[3:])
    om = omegas[t - 1].clone() if t and shots[t] == shots[t - 1] else torch.zeros(6, device=dev)
    r0 = (proj(landmarks(V[t], e)[w]) - src).norm(dim=1); before.append(float(r0.median()))
    for it in range(12):
        omv = om.clone().requires_grad_(True)
        res = (proj(landmarks(apply(omv, V[t]), e)[w]) - src).reshape(-1)
        J = torch.stack([torch.autograd.grad(res[i], omv, retain_graph=True)[0] for i in range(len(res))])
        rr = res.detach(); hub = torch.clamp(6.0 / rr.abs().clamp_min(1e-6), max=1.0)
        prior = torch.tensor([2e4, 2e4, 2e4, 2e6, 2e6, 2e6], device=dev)                   # px^2/rad^2 and px^2/m^2 (1 cm ~ 14 px^2)
        A = (J * hub[:, None]).T @ J + torch.diag(prior); g = (J * hub[:, None]).T @ rr + prior * om
        step = -torch.linalg.solve(A, g); om = (om + step).detach()
        if step.norm() < 1e-5: break
    omegas[t] = om
    if float(om[:3].norm()) > np.radians(35) or float(om[3:].norm()) > 0.04: excluded.add(t)                               # implausible: keep GEM-X's head
# temporal smoothing within shots (sigma 2 frames), then apply
sm = omegas.clone(); sh = np.array(shots)
for t in range(T):
    idx = np.where(sh == sh[t])[0]; ww = torch.as_tensor(np.exp(-0.5 * ((idx - t) / 2.0) ** 2), device=dev, dtype=torch.float32)
    keep = torch.as_tensor([i not in excluded for i in idx], device=dev, dtype=torch.float32); ww = ww * keep
    sm[t] = (ww[:, None] * omegas[idx]).sum(0) / ww.sum().clamp_min(1e-9)
for t in excluded: sm[t] = 0                                                           # excluded frames stay GEM-X's
out = []; head_repairs = 0; unrepaired = []; Fh = torch.as_tensor(rig['faces'], device=dev).long()
for t in range(T):
    a = Vg[t] - Vg[t].mean(0); b = V[t] - V[t].mean(0); U, S_, Vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(U @ Vt)); Rc = (U @ torch.diag(torch.tensor([1.0, 1.0, float(d)], device=dev)) @ Vt).T
    piv = Rc @ (torch.as_tensor(world[t, neck, :3, 3], device=dev, dtype=torch.float32) - Vg[t].mean(0)) + V[t].mean(0)
    Rm = rodrigues(sm[t][:3]); corr = (V[t] - piv) @ Rm.T - (V[t] - piv) + sm[t][3:]; wt_ = wh.clone()
    nb0 = torch.cross(V[t][Fh[:, 1]] - V[t][Fh[:, 0]], V[t][Fh[:, 2]] - V[t][Fh[:, 0]], dim=1); ok0 = nb0.norm(dim=1) > 1e-12
    for _ in range(8):                                                                  # flip repair: drop the correction around flipped triangles
        v = V[t] + wt_[:, None] * corr; nn = torch.cross(v[Fh[:, 1]] - v[Fh[:, 0]], v[Fh[:, 2]] - v[Fh[:, 0]], dim=1)
        bad = ((nn * nb0).sum(1) < 0) & ok0
        if not bool(bad.any()): break
        hit = torch.zeros(len(wt_), device=dev, dtype=torch.bool); hit[Fh[bad].reshape(-1)] = True
        for _r in range(2): hit[Fh[hit[Fh].any(1)].reshape(-1)] = True
        wt_ = torch.where(hit, torch.zeros_like(wt_), wt_); head_repairs += int(hit.sum())
    else:
        v = V[t] + wt_[:, None] * corr; nn = torch.cross(v[Fh[:, 1]] - v[Fh[:, 0]], v[Fh[:, 2]] - v[Fh[:, 0]], dim=1)
        if bool((((nn * nb0).sum(1) < 0) & ok0).any()):                                   # unrepairable: keep GEM-X's head this frame
            v = V[t].clone(); unrepaired.append(t)
    out.append(v)
    e = ev[t]; fid = torch.as_tensor(np.asarray(e["feature_ids"]), device=dev); allw = torch.as_tensor(e["weights"], device=dev) > 0
    hw = allw & ((fid % 4) == 0); w = allw & ~((fid % 4) == 0)
    k_ = torch.as_tensor(K[t], device=dev, dtype=torch.float32); src_all = torch.as_tensor(e["source_xy"], device=dev, dtype=torch.float32)
    def err(vv, msk):
        if msk.sum() < 2: return np.nan
        p = landmarks(vv, e)[msk]; return float((torch.stack([k_[0, 0] * p[:, 0] / p[:, 2] + k_[0, 2], k_[1, 1] * p[:, 1] / p[:, 2] + k_[1, 2]], 1) - src_all[msk]).norm(dim=1).median())
    held_before.append(err(V[t], hw)); held_after.append(err(v, hw))
    if w.sum() >= 6:
        k = torch.as_tensor(K[t], device=dev, dtype=torch.float32); src = torch.as_tensor(e["source_xy"], device=dev, dtype=torch.float32)[w]
        p = landmarks(v, e)[w]; after.append(float((torch.stack([k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]], 1) - src).norm(dim=1).median()))
    else: after.append(np.nan)
np.save(O / "head-registered-vertices.npy", torch.stack(out).cpu().numpy().astype(np.float32))
B, A_ = np.array(before), np.array(after); deg = np.degrees(sm[:, :3].norm(dim=1).cpu().numpy())
rep = {"neckJoint": names[neck], "landmarkMedianPx": {"before": float(np.nanmedian(B)), "after": float(np.nanmedian(A_))},
       "perShotBeforeAfter": {str(s): [float(np.nanmedian(B[sh == s])), float(np.nanmedian(A_[sh == s]))] for s in sorted(set(shots))},
       "excludedFrames": len(excluded), "flipRepairVertexFrames": head_repairs, "unrepairedFramesKeptGemx": unrepaired, "heldOutLandmarkMedianPx": {"before": float(np.nanmedian(held_before)), "after": float(np.nanmedian(held_after)), "framesUnder4px": int(np.nansum(np.array(held_after) <= 4))}, "translationCm": {"median": float(np.median(sm[:, 3:].norm(dim=1).cpu().numpy()) * 100), "max": float(sm[:, 3:].norm(dim=1).max()) * 100}, "rotationDeg": {"median": float(np.median(deg)), "p95": float(np.percentile(deg, 95)), "max": float(deg.max())}}
json.dump(rep, open(O / "head-registration.json", "w"), indent=1); print(json.dumps(rep, indent=1))
