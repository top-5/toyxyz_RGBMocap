"""Shape fit on GEM-X's fixed motion: bind-space normal displacement d = n_v * sum_k phi_k(v) c_k (part-gated
geodesic Wendland RBFs), posed with GEM-X's own joint rotations (A_vt = R_cam,t sum_j w_vj R_jt), on top of the
exact registered incam mesh. Contour ICP + damped Gauss-Newton over gated (frame, part) training outlines; step
acceptance on validation frames (IoU); test frames untouched. All CUDA.
Usage: fit_shape.py <gemx run> <capture> <registration dir> <out dir> [--centres 160 --radius 0.10 --levels 1]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
sys.path.insert(0, "/workspace/scratch/rgbmocap-test/shape"); sys.path.insert(0, "/workspace/seed/src")
from gemx_iou import raster
from register import outline, part_of
from seed import sapiens_classes as sc
dev = "cuda"

def procrustes_R(a, b):
    a = a - a.mean(0); b = b - b.mean(0); U, S, Vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.det(U @ Vt)); D = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=a.device))
    return (U @ D @ Vt).T

def vertex_normals(v, f):
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1); n = torch.zeros_like(v)
    for i in range(3): n.index_add_(0, f[:, i], fn)
    return Fn.normalize(n, dim=1)

def geodesic(src, v, f, limit, iters=200):
    """Approximate geodesic distance (S, V) from source vertices along mesh edges (Bellman-Ford on CUDA)."""
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); e = torch.cat([e, e.flip(1)]).unique(dim=0)
    L = (v[e[:, 0]] - v[e[:, 1]]).norm(dim=1)
    D = torch.full((len(src), len(v)), float("inf"), device=dev); D[torch.arange(len(src)), src] = 0
    for _ in range(iters):
        cand = D[:, e[:, 0]] + L[None]
        nd = D.clone(); nd.scatter_reduce_(1, e[:, 1][None].expand(len(src), -1), cand, "amin")
        if torch.equal(nd, D): break
        D = nd.clamp_max(limit * 1.01)
    D[D > limit] = float("inf"); return D

def fps(v, n, mask):
    idx = torch.nonzero(mask)[:, 0]; pts = v[idx]; sel = [int(torch.argmax(pts[:, 1]))]; d = (pts - pts[sel[0]]).norm(dim=1)
    for _ in range(n - 1):
        i = int(torch.argmax(d)); sel.append(i); d = torch.minimum(d, (pts - pts[i]).norm(dim=1))
    return idx[torch.as_tensor(sel, device=dev)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("capture"); ap.add_argument("reg"); ap.add_argument("out")
    ap.add_argument("--centres", type=int, default=160); ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--outer", type=int, default=4); ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--max-disp", type=float, default=0.04); ap.add_argument("--ridge", type=float, default=0.1); ap.add_argument("--gate"); ap.add_argument("--ramp", type=float, default=4.0); ap.add_argument("--exclude", default="head,hand,foot"); a = ap.parse_args()
    R, C, G, O = Path(a.run), Path(a.capture), Path(a.reg), Path(a.out); O.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fj = json.loads((C / "frames.json").read_text()); s = fj["resizedSize"][1] / fj["sourceSize"][1]
    rig = np.load(R / "soma-rig.npz"); bind = np.load(R / "soma-bind.npz")
    F = torch.as_tensor(rig["faces"], device=dev, dtype=torch.int32); Fl = F.long()
    B = torch.as_tensor(bind["bind_vertices"], device=dev); Wt = torch.as_tensor(bind["skin_weights"], device=dev)
    names = [str(n) for n in rig["names"]]
    rest = torch.as_tensor(rig["rest"], device=dev, dtype=torch.float32); world = torch.as_tensor(np.load(R / "soma-world.npy"), device=dev, dtype=torch.float32)
    Vg = torch.as_tensor(np.load(R / "soma-vertices.npy"), device=dev); V0 = torch.as_tensor(np.load(G / "registered-vertices.npy"), device=dev)
    K = torch.as_tensor(np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3), device=dev, dtype=torch.float32)
    T = V0.shape[0]; reg = json.loads((G / "registration.json").read_text())
    # per-frame joint rotation relative to bind, in the camera frame
    Rj = (world @ torch.linalg.inv(rest)[None])[..., :3, :3]                                  # (T,J,3,3) global
    Rc = torch.stack([procrustes_R(Vg[t], V0[t]) for t in range(T)])                         # (T,3,3) global->incam
    Rj = Rc[:, None] @ Rj
    # basis: part-gated geodesic Wendland RBFs on the bind mesh, scalar along bind normals
    pnames = sorted({part_of(n) for n in names}); jp = torch.as_tensor([pnames.index(part_of(n)) for n in names], device=dev)
    vpart = jp[Wt.argmax(1)]; nb = vertex_normals(B, Fl)
    excl = [x.strip() for x in a.exclude.split(",") if x.strip()]
    body = ~torch.isin(vpart, torch.as_tensor([pnames.index(p) for p in pnames if any(p == x or p.endswith(x) for x in excl)], device=dev))  # thin parts: no shape patches
    cen = fps(B, a.centres, body)
    D = geodesic(cen, B, Fl, a.radius)                                                         # (P,V)
    r = (D / a.radius).clamp(max=1); Phi = torch.where(torch.isfinite(D), (1 - r) ** 4 * (4 * r + 1), torch.zeros_like(D))
    Phi[:, ~body] = 0; P = Phi.shape[0]
    Phi = Phi / Phi.sum(0, keepdim=True).clamp_min(1)  # partition-of-unity-ish where centres overlap
    print(f"basis {P} centres, radius {a.radius} m, mean support {(Phi > 0).sum(1).float().mean():.0f} verts, setup {time.time()-t0:.1f}s", flush=True)
    # neighbour graph between centres for smoothness: centres whose supports overlap
    ov = ((Phi > 0).float() @ (Phi > 0).float().T) > 0; ov.fill_diagonal_(False)
    Lap = torch.diag(ov.sum(1).float()) - ov.float()
    # gate per (frame, part)
    typical = {p: np.median([f["parts"][p]["medianPx"] for f in reg["frames"] if p in f["parts"]]) for p in pnames if any(p in f["parts"] for f in reg["frames"])}
    kpg = json.loads(Path(a.gate).read_text())["frames"] if a.gate else None
    KPMAP = {"Larm": ("Larm",), "Rarm": ("Rarm",), "Lhand": ("Larm",), "Rhand": ("Rarm",), "Lleg": ("Lleg",), "Rleg": ("Rleg",),
             "Lfoot": ("Lleg",), "Rfoot": ("Rleg",), "torso": ("torso",)}
    def ok(t, p):
        if kpg is not None:                                                                # pose gate from source keypoints
            if p == "head" or p not in KPMAP: return False
            parts = kpg[t]["parts"]
            return all(parts.get(q, {}).get("ok", False) for q in KPMAP[p])
        st = reg["frames"][t]["parts"].get(p)
        return p != "head" and st is not None and st["medianPx"] <= max(12.0, 1.5 * typical[p]) and st["p90Px"] <= 30.0
    split = {"train": [t for t in range(T) if t % 3 == 0], "val": [t for t in range(T) if t % 3 == 1], "test": [t for t in range(T) if t % 3 == 2]}
    gate = {t: [pnames.index(p) for p in pnames if ok(t, p)] for t in split["train"]}
    print("gated train (frame,part) pairs", sum(len(v) for v in gate.values()), "of", len(split["train"]) * (len(pnames) - 1), flush=True)
    labs = {}
    def lab(t):
        if t not in labs: labs[t] = torch.as_tensor(np.load(C / f"labels/{t:04d}.npy"), device=dev)
        return labs[t]
    def posed(c, frames):
        d = nb * (Phi.T @ c)[:, None]                                                          # (V,3) bind delta
        A = torch.einsum("vj,tjab->tvab", Wt, Rj[frames]) if False else None
        out = []
        for t in frames:
            Av = torch.einsum("vj,jab->vab", Wt, Rj[t]); out.append(V0[t] + (Av @ d[..., None])[..., 0])
        return out
    def iou_frames(c, frames):
        vs = posed(c, frames); res = []
        for t, v in zip(frames, vs):
            L = lab(t); H, W = L.shape; sil, _ = raster(v, K[t], F, s, W, H)
            m = (L != 0) & (L != sc.HAIR); care = L != sc.HAIR
            res.append(float((sil & m).sum() / ((sil & care) | m).sum().clamp_min(1)))
        return float(np.mean(res))
    c = torch.zeros(P, device=dev)
    hist = {"val": [iou_frames(c, split["val"])], "train": [iou_frames(c, split["train"])]}
    print("start train", hist["train"][0], "val", hist["val"][0], flush=True)
    for outer in range(a.outer):
        JTJ = torch.zeros(P, P, device=dev, dtype=torch.float64); JTr = torch.zeros(P, device=dev, dtype=torch.float64); nres = 0
        vs = posed(c, split["train"])
        for t, v in zip(split["train"], vs):
            if not gate[t]: continue
            L = lab(t); H, W = L.shape; kk = K[t]
            hair = L == sc.HAIR; near_hair = Fn.max_pool2d(hair.float()[None, None], 7, 1, 3)[0, 0] > 0
            mask = (L != 0) & ~hair; mo = outline(mask) & ~near_hair; mo[[0, -1], :] = False; mo[:, [0, -1]] = False
            q = torch.nonzero(mo).flip(1).float()
            sil, _ = raster(v, kk, F, s, W, H); po = outline(sil) & ~near_hair; po[[0, -1], :] = False; po[:, [0, -1]] = False
            x = (kk[0, 0] * v[:, 0] / v[:, 2] + kk[0, 2]) * s; y = (kk[1, 1] * v[:, 1] / v[:, 2] + kk[1, 2]) * s; Pp = torch.stack([x, y], 1)
            vid = torch.cdist(torch.nonzero(po).flip(1).float(), Pp).argmin(1).unique()
            vid = vid[torch.isin(vpart[vid], torch.as_tensor(gate[t], device=dev))]
            if len(vid) < 10 or len(q) < 10: continue
            Pv = Pp[vid]; dq = torch.cdist(Pv, q); nn = dq.argmin(1); rvec = Pv - q[nn]
            rn = rvec.norm(dim=1); keep = rn < 30                                               # correspondences beyond 30 px are not shape
            vid, rvec, rn = vid[keep], rvec[keep], rn[keep]
            nrm = Fn.normalize(rvec, dim=1); res = rn                                           # point-to-outline distance along its direction
            w = torch.clamp(4.0 / rn.clamp_min(1e-6), max=1.0)                                  # Huber 4 px
            vv = v[vid]; z = vv[:, 2]; fx, fy = kk[0, 0] * s, kk[1, 1] * s
            Jp = torch.zeros(len(vid), 2, 3, device=dev); Jp[:, 0, 0] = fx / z; Jp[:, 0, 2] = -fx * vv[:, 0] / z**2; Jp[:, 1, 1] = fy / z; Jp[:, 1, 2] = -fy * vv[:, 1] / z**2
            Av = torch.einsum("vj,jab->vab", Wt[vid], Rj[t]); g = (Av @ nb[vid][..., None])[..., 0]   # posed direction of +normal
            dres = (nrm[:, None, :] @ Jp @ g[..., None])[:, 0, 0]                              # d res / d (normal disp)
            Jc = dres[:, None] * Phi[:, vid].T                                                  # (n, P)
            JTJ += ((Jc * w[:, None]).T @ Jc).double(); JTr += ((Jc * w[:, None]).T @ res).double(); nres += len(vid)
        reg_m = a.lam * float(torch.diagonal(JTJ).mean())                                      # relative to the data term
        A = JTJ + reg_m * (Lap.double() @ Lap.double() + a.ridge * torch.eye(P, device=dev, dtype=torch.float64)) + 1e-6 * torch.eye(P, device=dev, dtype=torch.float64)
        step = -torch.linalg.solve(A, JTr + reg_m * ((Lap.double() @ Lap.double() + a.ridge * torch.eye(P, device=dev, dtype=torch.float64)) @ c.double())).float()
        accepted = False
        for alpha in (1.0, 0.5, 0.25):
            cn = c + alpha * step
            cn = cn.clamp(-a.max_disp, a.max_disp)                                             # per-centre bound
            vi = iou_frames(cn, split["val"])
            if vi > hist["val"][-1]:
                c = cn; hist["val"].append(vi); hist["train"].append(iou_frames(c, split["train"])); accepted = True; break
        print(f"outer {outer}: residuals {nres}, step |{step.abs().max():.4f}|, accepted {accepted}, train {hist['train'][-1]:.4f} val {hist['val'][-1]:.4f}, max |disp| {(Phi.T @ c).abs().max():.4f} m, {time.time()-t0:.1f}s", flush=True)
        if not accepted: break
    test = (iou_frames(torch.zeros(P, device=dev), split["test"]), iou_frames(c, split["test"]))
    allv = posed(c, list(range(T))); np.save(O / "shaped-vertices.npy", torch.stack(allv).cpu().numpy().astype(np.float32))
    # PASS 1 (owner): apply the shape only on (frame, part) pairs that pass the gate; elsewhere GEM-X's geometry stays.
    # Per-vertex application weight = skin-weight share of passing parts (smooth across part borders).
    jpart_onehot = torch.zeros(len(names), len(pnames), device=dev); jpart_onehot[torch.arange(len(names)), jp] = 1
    vshare = Wt @ jpart_onehot                                                               # (V, parts)
    app_rows, pass1 = [], []
    # per-frame application weights first (exact exclusion), then a temporal ramp within shots: aw = min(raw, smoothed),
    # so rejected frames stay exactly 0 and their neighbours fade instead of popping
    shots_ = np.array([f["shot"] for f in fj["frames"]]); AW = []; OKP = []
    for t in range(T):
        okp = torch.zeros(len(pnames), device=dev); okp[[pnames.index(p) for p in pnames if ok(t, p)]] = 1
        rej = (vshare @ (1 - okp)).clamp(0, 1); aw = (1 - rej).clamp(0, 1)
        AW.append(torch.where(okp[vpart] > 0, aw, torch.zeros_like(aw))); OKP.append(okp)
    AW = torch.stack(AW); AWs = torch.zeros_like(AW)
    for t in range(T):
        idx = np.where(shots_ == shots_[t])[0]; w = torch.as_tensor(np.exp(-0.5 * ((idx - t) / a.ramp) ** 2), device=dev, dtype=torch.float32)
        AWs[t] = (w[:, None] * AW[idx]).sum(0) / w.sum()
    AW = torch.minimum(AW, AWs)
    for t in range(T):
        okp = OKP[t]; aw = AW[t]
        fixed_v = 0
        nb0 = torch.cross(V0[t][Fl[:, 1]] - V0[t][Fl[:, 0]], V0[t][Fl[:, 2]] - V0[t][Fl[:, 0]], dim=1); ok0 = nb0.norm(dim=1) > 1e-12
        for _ in range(8):
            vt = V0[t] + aw[:, None] * (allv[t] - V0[t])
            nn = torch.cross(vt[Fl[:, 1]] - vt[Fl[:, 0]], vt[Fl[:, 2]] - vt[Fl[:, 0]], dim=1)
            bad = ((nn * nb0).sum(1) < 0) & ok0
            if not bool(bad.any()): break
            hit = torch.zeros(len(aw), device=dev, dtype=torch.bool); hit[Fl[bad].reshape(-1)] = True
            for _r in range(2):
                fh = hit[Fl].any(1); hit[Fl[fh].reshape(-1)] = True
            fixed_v += int(hit.sum()); aw = torch.where(hit, torch.zeros_like(aw), aw)
        pass1.append(V0[t] + aw[:, None] * (allv[t] - V0[t]))
        app_rows.append({"frame": t, "appliedParts": [p for p in pnames if ok(t, p)], "appliedVertexShare": float(aw.mean()), "flipRepairVertices": fixed_v,
                         "rejectedOwnedMaxWeight": float((aw[okp[vpart] == 0]).max()) if bool((okp[vpart] == 0).any()) else 0.0,
                         "rejectedOwnedMaxDeltaM": float((pass1[-1] - V0[t]).norm(dim=1)[okp[vpart] == 0].max()) if bool((okp[vpart] == 0).any()) else 0.0})
    np.save(O / "shaped-vertices-pass1.npy", torch.stack(pass1).cpu().numpy().astype(np.float32))
    json.dump(app_rows, open(O / "pass1-application.json", "w"))
    def iou_list(vs, frames):
        out = []
        for t in frames:
            L = lab(t); H, W = L.shape; sil, _ = raster(vs[t], K[t], F, s, W, H); m = (L != 0) & (L != sc.HAIR); care = L != sc.HAIR
            out.append(float((sil & m).sum() / ((sil & care) | m).sum().clamp_min(1)))
        return float(np.mean(out))
    test_pass1 = iou_list(pass1, split["test"])
    disp = (Phi.T @ c)
    np.savez(O / "shape-basis.npz", centres=cen.cpu().numpy(), coef=c.cpu().numpy(), radius=a.radius, normal_disp=disp.cpu().numpy())
    frames_fully = sum(1 for r in app_rows if len(r["appliedParts"]) == len(pnames) - 1)
    nbr_sum = torch.zeros_like(disp); nbr_cnt = torch.zeros_like(disp)
    E2 = torch.cat([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]]); E2 = torch.cat([E2, E2.flip(1)])
    nbr_sum.index_add_(0, E2[:, 0], disp[E2[:, 1]]); nbr_cnt.index_add_(0, E2[:, 0], torch.ones(len(E2), device=dev))
    rough = (disp - nbr_sum / nbr_cnt.clamp_min(1)).abs()[body]
    summary = {"roughnessMm": {"mean": float(rough.mean()) * 1000, "p99": float(torch.quantile(rough, 0.99)) * 1000}, "pass1": {"framesAllPartsApplied": frames_fully, "framesNoPartApplied": sum(1 for r in app_rows if not r["appliedParts"]),
                         "meanAppliedVertexShare": float(np.mean([r["appliedVertexShare"] for r in app_rows]))}, "centres": P, "radius": a.radius, "history": hist, "testIoU": {"registeredOnly": test[0], "shapedUnmasked": test[1], "shapedPass1": test_pass1},
               "dispMetres": {"maxAbs": float(disp.abs().max()), "p95Abs": float(torch.quantile(disp.abs()[body], 0.95)), "meanSigned": float(disp[body].mean())},
               "seconds": time.time() - t0}
    json.dump(summary, open(O / "fit.json", "w"), indent=1); print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
