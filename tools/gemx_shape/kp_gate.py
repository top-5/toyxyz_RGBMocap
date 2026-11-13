"""Pose gate per (frame, part): GEM-X 3D joints (registered incam) projected vs 2D source keypoints (GEM-X image
head, recorder grid), error normalised by the person's mask height. Usage: kp_gate.py <gemx run> <capture> <registered.npy> <out.json> [--thr 0.035]"""
import json, sys, argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("capture"); ap.add_argument("verts"); ap.add_argument("out")
ap.add_argument("--thr", type=float, default=0.035); ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--ceiling", type=float, default=0.06); a = ap.parse_args()
for name in ("thr", "conf", "ceiling"):
    v = getattr(a, name)
    if not np.isfinite(v) or not 0 < v <= 1: raise SystemExit(f"--{name} must be finite in (0, 1]")
R, C = Path(a.run), Path(a.capture)
rig = np.load(R / "soma-rig.npz"); names = [str(n) for n in rig["names"]]; world = np.load(R / "soma-world.npy")
Vg = np.load(R / "soma-vertices.npy"); V = np.load(a.verts); K = np.load(R / "soma-incam.npz")["K_fullimg"].reshape(-1, 3, 3)
kp = np.load(C / "source-keypoints-67.npz"); kn = [str(n) for n in kp["joint_names"]]
fj = json.loads((C / "frames.json").read_text()); RW, RH = fj["resizedSize"]; SW, SH = fj["sourceSize"]; sx, sy = RW / SW, RH / SH
PARTS = {"Larm": ["LeftArm", "LeftForeArm", "LeftHand"], "Rarm": ["RightArm", "RightForeArm", "RightHand"],
         "Lleg": ["LeftUpLeg", "LeftLeg", "LeftFoot"], "Rleg": ["RightUpLeg", "RightLeg", "RightFoot"],
         "torso": ["Hips", "Spine1", "LeftShoulder", "RightShoulder", "Neck"], "head": ["Head"]}
KP2SOMA = {"LeftUpLeg": "LeftLeg", "LeftLeg": "LeftShin", "RightUpLeg": "RightLeg", "RightLeg": "RightShin", "Neck": "Neck1"}
def soma(j): return KP2SOMA.get(j, j)
for p, js in PARTS.items():
    for j in js: assert soma(j) in names and j in kn, (p, j)
T = V.shape[0]; rows = []; errs = {p: [] for p in PARTS}
for t in range(T):
    a0 = Vg[t] - Vg[t].mean(0); b0 = V[t] - V[t].mean(0); U, S, Vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(U @ Vt)); Rc = (U @ np.diag([1, 1, d]) @ Vt).T
    lab = np.load(C / f"labels/{t:04d}.npy"); ys = np.nonzero(lab != 0)[0]; hpx = int(ys.max() - ys.min()) if len(ys) else 0
    if hpx < 16: rows.append({"frame": t, "personHeightPx": hpx, "parts": {}}); continue          # no usable person: nothing admitted
    row = {"frame": t, "personHeightPx": int(hpx), "parts": {}}
    for p, js in PARTS.items():
        e = []
        if int(kp["frames"][t]) != t: raise ValueError(f"keypoint row {t} is frame {int(kp['frames'][t])}")
        for j in js:
            c = float(kp["confidence"][t, kn.index(j)]); sxy = kp["xy"][t, kn.index(j)]
            if not np.isfinite(c) or c < a.conf: continue
            if not np.isfinite(sxy).all() or not (0 <= sxy[0] < RW and 0 <= sxy[1] < RH): continue   # source point must be in the image
            X = Rc @ (world[t, names.index(soma(j)), :3, 3] - Vg[t].mean(0)) + V[t].mean(0)
            if not np.isfinite(X).all() or X[2] <= 0: continue
            u = (K[t, 0, 0] * X[0] / X[2] + K[t, 0, 2] + .5) * sx - .5; v = (K[t, 1, 1] * X[1] / X[2] + K[t, 1, 2] + .5) * sy - .5
            if not (0 <= u < RW and 0 <= v < RH): continue
            e.append(float(np.hypot(*(np.array([u, v]) - kp["xy"][t, kn.index(j)]))) / hpx)
        if len(e) >= (1 if p == "head" else 2):                                          # a limb needs >= 2 supported joints
            row["parts"][p] = {"err": float(np.max(e)), "n": len(e), "ok": bool(np.max(e) <= a.thr)}
            if t % 3 == 0: errs[p].append(np.max(e))                                     # limits from training frames only
    rows.append(row)
for p in PARTS:                                                                     # relative gate: flag abnormal frames for that part
    if not errs[p]: continue
    lim = min(a.ceiling, max(a.thr, 1.5 * float(np.median(errs[p]))))                    # absolute ceiling
    for r in rows:
        if p in r["parts"]: r["parts"][p]["ok"] = bool(r["parts"][p]["err"] <= lim); r["parts"][p]["limit"] = lim
summ = {p: {"limit": min(a.ceiling, max(a.thr, 1.5 * float(np.median(v)))), "nTrain": len(v), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "okShare": float(np.mean([r["parts"][p]["ok"] for r in rows if p in r["parts"]]))} for p, v in errs.items() if v}
json.dump({"threshold": a.thr, "conf": a.conf, "summary": summ, "frames": rows}, open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1))
