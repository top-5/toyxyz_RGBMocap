"""Decode a serialized GEM-X GLB (morph weights STEP + skin + node TRS at sampler times) and compare with target
vertices; geometry guards vs a baseline bank on the decoded result. Usage: decode_guard.py <glb> <targets.npy> <baseline.npy> <out.json>"""
import io, json, sys
import numpy as np, torch, pygltflib
from scipy.spatial.transform import Rotation
G, TGT, BASE, OUT = sys.argv[1:5]
DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}; NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
g = pygltflib.GLTF2().load(G); b = g.binary_blob()
def acc(i):
    a = g.accessors[i]; v = g.bufferViews[a.bufferView]; n = NC[a.type]
    arr = np.frombuffer(b, DT[a.componentType], a.count * n, v.byteOffset + (a.byteOffset or 0)).reshape(a.count, n)
    if a.sparse is not None: raise SystemExit("sparse accessors not handled")
    return arr
pr = g.meshes[0].primitives[0]; pos = acc(pr.attributes.POSITION).astype(np.float64)
jt = acc(pr.attributes.JOINTS_0).astype(np.int64); wt = acc(pr.attributes.WEIGHTS_0).astype(np.float64)
idx = acc(pr.indices).reshape(-1, 3).astype(np.int64)
targets = [acc(t.POSITION if hasattr(t, "POSITION") else t["POSITION"]).astype(np.float64) for t in pr.targets]
skin = g.skins[0]; J = len(skin.joints); ibm = acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
an = g.animations[0]; ch = {}
for c in an.channels:
    s = an.samplers[c.sampler]; ch[(c.target.node, c.target.path)] = (acc(s.input)[:, 0], acc(s.output), s.interpolation)
mesh_node = next(i for i, n in enumerate(g.nodes) if n.mesh is not None)
tw, wv, winterp = ch[(mesh_node, "weights")]; W = wv.reshape(len(tw), -1)
par = [-1] * len(g.nodes)
for i, n in enumerate(g.nodes):
    for c in (n.children or []): par[c] = i
times = ch[(skin.joints[0], "rotation")][0]; T = len(times)
def sample(node, path, t):
    tt, v, interp = ch[(node, path)]; k = int(np.searchsorted(tt, t + 1e-9) - 1); k = max(0, min(k, len(tt) - 1))
    if interp == "STEP" or k == len(tt) - 1: return v[k]
    a = (t - tt[k]) / (tt[k + 1] - tt[k]); return v[k] * (1 - a) + v[k + 1] * a if path != "rotation" else Rotation.from_quat([v[k], v[k + 1]]).as_quat()[0] if a == 0 else (Rotation.from_quat(v[k]) * Rotation.from_rotvec(a * (Rotation.from_quat(v[k]).inv() * Rotation.from_quat(v[k + 1])).as_rotvec())).as_quat()
tgt = np.load(TGT).astype(np.float64); base = np.load(BASE).astype(np.float64)
# glTF vertex -> SOMA vertex map: nearest at bind among targets is unreliable; use the exporter's split: compare per corner via faces
fsoma = None
errs, rep = [], {"frames": T, "weightsInterpolation": winterp, "decodedVsTargetMm": None}
dec_all = []
for f in range(T):
    t = times[f]; w = W[min(f, len(W) - 1)] if winterp == "STEP" else W[f]
    p = pos + sum(wi * targets[i] for i, wi in enumerate(w) if wi != 0)
    world = np.zeros((len(g.nodes), 4, 4)); order = list(range(len(g.nodes)))
    def local(n):
        node = g.nodes[n]; m = np.eye(4)
        tr = sample(n, "translation", t) if (n, "translation") in ch else np.array(node.translation or [0, 0, 0])
        q = sample(n, "rotation", t) if (n, "rotation") in ch else np.array(node.rotation or [0, 0, 0, 1])
        sc = sample(n, "scale", t) if (n, "scale") in ch else np.array(node.scale or [1, 1, 1])
        m[:3, :3] = Rotation.from_quat(q).as_matrix() * sc[None, :]; m[:3, 3] = tr; return m
    done = {}
    def wm(n):
        if n in done: return done[n]
        m = local(n) if par[n] < 0 else wm(par[n]) @ local(n); done[n] = m; return m
    Mj = np.stack([wm(j) @ ibm[k] for k, j in enumerate(skin.joints)])
    blend = (wt[..., None, None] * Mj[jt]).sum(1); v = (blend[:, :3, :3] @ p[..., None])[..., 0] + blend[:, :3, 3]
    dec_all.append(v)
dec = np.stack(dec_all)
# map glTF vertices to SOMA vertices through the triangle corners (same face order as the SOMA faces)
import numpy as _np
soma_faces = _np.load("/workspace/scratch/rgbmocap-test/headless-full-07/soma-rig.npz")["faces"].astype(_np.int64)
m = _np.full(len(pos), -1); m[idx.reshape(-1)] = soma_faces.reshape(-1)
assert (m >= 0).all()
d = np.linalg.norm(dec - tgt[:, m], axis=2) * 1000
rep["decodedVsTargetMm"] = {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}
# guards on decoded geometry (SOMA topology) vs baseline
V = np.zeros((T, tgt.shape[1], 3)); V[:, m] = dec
F = soma_faces
def fn(X):
    n = np.cross(X[:, F[:, 1]] - X[:, F[:, 0]], X[:, F[:, 2]] - X[:, F[:, 0]]); return n
nb, nd = fn(base), fn(V)
area_b, area_d = np.linalg.norm(nb, axis=2), np.linalg.norm(nd, axis=2)
flip = (np.einsum("tfk,tfk->tf", nb, nd) < 0) & (area_b > 1e-10)
E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
lb = np.linalg.norm(base[:, E[:, 0]] - base[:, E[:, 1]], axis=2); ld = np.linalg.norm(V[:, E[:, 0]] - V[:, E[:, 1]], axis=2)
strain = ld / np.maximum(lb, 1e-9); ar = area_d / np.maximum(area_b, 1e-12)
rep["guards"] = {"flippedTrianglesVsBaseline": {"total": int(flip.sum()), "framesWithAny": int(flip.any(1).sum()), "maxPerFrame": int(flip.sum(1).max())},
                 "edgeStrain": {"p01": float(np.percentile(strain, 1)), "p99": float(np.percentile(strain, 99)), "min": float(strain.min()), "max": float(strain.max()), "edgesOver1.5": int((strain > 1.5).sum()), "edgesUnder0.67": int((strain < 0.67).sum())},
                 "areaRatio": {"p01": float(np.percentile(ar, 1)), "p99": float(np.percentile(ar, 99)), "max": float(ar.max())},
                 "deltaVsBaselineMm": {"median": float(np.median(np.linalg.norm(V - base, axis=2)) * 1000), "max": float(np.linalg.norm(V - base, axis=2).max() * 1000)}}
names = _np.load("/workspace/scratch/rgbmocap-test/headless-full-07/soma-rig.npz")["names"]; Wsk = _np.load("/workspace/scratch/rgbmocap-test/headless-full-07/soma-bind.npz")["skin_weights"]
mv = np.linalg.norm(V - base, axis=2); tt, vv = np.unravel_index(mv.argmax(), mv.shape)
rep["guards"]["maxDeltaAt"] = {"frame": int(tt), "vertex": int(vv), "joint": str(names[Wsk[vv].argmax()])}
json.dump(rep, open(OUT, "w"), indent=1); print(json.dumps(rep, indent=1))
