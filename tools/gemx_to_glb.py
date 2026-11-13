"""Headless GEM-X/SOMA result -> animated, skinned GLB (seed-linux-headless fork).

    python tools/gemx_to_glb.py <result dir> <out.glb> [--fps 30] [--max-influences 4]

Reads the gemx_result files written by tools/gemx_headless.py:
  soma-rig.npz   names (77,), parents (77,) into the 77 public joints (-1 = root), rest (77,4,4) bind world transforms,
                 faces (F,3), bind_vertices (V,3) bind mesh in the same frame as rest, skin_weights (V,77) public LBS
  soma-world.npy (T,77,4,4) per-frame world transforms (same frame as rest)
  soma-vertices.npy (T,V,3) optional exact posed surface (with correctives): used only to report the LBS error
Writes one skinned mesh, a 77-joint skeleton and one animation (per-joint local translation/rotation/scale, linear).
No GUI, no Torch.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def decompose(m):
    """(..., 4, 4) -> translation (...,3), quaternion xyzw (...,4), scale (...,3) for TRS matrices."""
    t = m[..., :3, 3]
    basis = m[..., :3, :3]
    s = np.linalg.norm(basis, axis=-2)
    r = basis / np.where(s[..., None, :] > 1e-12, s[..., None, :], 1.0)
    q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat().reshape(*m.shape[:-2], 4)
    return t, q, s


def local_from_world(world, parents):
    local = world.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            local[..., j, :, :] = np.linalg.inv(world[..., p, :, :]) @ world[..., j, :, :]
    return local


def top_k_weights(weights, k):
    idx = np.argsort(-weights, axis=1)[:, :k]
    w = np.take_along_axis(weights, idx, axis=1).astype(np.float32)
    w = np.clip(w, 0, None)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return idx.astype(np.uint16), w


class Buffer:
    def __init__(self):
        self.data, self.views, self.accessors = bytearray(), [], []

    def add(self, array, component, kind, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        while len(self.data) % 4:
            self.data += b"\0"
        view = dict(buffer=0, byteOffset=len(self.data), byteLength=array.nbytes)
        if target:
            view["target"] = target
        self.data += array.tobytes()
        self.views.append(view)
        acc = dict(bufferView=len(self.views) - 1, componentType=component, count=int(array.shape[0]), type=kind)
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"], acc["max"] = flat.min(0).tolist(), flat.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def lbs(bind_vertices, joints, weights, skin_matrices):
    """Vertices (T,V,3) from bind vertices, top-k joints/weights and per-frame skin matrices (T,J,4,4)."""
    v = np.concatenate([bind_vertices, np.ones((len(bind_vertices), 1))], 1)
    out = np.zeros((skin_matrices.shape[0], len(bind_vertices), 3))
    for k in range(joints.shape[1]):
        m = skin_matrices[:, joints[:, k]]                       # (T,V,4,4)
        out += weights[None, :, k, None] * np.einsum("tvij,vj->tvi", m, v)[..., :3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-influences", type=int, default=4)
    ap.add_argument("--correctives", action="store_true",
                    help="exact surface: per-frame SOMA pose-corrective offsets (inverse-skinned to bind space) as one-hot "
                         "animated morph targets; needs soma-vertices.npy; ~12 bytes x vertices per frame")
    a = ap.parse_args()

    rig = np.load(a.result / "soma-rig.npz", allow_pickle=False)
    names, parents = [str(n) for n in rig["names"]], rig["parents"].astype(int)
    rest, faces = rig["rest"].astype(np.float64), rig["faces"].astype(np.uint32)
    bind_vertices, weights = rig["bind_vertices"].astype(np.float64), rig["skin_weights"].astype(np.float64)
    world = np.load(a.result / "soma-world.npy").astype(np.float64)
    n_frames, n_joints = world.shape[:2]
    if rest.shape != (n_joints, 4, 4) or weights.shape != (len(bind_vertices), n_joints) or len(names) != n_joints:
        raise SystemExit(f"shape mismatch: rest {rest.shape} weights {weights.shape} joints {n_joints} names {len(names)}")

    joints, w = top_k_weights(weights, a.max_influences)
    inverse_bind = np.linalg.inv(rest)
    rest_local = local_from_world(rest, parents)
    anim_local = local_from_world(world, parents)

    buf = Buffer()
    pos = buf.add(bind_vertices.astype(np.float32), 5126, "VEC3", 34962, minmax=True)
    jnt = buf.add(joints, 5123, "VEC4", 34962)
    wgt = buf.add(w, 5126, "VEC4", 34962)
    idx = buf.add(faces.reshape(-1), 5125, "SCALAR", 34963)
    ibm = buf.add(np.ascontiguousarray(inverse_bind.transpose(0, 2, 1)).astype(np.float32).reshape(n_joints, 16), 5126, "MAT4")
    times = buf.add((np.arange(n_frames) / a.fps).astype(np.float32), 5126, "SCALAR", minmax=True)
    targets, morph_weights = [], None
    if a.correctives:
        posed = np.load(a.result / "soma-vertices.npy", mmap_mode="r")
        if posed.shape != (n_frames, len(bind_vertices), 3):
            raise SystemExit(f"--correctives: soma-vertices.npy {posed.shape} does not match {n_frames} frames x {len(bind_vertices)} vertices")
        ji = joints.astype(int)
        for t in range(n_frames):
            skin_t = world[t] @ inverse_bind                                      # (J,4,4)
            blend = np.einsum("vk,vkij->vij", w, skin_t[ji])                      # per-vertex blended skin matrix
            lbs_t = np.einsum("vij,vj->vi", blend[:, :3, :3], bind_vertices) + blend[:, :3, 3]
            delta = np.linalg.solve(blend[:, :3, :3], (np.asarray(posed[t], np.float64) - lbs_t)[..., None])[..., 0]
            targets.append(dict(POSITION=buf.add(delta.astype(np.float32), 5126, "VEC3", 34962, minmax=True)))
        morph_weights = buf.add(np.eye(n_frames, dtype=np.float32).reshape(-1), 5126, "SCALAR")

    nodes, channels, samplers = [], [], []
    rt, rq, rs = decompose(rest_local)
    at, aq, as_ = decompose(anim_local)
    for j, name in enumerate(names):
        nodes.append(dict(name=name, translation=rt[j].tolist(), rotation=rq[j].tolist(), scale=rs[j].tolist()))
    for j, p in enumerate(parents):
        if p >= 0:
            nodes[p].setdefault("children", []).append(j)
    for j in range(n_joints):
        for path, values, kind in (("translation", at[:, j], "VEC3"), ("rotation", aq[:, j], "VEC4"), ("scale", as_[:, j], "VEC3")):
            out = buf.add(np.ascontiguousarray(values).astype(np.float32), 5126, kind)
            samplers.append(dict(input=times, output=out, interpolation="LINEAR"))
            channels.append(dict(sampler=len(samplers) - 1, target=dict(node=j, path=path)))
    roots = [j for j, p in enumerate(parents) if p < 0]
    if morph_weights is not None:
        mesh_index = len(nodes)  # the mesh node is appended next
        samplers.append(dict(input=times, output=morph_weights, interpolation="STEP"))
        channels.append(dict(sampler=len(samplers) - 1, target=dict(node=mesh_index, path="weights")))
    mesh_node = len(nodes)
    nodes.append(dict(name="SOMA", mesh=0, skin=0))
    gltf = dict(
        asset=dict(version="2.0", generator="toyxyz_RGBMocap seed-linux-headless gemx_to_glb"),
        scene=0, scenes=[dict(nodes=roots + [mesh_node])], nodes=nodes,
        meshes=[dict(name="SOMA", primitives=[dict(attributes=dict(POSITION=pos, JOINTS_0=jnt, WEIGHTS_0=wgt), indices=idx,
                                                     **(dict(targets=targets) if targets else {}))],
                     **(dict(weights=[0.0] * len(targets)) if targets else {}))],
        skins=[dict(joints=list(range(n_joints)), inverseBindMatrices=ibm, skeleton=roots[0])],
        animations=[dict(name="GEM-X", channels=channels, samplers=samplers)],
        buffers=[dict(byteLength=len(buf.data))], bufferViews=buf.views, accessors=buf.accessors)
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    while len(buf.data) % 4:
        buf.data += b"\0"
    blob = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf.data))
    blob += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(buf.data), 0x004E4942) + bytes(buf.data)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(blob)

    report = dict(out=str(a.out), bytes=len(blob), frames=n_frames, joints=n_joints, vertices=len(bind_vertices),
                  faces=len(faces), fps=a.fps, maxInfluences=a.max_influences, correctiveMorphTargets=len(targets),
                  droppedWeightMax=float(1 - np.sort(weights, 1)[:, ::-1][:, :a.max_influences].sum(1).min()))
    exact = a.result / "soma-vertices.npy"
    if exact.is_file():
        posed = np.load(exact, mmap_mode="r")
        sample = np.unique(np.linspace(0, n_frames - 1, min(n_frames, 24)).astype(int))
        skin = world[sample] @ inverse_bind[None]
        err = np.linalg.norm(lbs(bind_vertices, joints.astype(int), w, skin) - posed[sample], axis=-1)
        if targets:  # check the exact path: skin (bind + this frame's morph) per sampled frame
            ji = joints.astype(int); errs = []
            for t in sample:
                skin_t = world[t] @ inverse_bind; blend = np.einsum("vk,vkij->vij", w, skin_t[ji])
                dlt = np.frombuffer(bytes(buf.data), np.float32, count=len(bind_vertices) * 3,
                                    offset=buf.views[buf.accessors[targets[t]["POSITION"]]["bufferView"]]["byteOffset"]).reshape(-1, 3)
                v = bind_vertices + dlt
                errs.append(np.linalg.norm(np.einsum("vij,vj->vi", blend[:, :3, :3], v) + blend[:, :3, 3] - posed[t], axis=-1))
            e2 = np.concatenate(errs)
            report["exactWithCorrectivesMm"] = dict(median=float(np.median(e2) * 1000), p95=float(np.percentile(e2, 95) * 1000), max=float(e2.max() * 1000))
        report["lbsVsExactMeshMm"] = dict(frames=len(sample), median=float(np.median(err) * 1000),
                                         p95=float(np.percentile(err, 95) * 1000), max=float(err.max() * 1000))
    (a.out.with_suffix(".json")).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
