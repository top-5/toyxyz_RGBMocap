"""Decode an animated skinned GLB (pygltflib), skin it on CUDA from the file's own data, rasterize with DRTK (seed), and write
a side-by-side MP4 with the source video frames. Verifies the GLB content, not the npz it came from.
  python tools/render_glb_vs_video.py <out.glb> <frames dir> <out.mp4>   (camera on -z of the GEM-X world)
"""
import subprocess, sys
from pathlib import Path
import numpy as np, torch, pygltflib
from scipy.spatial.transform import Rotation
from PIL import Image
sys.path.insert(0, "/workspace/seed/src")
from seed.render import drtk_ops

glb, frames_dir, out_mp4 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = pygltflib.GLTF2().load(str(glb)); blob = g.binary_blob()
DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}; NC = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
def acc(i):
    a = g.accessors[i]; v = g.bufferViews[a.bufferView]
    return np.frombuffer(blob, DT[a.componentType], a.count * NC[a.type], v.byteOffset + (a.byteOffset or 0)).reshape(a.count, -1)
prim = g.meshes[0].primitives[0]
pos = acc(prim.attributes.POSITION).astype(np.float64); jnt = acc(prim.attributes.JOINTS_0).astype(int); wgt = acc(prim.attributes.WEIGHTS_0).astype(np.float64)
faces = acc(prim.indices).reshape(-1, 3).astype(np.int64)
skin = g.skins[0]; ibm = acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
J = len(skin.joints); anim = g.animations[0]
T = None; trs = {}
for ch in anim.channels:
    s = anim.samplers[ch.sampler]; vals = acc(s.output); T = len(acc(s.input)) if T is None else T
    trs[(ch.target.node, ch.target.path)] = vals
parents = [-1] * J
for i, n in enumerate(g.nodes[:J]):
    for c in (n.children or []): parents[c] = i
def mat(t, q, s):
    m = np.zeros((len(t), 4, 4)); m[:, :3, :3] = Rotation.from_quat(q).as_matrix() * s[:, None, :]; m[:, :3, 3] = t; m[:, 3, 3] = 1; return m
local = np.stack([mat(trs[(j, "translation")], trs[(j, "rotation")], trs[(j, "scale")]) for j in range(J)], 1)  # (T,J,4,4)
world = local.copy()
for j in range(J):
    if parents[j] >= 0: world[:, j] = world[:, parents[j]] @ local[:, j]
skinm = torch.as_tensor(world @ ibm[None], device="cuda", dtype=torch.float32)
P = torch.as_tensor(pos, device="cuda", dtype=torch.float32); Jt = torch.as_tensor(jnt, device="cuda"); Wt = torch.as_tensor(wgt, device="cuda", dtype=torch.float32)
F = torch.as_tensor(faces, device="cuda", dtype=torch.int32)
H, W = 1254, 704
src = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
tmp = out_mp4.with_suffix(".frames"); tmp.mkdir(parents=True, exist_ok=True)
allv = []
with torch.no_grad():
    for t in range(T):
        m = skinm[t][Jt]                                              # (V,4,4,4)
        blend = (Wt[..., None, None] * m).sum(1)
        v = (blend[:, :3, :3] @ P[..., None])[..., 0] + blend[:, :3, 3]
        allv.append(v)
    V = torch.stack(allv)                                              # (T,V,3)
    center = V.reshape(-1, 3).mean(0); ext = (V.reshape(-1, 3).max(0).values - V.reshape(-1, 3).min(0).values).max()
    f = 1.0 / np.tan(np.radians(18)); dist = float(ext) * 1.7
    for t in range(T):
        v = V[t] - center; v = torch.stack([-v[:, 0], v[:, 1], -v[:, 2] - dist], 1)     # camera on -z of the GEM-X world (180 deg about y), looking at the body
        near, far = 0.1, dist * 4
        clip = torch.stack([v[:, 0] * f * H / W, v[:, 1] * f, (-(far + near) / (far - near)) * v[:, 2] - 2 * far * near / (far - near), -v[:, 2]], 1)[None]
        r = drtk_ops.rasterize(clip, F, (H, W))
        e1 = v[F[:, 1].long()] - v[F[:, 0].long()]; e2 = v[F[:, 2].long()] - v[F[:, 0].long()]
        fn = torch.nn.functional.normalize(torch.cross(e1, e2, dim=1), dim=1)
        vn = torch.zeros_like(v).index_add_(0, F[:, 0].long(), fn).index_add_(0, F[:, 1].long(), fn).index_add_(0, F[:, 2].long(), fn)
        vn = torch.nn.functional.normalize(vn, dim=1)
        n = drtk_ops.interpolate(vn[None], r)[0] if isinstance(drtk_ops.interpolate(vn[None], r), tuple) else drtk_ops.interpolate(vn[None], r)
        n = n[0] if n.dim() == 4 and n.shape[0] == 1 else n
        n = n.permute(1, 2, 0) if n.shape[0] == 3 else n
        shade = (0.25 + 0.75 * n[..., 2].clamp(0, 1))[..., None].expand(-1, -1, 3)
        img = torch.where(r.valid[0][..., None], shade * torch.tensor([0.85, 0.8, 0.75], device="cuda"), torch.tensor([0.19, 0.2, 0.21], device="cuda"))
        img = (img.flip(0).clamp(0, 1) * 255).byte().cpu().numpy()       # GL rows are bottom-first
        vid = np.asarray(Image.open(src[t]).convert("RGB").resize((W, H)))
        Image.fromarray(np.concatenate([vid, img], 1)).save(tmp / f"{t:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out_mp4)], check=True)
print("frames", T, "joints", J, "vertices", len(pos), "->", out_mp4)
