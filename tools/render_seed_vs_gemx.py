"""3-panel MP4: source video | seed fit (textured) | GEM-X GLB (textured), same atlas, body-centred camera per panel."""
import glob, subprocess, sys
from pathlib import Path
import numpy as np, torch, pygltflib
from PIL import Image
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/workspace/seed/src")
from seed.render import drtk_ops
from seed.body_model.video_fit import load_pack
from seed.body_model import soma_forward_np as fk

RUN = glob.glob("/workspace/seed/projects/bald-european-video/runs/1280_*")[0]
SEED_TEX = glob.glob(RUN + "/fit/bake-idle-new/texture-bake/soma-direct-textured-tpose.glb")[0]
GEMX = Path("/workspace/scratch/rgbmocap-test/bald-gemx-textured.glb")
FRAMES = sorted(Path("/workspace/scratch/rgbmocap-test/headless-full-06/frames").glob("*.png"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/scratch/rgbmocap-test/glb/bald-seed-vs-gemx.mp4")
H, W = 1254, 704
DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}; NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

def gl(path):
    g = pygltflib.GLTF2().load(str(path)); b = g.binary_blob()
    def acc(i):
        a = g.accessors[i]; v = g.bufferViews[a.bufferView]
        return np.frombuffer(b, DT[a.componentType], a.count * NC[a.type], v.byteOffset + (a.byteOffset or 0)).reshape(a.count, -1)
    pr = g.meshes[0].primitives[0]
    img = g.images[g.textures[g.materials[pr.material].pbrMetallicRoughness.baseColorTexture.index].source]; iv = g.bufferViews[img.bufferView]
    import io
    atlas = np.asarray(Image.open(io.BytesIO(b[iv.byteOffset:iv.byteOffset + iv.byteLength])).convert("RGB"))
    return g, acc, pr, atlas

def render(verts, faces, uv, atlas_t, cam_sign):
    """verts (V,3) torch; body-centred camera on cam_sign*z looking at the body."""
    c = verts.mean(0); v = verts - c
    if cam_sign < 0: v = torch.stack([-v[:, 0], v[:, 1], -v[:, 2]], 1)
    f = 1.0 / np.tan(np.radians(18)); dist = 2.6
    v = torch.stack([v[:, 0], v[:, 1], v[:, 2] - dist], 1); near, far = 0.1, 20.0
    clip = torch.stack([v[:, 0] * f * H / W, v[:, 1] * f, (-(far + near) / (far - near)) * v[:, 2] - 2 * far * near / (far - near), -v[:, 2]], 1)[None]
    r = drtk_ops.rasterize(clip, faces, (H, W))
    it = drtk_ops.interpolate(uv[None], r); it = it[0] if isinstance(it, tuple) else it
    it = it[0] if it.dim() == 4 and it.shape[0] == 1 else it
    uvi = it.permute(1, 2, 0) if it.shape[0] == 2 else it                  # (H,W,2)
    grid = (uvi * 2 - 1)[None]
    col = torch.nn.functional.grid_sample(atlas_t, grid, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    bg = torch.tensor([0.19, 0.2, 0.21], device="cuda")
    img = torch.where(r.valid[0][..., None], col, bg).flip(0)
    return (img.clamp(0, 1) * 255).byte().cpu().numpy()

# --- seed fit 1280: per-frame posed vertices (correctives on), corner UVs from its own bake GLB
pack = load_pack(); p = np.load(RUN + "/fit/body/soma-direct-params.npz")
coeff = np.asarray(p["identity"], np.float64).reshape(1, -1); scale = float(p["scale"])
sg, sacc, spr, satlas = gl(SEED_TEX)
corner_uv = sacc(spr.attributes.TEXCOORD_0).astype(np.float32)
faces_np = np.asarray(pack["faces"], np.int64)
seed_faces = torch.as_tensor(np.arange(faces_np.size).reshape(-1, 3), device="cuda", dtype=torch.int32)
seed_uv = torch.as_tensor(corner_uv, device="cuda")
atlas_t = torch.as_tensor(satlas, device="cuda").permute(2, 0, 1)[None].float() / 255
# --- GEM-X textured GLB: decode + skin on CUDA from the file
g, acc, pr, gatlas = gl(GEMX)
gpos = torch.as_tensor(acc(pr.attributes.POSITION), device="cuda"); guv = torch.as_tensor(acc(pr.attributes.TEXCOORD_0).astype(np.float32), device="cuda")
gj = torch.as_tensor(acc(pr.attributes.JOINTS_0).astype(np.int64), device="cuda"); gw = torch.as_tensor(acc(pr.attributes.WEIGHTS_0), device="cuda")
gf = torch.as_tensor(acc(pr.indices).reshape(-1, 3).astype(np.int32), device="cuda")
skin = g.skins[0]; ibm = acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1); J = len(skin.joints)
an = g.animations[0]; trs = {}
for ch in an.channels:
    trs[(ch.target.node, ch.target.path)] = acc(an.samplers[ch.sampler].output)
par = [-1] * J
for i, n in enumerate(g.nodes[:J]):
    for cc in (n.children or []): par[cc] = i
def mat(t, q, s):
    m = np.zeros((len(t), 4, 4)); m[:, :3, :3] = Rotation.from_quat(q).as_matrix() * s[:, None, :]; m[:, :3, 3] = t; m[:, 3, 3] = 1; return m
local = np.stack([mat(trs[(j, "translation")], trs[(j, "rotation")], trs[(j, "scale")]) for j in range(J)], 1)
world = local.copy()
for j in range(J):
    if par[j] >= 0: world[:, j] = world[:, par[j]] @ local[:, j]
gskin = torch.as_tensor(world @ ibm[None], device="cuda", dtype=torch.float32)
gatlas_t = torch.as_tensor(gatlas, device="cuda").permute(2, 0, 1)[None].float() / 255

tmp = OUT.with_suffix(".frames"); tmp.mkdir(parents=True, exist_ok=True)
font = None
with torch.no_grad():
    for t in range(len(FRAMES)):
        o = fk.forward(pack, coeff, p["poses"][t:t + 1].astype(np.float64), correctives=True)
        sv = torch.as_tensor((o["vertices"][0] * scale)[faces_np.reshape(-1)], device="cuda", dtype=torch.float32)  # per corner
        a_img = render(sv, seed_faces, seed_uv, atlas_t, +1)
        m = gskin[t][gj]; blend = (gw[..., None, None] * m).sum(1)
        gv = (blend[:, :3, :3] @ gpos[..., None])[..., 0] + blend[:, :3, 3]
        b_img = render(gv, gf, guv, gatlas_t, -1)
        vid = np.asarray(Image.open(FRAMES[t]).convert("RGB").resize((W, H)))
        Image.fromarray(np.concatenate([vid, a_img, b_img], 1)).save(tmp / f"{t:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT)], check=True)
print("wrote", OUT)
