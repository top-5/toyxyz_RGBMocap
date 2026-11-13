"""N-panel MP4: source video | textured GEM-X GLBs (decoded + skinned on CUDA, body-centred camera from the front side),
labels in their own bars. Usage: render_gemx_ab.py out.mp4 label1=a.glb label2=b.glb ..."""
import io, subprocess, sys
from pathlib import Path
import numpy as np, torch, pygltflib
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
sys.path.insert(0, "/workspace/seed/src")
from seed.render import drtk_ops
FR = sorted(Path("/workspace/scratch/rgbmocap-test/headless-full-07/frames").glob("*.png"))
OUT = Path(sys.argv[1]); PANELS = [a.split("=", 1) for a in sys.argv[2:]]
H, W = 1254, 704
DT = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}; NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

def load(path):
    path, _, verts = str(path).partition("|")
    g = pygltflib.GLTF2().load(str(path)); b = g.binary_blob()
    def acc(i):
        a = g.accessors[i]; v = g.bufferViews[a.bufferView]
        return np.frombuffer(b, DT[a.componentType], a.count * NC[a.type], v.byteOffset + (a.byteOffset or 0)).reshape(a.count, -1)
    pr = g.meshes[0].primitives[0]
    assert verts or not pr.targets, "morph-target GLBs need |verts.npy (decoded positions) in this preview"
    img = g.images[g.textures[g.materials[pr.material].pbrMetallicRoughness.baseColorTexture.index].source]; iv = g.bufferViews[img.bufferView]
    atlas = np.asarray(Image.open(io.BytesIO(b[iv.byteOffset:iv.byteOffset + iv.byteLength])).convert("RGB"))
    d = lambda x: torch.as_tensor(np.ascontiguousarray(x), device="cuda")
    pos, uv = d(acc(pr.attributes.POSITION)), d(acc(pr.attributes.TEXCOORD_0).astype(np.float32))
    jt, wt = d(acc(pr.attributes.JOINTS_0).astype(np.int64)), d(acc(pr.attributes.WEIGHTS_0))
    faces = d(acc(pr.indices).reshape(-1, 3).astype(np.int32))
    skin = g.skins[0]; ibm = acc(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1); J = len(skin.joints)
    an = g.animations[0]; trs = {(c.target.node, c.target.path): acc(an.samplers[c.sampler].output) for c in an.channels}
    par = [-1] * J
    for i, n in enumerate(g.nodes[:J]):
        for c in (n.children or []): par[c] = i
    def mat(t, q, s):
        m = np.zeros((len(t), 4, 4)); m[:, :3, :3] = Rotation.from_quat(q).as_matrix() * s[:, None, :]; m[:, :3, 3] = t; m[:, 3, 3] = 1; return m
    local = np.stack([mat(trs[(j, "translation")], trs[(j, "rotation")], trs[(j, "scale")]) for j in range(J)], 1); world = local.copy()
    for j in range(J):
        if par[j] >= 0: world[:, j] = world[:, par[j]] @ local[:, j]
    extra = {}
    if verts:                                                                          # per-frame decoded positions (SOMA order) mapped to glTF vertices via corners
        soma_f = np.load("/workspace/scratch/rgbmocap-test/headless-full-07/soma-rig.npz")["faces"].astype(np.int64)
        m = np.full(len(pos), -1); m[acc(pr.indices).reshape(-1).astype(np.int64)] = soma_f.reshape(-1); assert (m >= 0).all()
        extra["frames"] = d(np.load(verts)[:, m].astype(np.float32))
    return dict(extra, pos=pos, uv=uv, jt=jt, wt=wt, faces=faces, skin=d((world @ ibm[None]).astype(np.float32)),
                atlas=d(atlas).permute(2, 0, 1)[None].float() / 255)

def render(m, t):
    if "frames" in m: v = m["frames"][t]
    else:
        blend = (m["wt"][..., None, None] * m["skin"][t][m["jt"]]).sum(1)
        v = (blend[:, :3, :3] @ m["pos"][..., None])[..., 0] + blend[:, :3, 3]
    v = v - v.mean(0); v = torch.stack([-v[:, 0], v[:, 1], -v[:, 2] - 2.6], 1)   # camera on -z of the GEM-X world
    f = 1.0 / np.tan(np.radians(18)); near, far = 0.1, 20.0
    clip = torch.stack([v[:, 0] * f * H / W, v[:, 1] * f, (-(far + near) / (far - near)) * v[:, 2] - 2 * far * near / (far - near), -v[:, 2]], 1)[None]
    r = drtk_ops.rasterize(clip, m["faces"], (H, W))
    it = drtk_ops.interpolate(m["uv"][None], r); it = it[0] if isinstance(it, tuple) else it
    it = it[0] if it.dim() == 4 and it.shape[0] == 1 else it
    uvi = it.permute(1, 2, 0) if it.shape[0] == 2 else it
    col = torch.nn.functional.grid_sample(m["atlas"], (uvi * 2 - 1)[None], mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    img = torch.where(r.valid[0][..., None], col, torch.tensor([0.19, 0.2, 0.21], device="cuda")).flip(0)
    return (img.clamp(0, 1) * 255).byte().cpu().numpy()

def bar(text):
    im = Image.new("RGB", (W, 56), (24, 24, 24)); dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except OSError: font = ImageFont.load_default()
    dr.text((14, 12), text, fill=(235, 235, 235), font=font); return np.asarray(im)

models = [(lab, load(p)) for lab, p in PANELS]
tmp = OUT.with_suffix(".frames")
if tmp.exists() and any(tmp.iterdir()): raise SystemExit(f"{tmp} not empty; pick a new output name")
tmp.mkdir(parents=True, exist_ok=True)
head = np.concatenate([bar("source video")] + [bar(l) for l, _ in models], 1)
with torch.no_grad():
    for t in range(len(FR)):
        vid = np.asarray(Image.open(FR[t]).convert("RGB").resize((W, H)))
        Image.fromarray(np.concatenate([head, np.concatenate([vid] + [render(m, t) for _, m in models], 1)], 0)).save(tmp / f"{t:04d}.png")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", str(tmp / "%04d.png"), "-frames:v", str(len(FR)),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(OUT)], check=True)
print("wrote", OUT)
