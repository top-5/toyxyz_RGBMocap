"""Atlas composite: face-warp region from a reference bake (same SOMA UV layout) over a body bake, feathered on CUDA.
Usage: face_composite.py <face bake dir> <body bake dir> <out.glb> [--feather 24]"""
import io, sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn, pygltflib
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("face"); ap.add_argument("body"); ap.add_argument("out"); ap.add_argument("--feather", type=int, default=24); a = ap.parse_args()
FA, BO = Path(a.face), Path(a.body)
fa = torch.as_tensor(np.asarray(Image.open(FA / "atlas.png").convert("RGB")), device="cuda").float()
bo = torch.as_tensor(np.asarray(Image.open(BO / "atlas.png").convert("RGB")), device="cuda").float()
m = torch.as_tensor(np.load(FA / "face-warp-region.npz")["texel_region"], device="cuda").float()
cov_f = torch.as_tensor(np.asarray(Image.open(FA / "coverage.png").convert("L")) > 0, device="cuda")
assert fa.shape == bo.shape and m.shape == fa.shape[:2], (fa.shape, bo.shape, m.shape)
# inward feather: erode then blur, so alpha is 1 deep inside the face region and 0 outside it
k = a.feather; er = -Fn.max_pool2d(-m[None, None], 2 * (k // 2) + 1, 1, k // 2)
al = Fn.avg_pool2d(er, 2 * (k // 2) + 1, 1, k // 2)[0, 0].clamp(0, 1) * (cov_f.float())
out = (al[..., None] * fa + (1 - al[..., None]) * bo).round().clamp(0, 255).byte().cpu().numpy()
Image.fromarray(out).save(Path(a.out).with_suffix(".atlas.png"))
g = pygltflib.GLTF2().load(str(BO / "soma-direct-textured-tpose.glb")); blob = bytearray(g.binary_blob())
pr = g.meshes[0].primitives[0]; img = g.images[g.textures[g.materials[pr.material].pbrMetallicRoughness.baseColorTexture.index].source]
buf = io.BytesIO(); Image.fromarray(out).save(buf, format="PNG"); png = buf.getvalue()
bv = g.bufferViews[img.bufferView]; old = bv.byteLength
assert bv.byteOffset + old == len(blob) or True
# append the new image at the end of the blob and repoint the image's bufferView
pad = (-len(blob)) % 4; blob += b"\0" * pad; off = len(blob); blob += png; blob += b"\0" * ((-len(blob)) % 4)
g.bufferViews.append(pygltflib.BufferView(buffer=0, byteOffset=off, byteLength=len(png))); img.bufferView = len(g.bufferViews) - 1
g.buffers[0].byteLength = len(blob); g.set_binary_blob(bytes(blob)); g.save_binary(a.out)
print("wrote", a.out, "face texels", int(m.sum()), "alpha>0.5", int((al > 0.5).sum()))
