import sys, json, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, "shape"); from gemx_iou import raster
sys.path.insert(0, "/workspace/seed/src"); from seed import sapiens_classes as sc
R=Path("headless-full-07"); C=Path(sys.argv[1]); frames=[int(x) for x in sys.argv[2].split(",")]; out=sys.argv[3]
V=torch.as_tensor(np.load(sys.argv[4] if len(sys.argv)>4 else R/"soma-incam-vertices.npy"),device="cuda")
K=torch.as_tensor(np.load(R/"soma-incam.npz")["K_fullimg"].reshape(-1,3,3),device="cuda",dtype=torch.float32)
F=torch.as_tensor(np.load(R/"soma-rig.npz")["faces"],device="cuda",dtype=torch.int32)
fj=json.loads((C/"frames.json").read_text()); s=fj["resizedSize"][1]/fj["sourceSize"][1]
tiles=[]
for t in frames:
    lab=np.load(C/f"labels/{t:04d}.npy"); H,W=lab.shape
    sil=raster(V[t],K[t],F,s,W,H)[0].cpu().numpy(); m=(lab!=0)&(lab!=sc.HAIR)
    img=np.zeros((H,W,3),np.uint8); rgb=np.asarray(Image.open(C/fj["frames"][t]["rgb"]).convert("RGB").resize((W,H)))
    img[:]=(rgb*0.5).astype(np.uint8); img[sil&m]=(200,200,200); img[m&~sil]=(255,60,60); img[sil&~m]=(60,120,255)
    tiles.append(img)
Image.fromarray(np.concatenate(tiles,1)).save(out); print(out)
