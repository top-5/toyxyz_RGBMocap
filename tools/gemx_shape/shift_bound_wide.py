import sys, json, numpy as np, torch, torch.nn.functional as Fn
from pathlib import Path
sys.path.insert(0,"shape"); from gemx_iou import raster
R=Path("headless-full-07"); C=Path(sys.argv[1])
V=torch.as_tensor(np.load(R/"soma-incam-vertices.npy"),device="cuda")
K=torch.as_tensor(np.load(R/"soma-incam.npz")["K_fullimg"].reshape(-1,3,3),device="cuda",dtype=torch.float32)
F=torch.as_tensor(np.load(R/"soma-rig.npz")["faces"],device="cuda",dtype=torch.int32)
fj=json.loads((C/"frames.json").read_text()); s=fj["resizedSize"][1]/fj["sourceSize"][1]
shifts=torch.arange(-48,49,8,device="cuda").float(); scales=torch.tensor([1.0,1.04,1.08,1.12,1.16,1.20],device="cuda")
best=[];base=[];params=[]
for t in range(0,V.shape[0],8):
    lab=torch.as_tensor(np.load(C/f"labels/{t:04d}.npy"),device="cuda"); H,W=lab.shape
    sil=raster(V[t],K[t],F,s,W,H)[0].float(); m=((lab!=0)&(lab!=4)).float(); care=(lab!=4).float()
    ys,xs=torch.nonzero(sil,as_tuple=True); cy,cx=ys.float().mean(),xs.float().mean()
    def iou(S): i=(S*m).sum(); u=((S*care)+m-(S*care*m)).sum(); return (i/u).item()
    base.append(iou(sil)); b=(base[-1],0,0,1.0)
    for sc_ in scales:
        for dy in shifts:
            # affine grid: output pixel p samples input at cx + (p - cx - dx)/sc
            for dx in shifts:
                th=torch.tensor([[1/sc_,0,0],[0,1/sc_,0]],device="cuda")
                ox=(cx+dx)*2/W-1; oy=(cy+dy)*2/H-1; icx=cx*2/W-1; icy=cy*2/H-1
                th[0,2]=icx-ox/sc_; th[1,2]=icy-oy/sc_
                g=Fn.affine_grid(th[None],(1,1,H,W),align_corners=False)
                S=(Fn.grid_sample(sil[None,None],g,align_corners=False)[0,0]>0.5).float()
                v=iou(S)
                if v>b[0]: b=(v,dx.item(),dy.item(),sc_.item())
    best.append(b[0]); params.append(b[1:])
    if t%40==0: print(t,base[-1],b,flush=True)
best=np.array(best);base=np.array(base);P=np.array(params)
print("base",base.mean(),"best shift+scale",best.mean(),"p10",np.percentile(best,10),"median |dx|,|dy|",np.median(np.abs(P[:,0])),np.median(np.abs(P[:,1])),"scale med",np.median(P[:,2]))
json.dump({"base":base.tolist(),"best":best.tolist(),"params":P.tolist()},open("shape/shift-bound-wide.json","w"))
