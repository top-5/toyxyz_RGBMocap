import sys, numpy as np, collections
rig = np.load("headless-full-07/soma-rig.npz"); F = rig["faces"].astype(np.int64); names = rig["names"]; W = np.load("headless-full-07/soma-bind.npz")["skin_weights"]; fj = names[W[F[:, 0]].argmax(1)]
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
na = np.cross(A[:, F[:, 1]] - A[:, F[:, 0]], A[:, F[:, 2]] - A[:, F[:, 0]]); nb = np.cross(B[:, F[:, 1]] - B[:, F[:, 0]], B[:, F[:, 2]] - B[:, F[:, 0]])
f = (np.einsum("tfk,tfk->tf", na, nb) < 0) & (np.linalg.norm(na, axis=2) > 1e-12)
print("flips", int(f.sum()), "frames", int(f.any(1).sum()), collections.Counter(fj[np.nonzero(f)[1]]).most_common(4))
