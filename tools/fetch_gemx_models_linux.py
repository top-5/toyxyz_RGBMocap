"""Linux model fetch for the seed-linux-headless fork: download RGB MoCap's pinned GEM-X/SOMA-X/detector files into
DEST with exact size + sha256 checks; reuse identical local copies (hard link) when present."""
import hashlib, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/seed/checkpoints/gemx-rgbmocap")
LOCAL = [Path("/source/gem-x/src/inputs/checkpoints/vitpose/vitpose.pth"),
         Path("/source/gem-x/src/third_party/soma/assets/SOMA_neutral.npz")]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
    return h.hexdigest()

models = json.load(open(ROOT / "installer/gemx-models.json"))["models"]
for m in models:
    out = DEST / Path(m["relative"]).name
    want = m["expected_sha256"]
    if out.exists() and sha(out) == want:
        print("ok (present)", out.name, flush=True); continue
    reused = False
    for cand in LOCAL:
        if cand.name == out.name and cand.exists() and cand.stat().st_size >= m["minimum_bytes"] and sha(cand) == want:
            os.link(cand, out) if not out.exists() else None; print("ok (hard link)", out.name, "<-", cand, flush=True); reused = True; break
    if reused: continue
    url = m["download_url"]
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        dl = Path(tmp) / "download"
        subprocess.run(["curl", "-fL", "--retry", "5", "-sS", "-o", str(dl), url], check=True)
        if url.endswith(".zip"):
            with zipfile.ZipFile(dl) as z:
                name = next(n for n in z.namelist() if n.endswith("end2end.onnx"))
                with z.open(name) as src, open(Path(tmp) / "x", "wb") as dst:
                    while chunk := src.read(1 << 24): dst.write(chunk)
            dl = Path(tmp) / "x"
        got = sha(dl)
        if got != want or dl.stat().st_size < m["minimum_bytes"]:
            raise SystemExit(f"FAIL {out.name}: sha {got} size {dl.stat().st_size}")
        os.replace(dl, out); print("ok (downloaded)", out.name, dl.stat().st_size if dl.exists() else out.stat().st_size, flush=True)
print("ALL VERIFIED", DEST)
