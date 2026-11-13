# Seed Linux GEM-X CLI

Run the CUDA-only native77/SOMA pipeline with the existing seed environment:

```bash
timeout 900 /workspace/seed/.venv/bin/python tools/gemx_headless.py \
  /workspace/seed/projects/bald-european-video/bald-european-dance.mp4 \
  --out /workspace/scratch/rgbmocap-test/headless-full-06
```

Optional `--start-frame N --frames N` select the normalized **30 Hz** timeline.
Select 4–1800 frames within the source. The output directory must be new;
existing results are never overwritten. `--timeout` defaults to 900 seconds;
ffmpeg and ffprobe have separate bounds. The CLI does not install packages,
start the GUI, register runtimes, apply contact correction, or modify projects.
All generated images, derived model, Warp cache and results stay under `--out`.

## Verified full-video run

The exact command above exited 0 on 2026-11-13 (owner local date), using GEM-X
source `4aef1452e5df171deff4f293c08e643eb450981e` and the nine SHA-verified
assets in `/workspace/seed/checkpoints/gemx-rgbmocap`. The fetch log said
`ALL VERIFIED` before testing; the CLI also rechecks every model's SHA-256.
The three-shot source was processed as one fixed-camera sequence, without
shot splitting, at 704×1254 pixels, 284 frames, 30 fps. Scene cuts and the
fixed-camera assumption can affect reconstruction quality.

| Stage | Seconds |
|---|---:|
| Verify model hashes | 3.622 |
| Prepare 30 Hz working video | 2.737 |
| Import inference modules | 1.632 |
| Extract PNG frames | 3.312 |
| Detect/track primary person | 6.429 |
| Native77 2D observations | 7.213 |
| Static camera rotation | 0.0004 |
| Frozen body Pose features | 5.726 |
| Temporal motion | 0.573 |
| SOMA evaluation | 1.782 |
| Auxiliary bind sidecar | 0.014 |
| Publish | 0.173 |
| Total | **33.264** |

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition; image batch 4.
PyTorch peak allocated: **3,737,261,568 bytes (3.48 GiB)**; reserved:
3,925,868,544 bytes. NVML sampled this process every 0.2 seconds across GPUs:
**4,643,094,528 bytes (4.32 GiB)** peak including context, ORT and Warp.
Sampling can miss shorter transients. These are shared-host observations,
not isolated benchmark guarantees. Warp kernels were warmed by prior tests;
initial compilation can take longer.

`headless-full-06/report.json` contains timings/versions/memory;
`validation.json` records the actual array dimensions. Validation passed
`gemx_result.current()`, `gemx_result.load_rig()`, finite pose/mesh checks,
and auxiliary weight shape/normalization checks. `evaluate()` itself verifies
finite geometry and proper orthonormal joint rotations before publishing.

## Output contract

The CLI reuses upstream `build_body_conditions`, `load_predictor`, `run_motion`,
and RGB MoCap's unchanged `mocap.gemx_pipeline.evaluate`. The original single
floor offset and `gemx_result.FILES`/receipt semantics are retained.

| File | Actual full-video arrays |
|---|---|
| `soma-rig.npz` | names (77), parents (77), rest (77,4,4), faces (36108,3), ends (77,3) |
| `soma-world.npy` | float32 (284,77,4,4), joint-to-world, native Y up, metres |
| `soma-vertices.npy` | float32 (284,18056,3), MID surface with correctives |
| `soma-parameters.npz` | global_orient (284,3), body_pose (284,228), transl (284,3), identity_coeffs (284,128), scale_params (284,57) |
| `input.mp4` | 284 frames at 30 Hz, 704×1254 |
| `soma-result.json` | Original receipt fields/output stats, model hashes, seed runtime and derived-detector provenance |
| `project.json` | Native result metadata readable by `gemx_result.current` |
| `soma-bind.npz` | Auxiliary bind_vertices (18056,3), skin_weights (18056,77), for `tools/gemx_to_glb.py` |
| `report.json` | Status, stages, versions and scoped GPU memory measurements |

The rig archive remains exactly the original five keys. The auxiliary bind
surface receives the same neutral floor shift as `rest`; weights fold onto
the public hierarchy and omit virtual Root. The exporter accepts the sidecar
in commit `a36bfdf` (seeker-claude). Extra diagnostics are `frames/`,
`warp-cache/`, and a derived `yolox-prenms-cuda.onnx` under the run directory.
Failed runs keep diagnostics but publish no success receipt.

## Version and platform adaptations

No packages were installed, upgraded or downgraded. The actual imported
versions were Torch 2.9.1+cu128, Transformers 4.57.1, ONNX Runtime GPU 1.24.4,
Warp 1.18.0, SOMA-X 0.3.4, and **OpenCV 4.14.0**. Seed's imported custom
OpenCV build disables FFmpeg and GStreamer, despite the nominal environment
listing 4.12. The CLI uses PATH ffmpeg to extract PNG frames, retaining the
upstream bounded sequence/slice interface and RGB pixels. No Lightning is
needed by the inference imports or forward paths.

The CLI bypasses Windows runtime registration, exact package-version equality,
Model Manager, GUI workspace publishing and Windows subprocess flags. Explicit
seed paths replace runtime/HOME lookups; source and model pins are checked.

Transformers 5's `ALL_ATTENTION_FUNCTIONS.get_interface()` is mapped to
Transformers 4's `.get()` in a process-scoped adapter. The selected attention
backend and fallback function are retained; detector and body feature forward
paths passed on CUDA. Torch and Warp required no API adaptations.

A scoped SOMA factory accepts 0.3.4 and explicitly retains
`reference_pose={'version': '0.2.0'}`, MID geometry, native procedural transforms,
metres and the pinned corrective weights. The upstream factory accepts only
0.3.0/0.3.3. Both four-frame synthetic neutral evaluation and the full-video
pose/surface evaluation passed on CUDA. Sidecar weights use 0.3.4's
`public_skinning_weights()` method.

## A/B detector alternatives

**A — original graph, strict CUDA:** add `--detector strict-original`. This
reproduces the initialization failure in `headless-full-03/report.json`:
ONNX Runtime assigns nodes to CPU and rejects the graph when CPU fallback
is disabled. Verbose diagnostics name CPU-preferred Gather/ Less/ Where/
Reshape nodes after NMS. Disabling graph optimizations also failed. TensorRT
is advertised as an EP but its required library is absent in this environment.

**B — pre-NMS graph and CUDA Torch NMS (default, successful):**
`tools/gemx_onnx_cuda.py` derives a graph containing only ancestors of the
original decoded boxes/scores. It preserves all weights and non-node protobuf
fields verbatim, checks the pinned graph layout/NMS constants, and writes the
derived graph only into the run directory. No `onnx` package is needed.
The graph runs with `session.disable_cpu_ep_fallback=1` and runtime provider
fallback disabled. ORT registers a CPU provider, but the strict flag rejects
initialization if any node needs CPU; the derived graph initializes and runs.
CUDA IO binding keeps outputs on GPU, followed by torchvision CUDA NMS.

Original graph constants are retained: score >0.01, IoU 0.65, at most 200
per class and global top 100. The original wrapper then applies score >0.5,
discards labels and maps boxes to image pixels. Its `nms_thr=0.45` argument
has no effect on the original baked-NMS graph; the CUDA implementation uses
the actual baked threshold 0.65. Equal-score tie order versus ONNX has not
been measured. No CPU reference inference or numerical equivalence against
the Windows pinned runtime was run.

Owner A/B evidence and delivery receipts are recorded in
`/workspace/scratch/rgbmocap-test/gemx-owner-ab.md`. Claude's review artifact
`gemx-glb-contract-review.md` was read and its exporter sidecar fix verified
in the checkout. No GPU result or completed artifact is attributed to an
agent solely because a message was accepted or delivered.
