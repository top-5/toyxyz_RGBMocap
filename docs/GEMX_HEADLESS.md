# Seed Linux GEM-X CLI (blocked at strict CUDA detector initialization)

`tools/gemx_headless.py` uses only `/workspace/seed/.venv/bin/python`, the
GEM-X source at `4aef1452e5df171deff4f293c08e643eb450981e`, and the nine
SHA-256-verified assets in `/workspace/seed/checkpoints/gemx-rgbmocap`.
It does not install packages, start the GUI, register runtimes, or change an
offline project. The output directory must be new. Frame arguments use the
normalized 30 Hz timeline, with the original 4–1800 frame limit. The default
whole-run timeout is 900 seconds; ffmpeg and ffprobe have separate bounds.

```bash
timeout 900 /workspace/seed/.venv/bin/python tools/gemx_headless.py \
  /workspace/seed/projects/bald-european-video/bald-european-dance.mp4 \
  --out /workspace/scratch/rgbmocap-test/headless-full-03
```

This exact test exited 1 after 13.83 seconds on 2026-11-12 UTC. The models'
fetch log said `ALL VERIFIED` before running. Timings from `report.json`:

| Stage | Seconds |
| --- | ---: |
| Verify model hashes | 3.908 |
| Prepare 30 Hz video | 3.899 |
| Import inference modules | 2.103 |
| Extract PNG frames | 3.662 |
| Detector initialization (failed) | 0.191 |

The prepared `input.mp4` contains 284 frames at 30 fps, 704×1254 pixels.
The `frames/` directory has 284 RGB-readable PNGs. Only these preparation
artifacts and the failure report were produced; there are **no reconstructed
poses, meshes, parameters, rig or success receipt**. The three shots were not
split: configuration assumes a fixed camera throughout.

ONNX Runtime 1.24.4 fails to initialize the pinned detector when
`session.disable_cpu_ep_fallback=1`. Exact error:

> This session contains graph nodes that are assigned to the default CPU EP,
> but fallback to CPU EP has been explicitly disabled by the user.

The verbose initialization log at
`/workspace/scratch/rgbmocap-test/ort-strict-cuda.log` identifies CPU-preferred
`Gather_760`, `Less_763`, `Where_764`, and `Reshape_765`. No detector inference
was run on CPU. No package changes were attempted. The `onnx` Python package
is absent, so graph editing with that package is unavailable in this venv.
Further reconstruction stops here under the requested no-CPU-fallback rule.

PyTorch allocator peak was zero bytes because failure occurred before Torch
model loading. This is **not** a measurement of ONNX Runtime's GPU usage.
The CLI labels allocator measurements explicitly; whole-process GPU peak has
not been measured.

## Adaptations implemented

- Bypass Windows Python/runtime registration, version equality checks,
  Model Manager and GUI workspace publishing with explicit seed paths.
  Hashes still come from RGB MoCap's committed model manifest.
- Use PATH ffmpeg/ffprobe with Linux subprocess calls. The actual imported
  OpenCV is **4.14.0**, despite the nominal 4.12 environment description. Its
  build disables FFmpeg and GStreamer, so the CLI uses disk-backed ffmpeg PNG
  extraction with GEM-X's sequence/slice interface instead of VideoCapture.
- Set YOLOX to CUDA and disable both session CPU-node fallback and runtime
  provider fallback. This deliberately fails rather than running CPU nodes.
- A process-scoped SOMA factory accepts installed 0.3.4 and explicitly retains
  `reference_pose={'version': '0.2.0'}`, MID geometry, native procedural
  transforms, metres and the pinned corrective weights. Upstream's factory
  only accepts 0.3.0/0.3.3. The 0.3.4 constructor has the required arguments;
  actual geometry evaluation is **untested** due to the earlier detector failure.
- Torch 2.9.1, Transformers 4.57.1 and Warp 1.18.0 import successfully without
  changes. GEM-X inference imports do not require Lightning. Their model
  forward paths have **not** been reached or validated.

## Intended successful output contract (not yet validated end to end)

The CLI calls upstream `build_body_conditions`, `load_predictor`, `run_motion`,
and RGB MoCap's unchanged `mocap.gemx_pipeline.evaluate`. It writes the original
single floor offset and preserves `gemx_result.FILES` and receipt semantics:

| File | Arrays / shape |
| --- | --- |
| `soma-rig.npz` | names (77), parents (77), rest (77,4,4), faces (F,3), ends (77,3) |
| `soma-world.npy` | float32 (T,77,4,4), joint-to-world, native Y up, metres |
| `soma-vertices.npy` | float32 (T,18056,3), MID surface with correctives |
| `soma-parameters.npz` | global_orient (T,3), body_pose (T,228), transl (T,3), identity_coeffs (T,128), scale_params (T,57) |
| `input.mp4` | Normalized 30 Hz working video |
| `soma-result.json` | Original numeric receipt fields, model hashes and output file stats; seed runtime provenance |
| `project.json` | Native result metadata readable by `gemx_result.current` |
| `report.json` | Status, stage times, versions and explicitly scoped PyTorch VRAM peaks |

No correction, retargeting, GLB export, playback, or visual acceptance was run.
Failed runs retain diagnostics without publishing a success receipt.
