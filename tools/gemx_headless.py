#!/usr/bin/env python3
"""CUDA-only Linux GEM-X/SOMA reconstruction using the existing seed venv.

Frame arguments refer to RGB MoCap's normalized 30 Hz timeline. Outputs use
mocap.gemx_result's native cache contract; no GUI/workspace is opened.
"""

import argparse
from contextlib import contextmanager
import gc
import importlib.metadata
import json
import os
import operator
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path('/workspace/seed/external/GEM-X')
PYTHON = Path('/workspace/seed/.venv/bin/python')
MODELS = Path('/workspace/seed/checkpoints/gemx-rgbmocap')
REVISION = '4aef1452e5df171deff4f293c08e643eb450981e'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENGINE))

from mocap.gemx_result import FILES, FPS, PARAMETERS, RECEIPT
from mocap.result_artifacts import file_stat
from mocap.storage import atomic_json, sha256


class Stages:
    def __init__(self, torch):
        self.torch = torch
        self.times = {}
        self.name = None
        self.started = None

    def begin(self, name):
        self.finish()
        print(name, flush=True)
        self.name, self.started = name, time.monotonic()

    def finish(self):
        if self.name is not None:
            self.torch.cuda.synchronize()
            self.times[self.name] = self.times.get(self.name, 0) + time.monotonic() - self.started
            self.name = None


@contextmanager
def seed_adapters():
    """Scope compatibility changes to this process, leaving upstream untouched."""
    import onnxruntime as ort
    import soma
    from soma import SOMALayer, Unit
    import gem.common.soma.body as body
    import gem.inference.body_video as video

    if soma.__version__ != '0.3.4':
        raise RuntimeError('This seed adapter requires py-soma-x 0.3.4')
    original_factory, original_detector = body._build_official_soma_runtime, video.YOLOXONNX

    def soma_runtime(config):
        if config.procedural_reference != 'runtime' or str(config.device) != 'cuda':
            raise ValueError('Headless Motion requires native procedural transforms on CUDA')
        options = dict(data_root=str(config.data_root), identity_model_type='soma',
                       lod=config.lod, mode=config.mode, device='cuda', output_unit=Unit.METERS,
                       enable_procedural_transforms=config.enable_procedural_transforms,
                       reference_pose={'version': '0.2.0'})
        if not config.apply_correctives:
            options['correctives_model_path'] = None
        return SOMALayer(**options)

    class CudaDetector(original_detector):
        def __init__(self, onnx_path, input_size=(640, 640), score_thr=0.5,
                     nms_thr=0.45, device='cuda'):
            if device != 'cuda' or 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('YOLOX requires ONNX Runtime CUDA')
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            self.sess = ort.InferenceSession(str(onnx_path), sess_options=options,
                                            providers=['CUDAExecutionProvider'])
            self.sess.disable_fallback()
            self.execution_providers = self.sess.get_providers()
            if self.execution_providers != ['CUDAExecutionProvider']:
                raise RuntimeError(f'Unexpected detector providers: {self.execution_providers}')
            self.input_size, self.score_thr, self.nms_thr = input_size, score_thr, nms_thr

    body._build_official_soma_runtime = soma_runtime
    video.YOLOXONNX = CudaDetector
    try:
        yield
    finally:
        body._build_official_soma_runtime, video.YOLOXONNX = original_factory, original_detector


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_streams', '-show_format', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True, timeout=30)
    data = json.loads(result.stdout)
    stream = data['streams'][0]
    numerator, denominator = map(int, stream['avg_frame_rate'].split('/'))
    fps = numerator / denominator
    duration = float(stream.get('duration', data['format'].get('duration', 0)))
    seek_offset = float(stream.get('start_time', 0)) - float(data['format'].get('start_time', 0))
    return dict(width=int(stream['width']), height=int(stream['height']), fps=fps,
                frames=int(stream.get('nb_frames', round(duration * fps))), duration=duration,
                seek_offset=seek_offset)


def run(args):
    if Path(sys.executable).absolute() != PYTHON:
        raise RuntimeError(f'Run with {PYTHON}; packages must not be installed or changed')
    revision = subprocess.check_output(['git', '-C', str(ENGINE), 'rev-parse', 'HEAD'],
                                       text=True, timeout=10).strip()
    if revision != REVISION:
        raise RuntimeError(f'GEM-X revision differs from pin: {revision}')
    video = args.video.resolve(strict=True)
    source = probe(video)
    length = args.frames if args.frames is not None else round(source['duration'] * FPS) - args.start_frame
    start = args.start_frame / FPS
    if args.start_frame < 0 or not 4 <= length <= 1800 or start + length / FPS > source['duration'] + .002:
        raise ValueError('Select 4–1800 frames within the source, on the 30 fps timeline')
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    args.created_output = out
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      WARP_CACHE_PATH=str(out / 'warp-cache'))
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU inference fallback')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    stages = Stages(torch)
    args.stages = stages
    began = time.monotonic()
    args.began = began
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'onnxruntime-gpu', 'warp-lang', 'py-soma-x')}
    import cv2
    versions['opencv_imported'] = cv2.__version__
    args.versions = versions
    stages.begin('verify_models')
    manifest = json.loads((ROOT / 'installer/gemx-models.json').read_text())['models']
    models = {}
    for row in manifest:
        path = MODELS / Path(row['relative']).name
        if path.stat().st_size != row['minimum_bytes'] or sha256(path) != row['expected_sha256']:
            raise ValueError(f'Model verification failed: {path}')
        models[row['key']] = path
    stamps = {path: file_stat(path) for path in (video, *models.values())}
    stages.begin('prepare_video')
    clip = out / 'input.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', f'{start + source["seek_offset"]:.9f}',
                    '-i', str(video), '-map', '0:v:0', '-vf',
                    'fps=30:start_time=0,pad=ceil(iw/2)*2:ceil(ih/2)*2', '-frames:v', str(length),
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-an', '-movflags', '+faststart', str(clip)],
                   check=True, capture_output=True, timeout=min(args.timeout, 180))
    media = probe(clip)
    if abs(media['fps'] - FPS) > .001 or not max(4, length - 1) <= media['frames'] <= length:
        raise ValueError(f'Unexpected working video: {media}')
    length = media['frames']
    args.prepared_frames = length
    stages.begin('load_inference_modules')
    from gem.inference.body_video import build_body_conditions, load_body_video_config
    from gem.inference.motion import load_predictor, run_motion
    from gem.inference.video import VideoFrames
    from gem.motion.geometry import MotionGeometry
    from mocap.gemx_pipeline import evaluate
    # Seed's custom CUDA OpenCV has no video decoder. Preserve ffmpeg pixels
    # on disk, with the upstream bounded slice-reading interface.
    stages.begin('extract_frames')
    images = out / 'frames'
    images.mkdir()
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(clip),
                    '-threads', '2', str(images / '%06d.png')],
                   check=True, capture_output=True, timeout=min(args.timeout, 180))

    class ImageFrames(VideoFrames):
        def __init__(self, path):
            self.path = str(path)
            self.width, self.height, self.fps = media['width'], media['height'], FPS
            self.paths = sorted(images.glob('*.png'))
            self.frame_count = len(self.paths)
            if self.frame_count != length:
                raise ValueError('ffmpeg extracted an unexpected frame count')

        def __getitem__(self, index):
            import cv2
            if isinstance(index, slice):
                return [self[i] for i in range(*index.indices(len(self)))]
            index = operator.index(index)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            frame = cv2.imread(str(self.paths[index]))
            if frame is None:
                raise OSError(f'Cannot read {self.paths[index]}')
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        def close(self):
            pass
    config = load_body_video_config(ENGINE / 'docs/configs/body_video_native77.yaml')
    for section, key in (('pose', 'gemx_pose'), ('observations', 'gemx_observations'),
                         ('detector', 'gemx_detector'), ('motion', 'gemx_motion')):
        config[section]['checkpoint'] = str(models[key])
    config['detector']['device'] = 'cuda'
    config['camera']['mode'] = 'static'
    config['motion']['stats'] = str(models['gemx_stats'])
    free, _ = torch.cuda.mem_get_info()
    batch = 1 if free < 5 * 1024**3 else 4
    with seed_adapters():
        for attempt in range(2):
            try:
                for section in ('pose', 'observations'):
                    config[section]['batch_size'] = batch
                with ImageFrames(clip) as frames:
                    sequence, _, _ = build_body_conditions(frames, config, device='cuda', progress=stages.begin)
                break
            except torch.cuda.OutOfMemoryError:
                if batch == 1 or attempt:
                    raise
                batch = 1
                gc.collect()
                torch.cuda.empty_cache()
        stages.begin('temporal_motion')
        predictor = load_predictor(config['motion']['checkpoint'], config['motion'],
                                   stats_path=config['motion']['stats'], device='cuda')
        result = run_motion(predictor, sequence, device='cuda')
        parameters = {key: value.detach().cpu() for key, value in result['pred_soma_params_global'].items()}
        shapes = dict(global_orient=3, body_pose=228, transl=3, identity_coeffs=128, scale_params=57)
        if set(parameters) != set(shapes):
            raise ValueError('Unexpected SOMA parameters')
        for key, width in shapes.items():
            value = parameters[key]
            if key in ('identity_coeffs', 'scale_params') and value.shape == (1, width):
                value = value[:, None].expand(1, length, width)
            if value.shape != (1, length, width) or not torch.isfinite(value).all():
                raise ValueError(f'Invalid SOMA parameter: {key} {value.shape}')
            parameters[key] = value[0].contiguous()
        np.savez(out / PARAMETERS, **{key: value.numpy() for key, value in parameters.items()})
        del predictor, result, sequence
        gc.collect()
        torch.cuda.empty_cache()
        stages.begin('evaluate_soma')
        engine = MotionGeometry(str(MODELS), device='cuda', chunk_size=16)
        vertical_offset = evaluate(engine, parameters, out, torch, np)
    stages.begin('publish')
    if any(file_stat(path) != stamp for path, stamp in stamps.items()):
        raise ValueError('Input files changed during inference')
    atomic_json(out / 'project.json', dict(version=1, source=source, start=start, end=start + length / FPS,
                fps=FPS, frames=length, width=media['width'], height=media['height'], camera='static',
                stages={}, timeline=dict(start=start, step=1/FPS, frames=length),
                pipeline=dict(model='gemx', result='initial')))
    atomic_json(out / RECEIPT, dict(version=1, frames=length, fps=FPS, skeleton='soma-native77',
                virtual_root='omitted', units='metres', axes='native-y-up', transforms='joint-to-world',
                geometry='soma-mid-correctives', camera='assumed-static', vertical_offset_m=vertical_offset,
                runtime=dict(engine_revision=revision, versions=versions, entrypoint='tools/gemx_headless.py'),
                models={row['key']: row['expected_sha256'] for row in manifest},
                outputs={name: file_stat(out / name) for name in FILES}))
    stages.finish()
    report = dict(status='complete', frames=length, fps=FPS, seconds_per_stage=stages.times,
                  total_seconds=time.monotonic() - began, image_batch=batch,
                  peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
                  peak_vram_scope='PyTorch allocator only; excludes ONNX Runtime and Warp',
                  device=torch.cuda.get_device_name(), versions=versions, engine_revision=revision,
                  camera='assumed-static', outputs=list(FILES) + [RECEIPT],
                  parameter_shapes={key: list(value.shape) for key, value in parameters.items()})
    atomic_json(out / 'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='New output directory (never overwrites)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start on the normalized 30 Hz timeline')
    parser.add_argument('--frames', type=int, help='Frame count at 30 Hz; default: remaining video')
    parser.add_argument('--timeout', type=int, default=900, help='Whole-run wall time limit in seconds')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')

    def expired(signum, frame):
        raise TimeoutError(f'GEM-X exceeded {args.timeout} seconds')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout)
    try:
        run(args)
    except Exception as error:
        if hasattr(args, 'created_output'):
            report = dict(status='failed', error_type=type(error).__name__, error=str(error),
                          fps=FPS, source=str(args.video.resolve()),
                          frames=getattr(args, 'prepared_frames', None),
                          outputs_present=[name for name in FILES if (args.created_output / name).is_file()])
            if hasattr(args, 'stages'):
                failed_stage = args.stages.name
                args.stages.finish()
                torch = args.stages.torch
                report.update(failed_stage=failed_stage, seconds_per_stage=args.stages.times,
                              total_seconds=time.monotonic() - args.began,
                              peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(),
                              peak_vram_scope='PyTorch allocator only; excludes ONNX Runtime and Warp',
                              versions=args.versions)
            atomic_json(args.created_output / 'report.json', report)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
