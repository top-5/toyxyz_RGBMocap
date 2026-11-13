"""CUDA-only adapter for the pinned HumanArt ONNX detector's NMS boundary.

Keep original protobuf fields/weights byte-for-byte, prune only non-ancestors
of its decoded boxes/scores, and replace its NMS/TopK tail with CUDA Torch.
This small wire reader avoids adding the absent `onnx` package to seed.
Only RGB MoCap's SHA-verified HumanArt graph is supported.
"""

from pathlib import Path
import struct


def _varint(data, index):
    value = shift = 0
    while shift < 70:
        byte = data[index]
        index += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, index
        shift += 7
    raise ValueError('Invalid protobuf varint')


def _fields(data):
    index = 0
    while index < len(data):
        start = index
        tag, index = _varint(data, index)
        number, wire = tag >> 3, tag & 7
        if wire == 2:
            size, index = _varint(data, index)
            value = data[index:index + size]
            index += size
        elif wire == 0:
            value, index = _varint(data, index)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[index:index + size]
            index += size
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        if index > len(data):
            raise ValueError('Truncated protobuf field')
        yield number, wire, value, data[start:index]


def _encode(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _item(number, value):
    return _encode((number << 3) | 2) + _encode(len(value)) + value


def derive_prenms(source, destination):
    """Write a derived graph under the run directory, never edit source assets."""
    model = list(_fields(memoryview(Path(source).read_bytes())))
    graph = next(value for number, _, value, _ in model if number == 7)
    graph_fields = list(_fields(graph))
    nodes, constants = [], {}
    for number, _, value, raw in graph_fields:
        if number == 1:
            props = {1: [], 2: []}
            for key, _, entry, _ in _fields(value):
                if key in props:
                    props[key].append(bytes(entry).decode())
            nodes.append((props, raw))
        elif number == 5:
            entries = list(_fields(value))
            name = next((bytes(x).decode() for k, _, x, _ in entries if k == 8), '')
            if name in ('1933', '1934', '1935', '1996'):
                constants[name] = bytes(next(x for k, _, x, _ in entries if k == 9))
    expected = {'1933': struct.pack('<q', 200), '1934': struct.pack('<f', .65),
                '1935': struct.pack('<f', .01), '1996': struct.pack('<q', 100)}
    if len(nodes) != 573 or constants != expected:
        raise ValueError('Unsupported YOLOX graph/NMS constants; use the pinned model')
    required, kept = {'1952', '1955'}, []
    for props, raw in reversed(nodes):
        if required.intersection(props[2]):
            kept.append(raw)
            required.update(props[1])
    if len(kept) != 548:
        raise ValueError('Unexpected pre-NMS graph ancestry')
    # ValueInfoProto: name=1, type=2; TypeProto tensor_type=1,
    # TensorTypeProto elem_type=1 (FLOAT). ORT infers output dimensions.
    outputs = [_item(1, name.encode()) + _item(2, _item(1, b'\x08\x01'))
               for name in ('1952', '1955')]
    new_graph = (b''.join(reversed(kept))
                 + b''.join(bytes(raw) for number, _, _, raw in graph_fields if number not in (1, 12))
                 + b''.join(_item(12, value) for value in outputs))
    Path(destination).write_bytes(b''.join(_item(7, new_graph) if number == 7 else bytes(raw)
                                         for number, _, _, raw in model))


def detector_class(original, derived):
    """Retain upstream crops/postprocessing; replace graph execution/NMS only."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from torchvision.ops import nms

    class CudaDetector(original):
        def __init__(self, onnx_path, input_size=(640, 640), score_thr=.5,
                     nms_thr=.45, device='cuda'):
            if device != 'cuda' or 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('YOLOX requires ONNX Runtime CUDA')
            derive_prenms(onnx_path, derived)
            options = ort.SessionOptions()
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            self.sess = ort.InferenceSession(str(derived), sess_options=options,
                                            providers=['CUDAExecutionProvider'])
            self.sess.disable_fallback()
            self.execution_providers = self.sess.get_providers()
            # ORT registers a CPU EP even when every node must use CUDA.
            # The strict session flag rejects initialization if ANY node needs it.
            if 'CUDAExecutionProvider' not in self.execution_providers:
                raise RuntimeError('CUDA provider did not activate')
            self.input_size, self.score_thr, self.nms_thr = input_size, score_thr, nms_thr
            shapes = [v.shape for v in self.sess.get_outputs()]
            if shapes != [[1, 5000, 4], [1, 80, 5000]]:
                raise ValueError(f'Unexpected decoded YOLOX outputs {shapes}')
            self.buffers = [torch.empty(shape, device='cuda', dtype=torch.float32) for shape in shapes]

        def detect(self, img_bgr):
            padded, ratio = self._preprocess(img_bgr)
            image = torch.from_numpy(padded.transpose(2, 0, 1)[None].astype(np.float32)).to('cuda')
            binding = self.sess.io_binding()
            binding.bind_input(self.sess.get_inputs()[0].name, 'cuda', 0, np.float32,
                               tuple(image.shape), image.data_ptr())
            for output, tensor in zip(self.sess.get_outputs(), self.buffers):
                binding.bind_output(output.name, 'cuda', 0, np.float32,
                                    tuple(tensor.shape), tensor.data_ptr())
            torch.cuda.synchronize()
            self.sess.run_with_iobinding(binding)
            binding.synchronize_outputs()
            boxes, scores = self.buffers[0][0], self.buffers[1][0]
            selections = []
            # Preserve the graph's per-class NMS (.01 score, .65 IoU,
            # max 200/class) and final global top 100; original YOLOX adapter
            # discards labels and applies its .5 score filter afterwards.
            for category in torch.nonzero((scores > .01).any(dim=1)).flatten().tolist():
                indices = torch.nonzero(scores[category] > .01).flatten()
                keep = nms(boxes[indices], scores[category, indices], .65)[:200]
                indices = indices[keep]
                selections.append(torch.cat((boxes[indices], scores[category, indices, None]), dim=1))
            if not selections:
                return np.empty((0, 4), np.float32), np.empty(0, np.float32)
            detections = torch.cat(selections)
            detections = detections[detections[:, 4].argsort(descending=True)[:100]]
            return self._postprocess(detections[None].cpu().numpy(), ratio)

    return CudaDetector
