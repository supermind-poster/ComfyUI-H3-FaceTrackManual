"""On-demand access to the frames of a video file for the graphic editor.

Nothing is decoded up front and nothing is stored: opening a video only reads the container
index (a pass over the packets, no decoding), and frames are decoded when the editor asks for
them - sequentially when it plays, with a seek when it jumps. The frame numbering comes from the
presentation timestamps, i.e. exactly the order ComfyUI's own loader decodes them in.
"""
from __future__ import annotations

import io
import math
import os
import threading
import time

MAX_OPEN = 2
_lock = threading.Lock()
_open: dict = {}                      # path -> (file signature, index)


def file_sig(path: str) -> str:
    """Identity of a video file: name, size and modification time."""
    st = os.stat(path)
    return f"file:{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"


# ----------------------------------------------------------------------------- frame-rate conversion
# One definition used by the node and by the editor, so both see the same frames with the same
# numbers. Nearest-frame conversion (as ffmpeg's fps filter): frames are dropped or repeated, the
# duration stays the same, so the audio needs no change. fps <= 0 means "keep the source rate".
def convert_active(src_fps: float, tgt_fps: float) -> bool:
    return bool(tgt_fps) and tgt_fps > 0 and abs(float(src_fps) - float(tgt_fps)) >= 1e-3


def target_count(n_src: int, src_fps: float, tgt_fps: float) -> int:
    if not convert_active(src_fps, tgt_fps):
        return int(n_src)
    return max(1, int(math.floor(n_src * tgt_fps / src_fps + 0.5)))


def source_index(i: int, n_src: int, src_fps: float, tgt_fps: float) -> int:
    if not convert_active(src_fps, tgt_fps):
        return min(int(i), int(n_src) - 1)
    return max(0, min(int(n_src) - 1, int(math.floor(i * src_fps / tgt_fps + 0.5))))


class _AvIndex:
    def __init__(self, path):
        import av
        self.path = path
        self.lock = threading.RLock()
        pts = []
        with av.open(path) as c:                            # pass 1: packets only, no decoding
            s = c.streams.video[0]
            self.fps = float(s.average_rate or s.guessed_rate or 24.0)
            self.width = int(s.codec_context.width)
            self.height = int(s.codec_context.height)
            for pkt in c.demux(s):
                if pkt.pts is not None:
                    pts.append(int(pkt.pts))
        pts = sorted(pts)
        if not pts:
            raise RuntimeError("no video frames found")
        self.pts = pts
        self.frames = len(pts)
        self._c = None
        self._s = None
        self._gen = None
        self._last = -2
        self._last_rgb = None

    def _ensure(self):
        if self._c is None:
            import av
            self._c = av.open(self.path)
            self._s = self._c.streams.video[0]

    def close(self):
        with self.lock:
            self._gen = None
            if self._c is not None:
                try:
                    self._c.close()
                except Exception:
                    pass
                self._c = None

    def rgb(self, i):
        """Frame i as an RGB uint8 array (H, W, 3)."""
        with self.lock:
            i = max(0, min(self.frames - 1, int(i)))
            target = self.pts[i]
            if i == self._last and self._last_rgb is not None:         # the same frame again
                return self._last_rgb
            self._ensure()
            frame = None
            if self._gen is not None and 0 < i - self._last <= 64:     # moving forward: keep decoding
                for fr in self._gen:
                    if fr.pts is None or fr.pts >= target:
                        frame = fr
                        break
            if frame is None:                                          # jump: seek, then decode up to it
                self._c.seek(target, stream=self._s, backward=True)
                try:
                    self._s.codec_context.flush_buffers()
                except Exception:
                    pass
                self._gen = self._c.decode(self._s)
                for fr in self._gen:
                    if fr.pts is None or fr.pts >= target:
                        frame = fr
                        break
            if frame is None:
                self._gen, self._last, self._last_rgb = None, -2, None
                raise RuntimeError(f"could not decode frame {i}")
            self._last = i
            self._last_rgb = frame.to_ndarray(format="rgb24")
            return self._last_rgb


class _CvIndex:
    """Fallback when PyAV is missing: OpenCV (its frame count comes from metadata)."""

    def __init__(self, path):
        import cv2
        self.path = path
        self.lock = threading.RLock()
        self._cap = cv2.VideoCapture(path)
        self.frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 24.0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.frames <= 0:
            raise RuntimeError("OpenCV cannot read this video")
        self._last = -2

    def close(self):
        with self.lock:
            self._cap.release()

    def rgb(self, i):
        import cv2
        with self.lock:
            i = max(0, min(self.frames - 1, int(i)))
            if self._last + 1 != i:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = self._cap.read()
            self._last = i if ok else -2
            if not ok:
                raise RuntimeError(f"could not decode frame {i}")
            return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


class TensorKey(str):
    """Marks a source that is not a file but the frames a node actually received (after Trim Video,
    Crop Video, ... ): they are served from that node's own output tensor, nothing is copied."""


_files: dict = {}                                     # node id -> {"path", "sig", "time"}: the file a node's video is
MAX_TENSORS = 2                                       # frames held for the editor (one per open node)
_tensors: dict = {}                                   # node id -> {"t": uint8 frames, "fps", "sig", "time"}


def content_sig(images) -> str:
    """Fingerprint of a clip from its content: size, frame count and a coarse look at 24 frames spread
    over the clip (a shifted trim, a changed crop or another video all give a different one)."""
    import hashlib
    import torch.nn.functional as F
    B, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    n = min(24, B)
    idx = sorted({round(i * (B - 1) / max(1, n - 1)) for i in range(n)})
    x = images[idx][..., :3].float().mean(-1).unsqueeze(1)
    p = F.adaptive_avg_pool2d(x, (32, 32)).clamp(0, 1).mul(63).round().to(__import__("torch").uint8)
    return f"content:{B}x{H}x{W}:" + hashlib.md5(p.cpu().numpy().tobytes()).hexdigest()[:20]


def register_tensor(node_id, images, fps: float, sig: str):
    """Keep the frames a node received (after Trim / Crop Video ...) for the editor. They are held as
    8-bit pictures - ComfyUI frees its own copy right after the run - and released when the editor
    closes (release_tensor), so nothing is saved and nothing stays in memory afterwards."""
    import torch
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    u8 = torch.empty((n, h, w, 3), dtype=torch.uint8)
    for a in range(0, n, 32):
        u8[a:a + 32] = images[a:a + 32, ..., :3].float().clamp(0, 1).mul(255).round().to(torch.uint8)
    _tensors[str(node_id)] = {"t": u8, "fps": float(fps), "sig": sig, "time": time.time()}
    _files.pop(str(node_id), None)
    while len(_tensors) > MAX_TENSORS:
        del _tensors[min(_tensors, key=lambda k: _tensors[k]["time"])]


def release_tensor(node_id):
    _tensors.pop(str(node_id), None)


def register_file(node_id, path: str, sig: str):
    """The video a node received IS this file (a Load Video, or a Trim / Crop node that wrote one to
    temp). Only the path is remembered, so the editor can open it later without running anything."""
    _files[str(node_id)] = {"path": path, "sig": sig, "time": time.time()}
    _tensors.pop(str(node_id), None)


def clear_file(node_id):
    _files.pop(str(node_id), None)


def forget(path: str):
    """Drop the cached reader of a file (before it is replaced or deleted)."""
    with _lock:
        ent = _open.pop(path, None)
    if ent:
        ent[1].close()


def node_source(node_id):
    """(source, entry) for the video a node last received: the file when there is one, otherwise
    the frames held for the editor."""
    k = str(node_id)
    ent = _files.get(k)
    if ent and os.path.isfile(ent["path"]):
        return ent["path"], ent
    if k in _tensors:
        return tensor_key(k), _tensors[k]
    raise FileNotFoundError("this node has not received a video yet (or its file is gone)")


def tensor_key(node_id) -> TensorKey:
    return TensorKey(str(node_id))


class _TensorIndex:
    def __init__(self, t, fps):
        self.t, self.fps = t, float(fps)
        self.frames, self.height, self.width = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])

    def close(self):
        pass

    def rgb(self, i):
        i = max(0, min(self.frames - 1, int(i)))
        f = self.t[i, ..., :3]
        if f.dtype != __import__("torch").uint8:
            f = f.float().clamp(0, 1).mul(255).round().to(__import__("torch").uint8)
        return f.cpu().numpy()


def _tensor_entry(key):
    ent = _tensors.get(str(key))
    if ent is None:
        raise FileNotFoundError("the video of this node is not loaded (anymore): open the editor again")
    return ent, ent["t"]


def get(path):
    """The (cached) index of a video file (re-opened when the file changed), or the frames a node
    registered."""
    if isinstance(path, TensorKey):
        ent, t = _tensor_entry(path)
        return _TensorIndex(t, ent["fps"])
    sig = file_sig(path)
    with _lock:
        ent = _open.get(path)
        if ent and ent[0] == sig:
            return ent[1]
        if ent:
            ent[1].close()
            del _open[path]
        while len(_open) >= MAX_OPEN:
            k = next(iter(_open))
            _open.pop(k)[1].close()
    try:
        idx = _AvIndex(path)
    except ImportError:
        idx = _CvIndex(path)
    with _lock:
        _open[path] = (sig, idx)
    return idx


def _fps_for(src, fps):
    return 0.0 if isinstance(src, TensorKey) else fps


def frame_rgb(path, i: int, fps: float = 0.0):
    """Full-resolution RGB frame i (numbered after the frame-rate conversion)."""
    idx = get(path)
    fps = _fps_for(path, fps)
    return idx.rgb(source_index(i, idx.frames, idx.fps, fps))


def info(path, fps: float = 0.0) -> dict:
    idx = get(path)
    fps = _fps_for(path, fps)
    if isinstance(path, TensorKey):
        ent, _ = _tensor_entry(path)
        return {"frames": idx.frames, "fps": idx.fps, "src_fps": idx.fps, "src_frames": idx.frames,
                "src_width": idx.width, "src_height": idx.height, "sig": ent["sig"], "time": ent["time"]}
    eff = float(fps) if convert_active(idx.fps, fps) else float(idx.fps)
    return {"frames": target_count(idx.frames, idx.fps, fps), "fps": eff, "src_fps": idx.fps,
            "src_frames": idx.frames, "src_width": idx.width, "src_height": idx.height,
            "sig": file_sig(path)}


def frame_jpeg(path, i: int, max_side: int = 1280, fps: float = 0.0) -> bytes:
    from PIL import Image
    idx = get(path)
    fps = _fps_for(path, fps)
    im = Image.fromarray(idx.rgb(source_index(i, idx.frames, idx.fps, fps)))
    w, h = im.size
    k = min(1.0, max_side / float(max(w, h)))
    if k < 1.0:
        im = im.resize((max(1, int(w * k)), max(1, int(h * k))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def iter_bgr(path, a: int, b: int, max_side: int = 640, fps: float = 0.0):
    """Yield (index, downscaled BGR frame, scale = frame px per source px) for frames a..b
    (numbered after the frame-rate conversion)."""
    import cv2
    idx = get(path)
    fps = _fps_for(path, fps)
    n = target_count(idx.frames, idx.fps, fps)
    for i in range(max(0, a), min(b, n - 1) + 1):
        fr = cv2.cvtColor(idx.rgb(source_index(i, idx.frames, idx.fps, fps)), cv2.COLOR_RGB2BGR)
        h, w = fr.shape[:2]
        k = min(1.0, max_side / float(max(h, w)))
        if k < 1.0:
            fr = cv2.resize(fr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        yield i, fr, fr.shape[1] / float(idx.width)
