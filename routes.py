"""Endpoints for the graphic keyframe editor. The editor reads the video file chosen in the
Load Video node directly; frames are decoded on demand (see video_source.py)."""
import asyncio
import os

from aiohttp import web
from server import PromptServer

from . import video_source

routes = PromptServer.instance.routes
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".gif")


def _path(name):
    """A file name as Load Video shows it (annotated names like 'clip.mp4 [input]' work too)."""
    import folder_paths
    v = str(name or "").strip().strip('"').strip("'")
    if not v:
        raise ValueError("no video file given")
    p = folder_paths.get_annotated_filepath(v)
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(f"video not found: {v}")
    if not p.lower().endswith(_VIDEO_EXTS):
        raise ValueError("not a video file")
    return p


def _fps(request_or_dict):
    try:
        q = request_or_dict.query if hasattr(request_or_dict, "query") else request_or_dict
        return float(q.get("fps", 0) or 0)
    except Exception:
        return 0.0


def _src(q):
    """Where the frames come from: `node` = the frames that node received (Trim/Crop Video in
    front of it), otherwise `name` = the file chosen in Load Video."""
    node = q.get("node") if hasattr(q, "get") else None
    if node:
        return video_source.tensor_key(node)
    return _path(q.get("name"))


async def _run(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


# ----------------------------------------------------------------------------- video + frames
@routes.get("/h3manual/vinfo")
async def h3manual_vinfo(request):
    try:
        return web.json_response(await _run(video_source.info, _src(request.query), _fps(request)))
    except Exception as e:
        return web.Response(status=400, text=str(e))


@routes.get("/h3manual/vframe")
async def h3manual_vframe(request):
    try:
        path = _src(request.query)
        i = int(request.query.get("index", "0"))
        mx = max(256, min(4096, int(request.query.get("max", "1280"))))
        body = await _run(video_source.frame_jpeg, path, i, mx, _fps(request))
        return web.Response(body=body, content_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as e:
        return web.Response(status=400, text=str(e))


# ----------------------------------------------------------------------------- tracking
_TRACKERS = {"csrt": "TrackerCSRT_create", "kcf": "TrackerKCF_create", "mil": "TrackerMIL_create"}
_FALLBACK = {"csrt": ["csrt", "kcf", "mil"], "kcf": ["kcf", "mil"], "mil": ["mil"]}


def _make_tracker(name):
    import cv2
    for n in _FALLBACK.get(name, ["mil"]):
        for ns in (cv2, getattr(cv2, "legacy", None)):
            fn = getattr(ns, _TRACKERS[n], None) if ns is not None else None
            if fn is not None:
                return fn(), n
    raise RuntimeError("no OpenCV tracker available (need opencv-python >= 4.5.1)")


class _ScaleEstimator:
    """Relative size of the tracked head, from the way sparse feature points spread around their
    centre (Lucas-Kanade). OpenCV's KCF/MIL trackers never change their box size and CSRT hardly
    does, so the size comes from here whichever tracker is used."""

    def __init__(self):
        self.s, self.pts, self.prev, self.age = 1.0, None, None, 0

    def _seed(self, gray, box):
        import cv2
        import numpy as np
        x1, y1, x2, y2 = [int(v) for v in box]
        mask = np.zeros_like(gray)
        mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 255
        self.pts = cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=4, mask=mask)
        self.age = 0

    def start(self, gray, box):
        self.prev = gray
        self._seed(gray, box)

    def update(self, gray, box):
        import cv2
        import numpy as np
        ratio = 1.0
        if self.pts is not None and len(self.pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.reshape(-1) == 1
            p0, p1 = self.pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)
            if len(p0) >= 8:
                d0 = np.linalg.norm(p0 - p0.mean(0), axis=1)
                d1 = np.linalg.norm(p1 - p1.mean(0), axis=1)
                m = d0 > 1.0
                if m.sum() >= 6:
                    ratio = float(np.median(d1[m] / d0[m]))
                self.pts = p1.reshape(-1, 1, 2).astype(np.float32)
            else:
                self.pts = None
        self.s *= min(1.05, max(0.95, ratio))
        self.prev = gray
        self.age += 1
        if self.pts is None or len(self.pts) < 15 or self.age >= 25:     # re-seed on the tracked head
            self._seed(gray, box)
        return self.s


def _follow(frames, box, name):
    """Track `box` (source px x1,y1,x2,y2) through `frames` (first one = start frame).
    Returns {frame: [x1, y1, x2, y2, scale]} with scale relative to the start frame."""
    import cv2
    out, lost, used, trk, first, est = {}, None, name, None, True, _ScaleEstimator()
    for i, fr, sc in frames:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if first:
            trk, used = _make_tracker(name)
            x1, y1, x2, y2 = [v * sc for v in box]
            trk.init(fr, (int(round(x1)), int(round(y1)), max(4, int(round(x2 - x1))),
                          max(4, int(round(y2 - y1)))))
            est.start(gray, (x1, y1, x2, y2))
            first = False
            continue
        ok, bb = trk.update(fr)
        if not ok or bb[2] < 4 or bb[3] < 4:
            lost = i
            break
        s = est.update(gray, (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]))
        out[i] = [bb[0] / sc, bb[1] / sc, (bb[0] + bb[2]) / sc, (bb[1] + bb[3]) / sc, s]
    if first:
        raise RuntimeError("could not read the start frame")
    return out, lost, used


def _track(path, start, box, lo, hi, direction, name, max_frames, fps=0.0):
    lo = max(lo, start - max_frames)
    hi = min(hi, start + max_frames)
    boxes, lost, used = {}, {"fwd": None, "back": None}, name
    if direction in ("fwd", "both") and hi > start:
        b, l, used = _follow(video_source.iter_bgr(path, start, hi, fps=fps), box, name)
        boxes.update(b); lost["fwd"] = l
    if direction in ("back", "both") and lo < start:
        frames = list(video_source.iter_bgr(path, lo, start, fps=fps))[::-1]     # start frame first, then back
        b, l, used = _follow(frames, box, name)
        boxes.update(b); lost["back"] = l
    return {"tracker": used, "boxes": {str(k): v for k, v in boxes.items()}, "lost": lost}


@routes.post("/h3manual/track")
async def h3manual_track(request):
    try:
        d = await request.json()
        res = await _run(_track, _src(d), int(d["start"]), [float(v) for v in d["box"]],
                         int(d["lo"]), int(d["hi"]), d.get("direction", "both"),
                         d.get("tracker", "csrt"), int(d.get("max_frames", 600)), _fps(d))
        return web.json_response(res)
    except Exception as e:
        return web.Response(status=400, text=str(e))


# ----------------------------------------------------------------------------- cuts
def _cuts_scenedetect(path, threshold, n, fps=0.0):
    """PySceneDetect fed with the decoded frames (same numbering as everywhere else)."""
    from scenedetect import AdaptiveDetector, FrameTimecode
    det = AdaptiveDetector(adaptive_threshold=threshold)
    fps, cuts, use_tc, last = 24.0, set(), True, 0

    def add(found):
        for c in found or []:
            cuts.add(int(getattr(c, "frame_num", c)))

    for i, fr, _ in video_source.iter_bgr(path, 0, n - 1, fps=fps):
        last = i
        if use_tc:
            try:
                add(det.process_frame(FrameTimecode(i, fps), fr))
                continue
            except TypeError:
                use_tc = False                   # older PySceneDetect: plain frame numbers
        add(det.process_frame(i, fr))
    try:                                         # flush the detector's look-ahead window
        add(det.post_process(FrameTimecode(last, fps) if use_tc else last))
    except Exception:
        pass
    return sorted(c for c in cuts if 0 < c < n)


def _cuts_histogram(path, threshold, n, fps=0.0):
    """Fallback: HSV-histogram jumps, relative to the recent frame-to-frame change."""
    import cv2
    import numpy as np
    prev, diffs, cuts = None, [], []
    for i, fr, _ in video_source.iter_bgr(path, 0, n - 1, max_side=320, fps=fps):
        small = cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA)
        h = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1, 2], None,
                         [12, 6, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
        if prev is not None:
            d = cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA)
            base = float(np.mean(diffs[-12:])) if diffs else 0.0
            if d > 0.35 and d > max(threshold, 1.5) * max(base, 0.02) and (not cuts or i - cuts[-1] > 3):
                cuts.append(i)
            diffs.append(d)
        prev = h
    return cuts


def _cuts(path, threshold, fps=0.0):
    idx = video_source.get(path)
    n = video_source.target_count(idx.frames, idx.fps, video_source._fps_for(path, fps))
    try:
        return {"cuts": _cuts_scenedetect(path, threshold, n, fps), "method": "PySceneDetect (adaptive)"}
    except Exception as exc:                     # not installed or incompatible: built-in detector
        print(f"[H3FaceTrackManual] PySceneDetect unavailable ({exc}); using the histogram detector")
        return {"cuts": _cuts_histogram(path, threshold, n, fps), "method": "histogram detector"}


@routes.get("/h3manual/cuts")
async def h3manual_cuts(request):
    try:
        res = await _run(_cuts, _src(request.query), float(request.query.get("threshold", "3.0")), _fps(request))
        return web.json_response(res)
    except Exception as e:
        return web.Response(status=400, text=str(e))


# ----------------------------------------------------------------------------- YOLO calibration
def _calibrate(src, fps, detector, confidence, items):
    """For each keyframe box, find the face YOLO sees inside it. Returns the face box (source px)
    or None. Only the area around the box is searched, so other people nearby are mostly ignored."""
    import cv2
    from .detector import load_detector
    model = load_detector(detector)
    out = []
    for it in items:
        f, box = int(it["frame"]), [float(v) for v in it["box"]]
        rgb = video_source.frame_rgb(src, f, fps)
        H, W = rgb.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rw, rh = (box[2] - box[0]) * 1.6, (box[3] - box[1]) * 1.6
        rx0, ry0 = max(0, int(cx - rw / 2)), max(0, int(cy - rh / 2))
        rx1, ry1 = min(W, int(cx + rw / 2)), min(H, int(cy + rh / 2))
        region = cv2.cvtColor(rgb[ry0:ry1, rx0:rx1], cv2.COLOR_RGB2BGR)
        face, best = None, 0.0
        res = model.predict(region, conf=float(confidence), verbose=False)[0]
        if len(res.boxes):
            confs = res.boxes.conf.tolist() if getattr(res.boxes, "conf", None) is not None else None
            for k, q in enumerate(res.boxes.xyxy.tolist()):
                fx1, fy1, fx2, fy2 = q[0] + rx0, q[1] + ry0, q[2] + rx0, q[3] + ry0
                mx, my = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                if not (box[0] <= mx <= box[2] and box[1] <= my <= box[3]):
                    continue                                   # not inside the box you drew
                area = (fx2 - fx1) * (fy2 - fy1)
                if area > best:                                # the biggest face in the box
                    best, face = area, [fx1, fy1, fx2, fy2, float(confs[k]) if confs else 1.0]
        out.append({"frame": f, "face": face})
    return out


@routes.post("/h3manual/calibrate")
async def h3manual_calibrate(request):
    try:
        d = await request.json()
        res = await _run(_calibrate, _src(d), _fps(d), str(d.get("detector", "")),
                         float(d.get("confidence", 0.35)), d.get("items", []))
        return web.json_response({"results": res})
    except Exception as e:
        return web.Response(status=400, text=str(e))
