"""H3 Face Track Manual - hand-keyframed crop box, replaces `H3 Face Track + Crop`.

Takes a VIDEO (Load Video), converts it to the chosen frame rate (default 24), and outputs the
crops for H3, the transform for H3 Face Stitch Back / Per-Frame Denoise, and the converted
`images`, `audio` and `fps` so the rest of the graph works on exactly the same frames.
The keyframes are set in the graphic editor (button on the node).
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from .detector import detector_list
from .video_source import (content_sig, convert_active, forget as _vs_forget,
                           get as _vs_get,
                           clear_file, register_file, register_tensor, source_index, target_count)
from .video_source import _files as _vs_files


# --------------------------------------------------------------------------- helpers
def _ints(text: str) -> list[int]:
    return [int(v) for v in re.findall(r"\d+", text or "")]


def _ranges(text: str, B: int) -> list[tuple[int, int]]:
    out = []
    for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", text or ""):
        a, b = int(a), int(b)
        out.append((max(0, min(a, b)), min(B - 1, max(a, b))))
    rest = re.sub(r"\d+\s*-\s*\d+", " ", text or "")
    out += [(n, n) for n in _ints(rest) if 0 <= n < B]
    return out


class _NoKeyframes(ValueError):
    pass


def _editor_file(video, sig, n_frames, width, height, fps):
    """A file the editor can read the frames from, for any VIDEO. Nothing is saved when it can be avoided:
    1. a copy made earlier for this very video (same fingerprint) in ComfyUI's temp folder -> reused;
    2. the video already is a file (the standard VIDEO.get_stream_source() gives its path) -> used as is;
    3. otherwise it is written ONCE to the temp folder (stream bytes, or the standard VIDEO.save_to).
    Returns the path, or None if nothing worked (the caller then keeps the frames in memory)."""
    import folder_paths
    base = os.path.join(folder_paths.get_temp_directory(), "h3manual")
    path = os.path.join(base, re.sub(r"[^A-Za-z0-9_.-]", "_", sig) + ".mp4")

    def matches(p):
        try:
            idx = _vs_get(p)
            return idx.frames == n_frames and idx.width == width and idx.height == height
        except Exception:
            return False

    try:
        if os.path.isfile(path) and os.path.getsize(path) > 0 and matches(path):
            return path                                         # saved before for this same video
        src = None
        try:
            src = video.get_stream_source()                     # standard: a file path, or the video as bytes
        except Exception:
            pass
        if isinstance(src, str) and os.path.isfile(src):
            idx = _vs_get(src)
            if idx.frames == n_frames and idx.width == width and idx.height == height and abs(idx.fps - fps) < 1e-3:
                return src                                      # the video already is this very file
        os.makedirs(base, exist_ok=True)
        part = path + ".part.mp4"
        try:
            if hasattr(src, "getvalue"):                        # the stream of the video: no second encode
                with open(part, "wb") as fh:
                    fh.write(src.getvalue())
            else:
                video.save_to(part)                             # standard VIDEO API
            if not matches(part):
                raise RuntimeError("the saved copy does not match the video's frames")
            _vs_forget(part)
            os.replace(part, path)
        except Exception:
            _vs_forget(part)
            if os.path.exists(part):
                os.remove(part)
            raise
        olds = sorted((f for f in os.listdir(base) if f.endswith(".mp4") and os.path.join(base, f) != path),
                      key=lambda f: os.path.getmtime(os.path.join(base, f)), reverse=True)[2:]
        for f in olds:                                          # keep only the last few copies
            try:
                _vs_forget(os.path.join(base, f))
                os.remove(os.path.join(base, f))
            except OSError:
                pass
        return path
    except Exception as exc:
        print(f"[H3FaceTrackManual] could not make a file for the editor ({exc}); keeping the frames in memory instead")
        return None


def _notify(node_id, kind, text):
    """Tell the UI (a toast, and for kind='reset' also to clear the node's saved keyframes)."""
    print(f"[H3FaceTrackManual] {text}")
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("h3manual.notice",
                                        {"node": str(node_id), "kind": kind, "text": text})
    except Exception:
        pass


def _blocked(n_outputs):
    """Silently stop everything downstream of this node (no red error) when it has nothing to do."""
    try:
        from comfy_execution.graph import ExecutionBlocker
        return tuple(ExecutionBlocker(None) for _ in range(n_outputs))
    except Exception:
        return None


def _parse_keyframes(text: str, B: int, clamped: list) -> dict[int, tuple[float, float, float, float]]:
    kf = {}
    for n, line in enumerate((text or "").splitlines(), 1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) != 5:
            raise ValueError(f"keyframes line {n}: expected 'frame: x1 y1 x2 y2', got '{line}'")
        f = int(float(nums[0]))
        x1, y1, x2, y2 = (float(v) for v in nums[1:])
        if not 0 <= f < B:
            clamped.append(f)
            f = B - 1
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"keyframes line {n}: need x2 > x1 and y2 > y1")
        kf[f] = (x1, y1, x2, y2)
    if not kf:
        raise _NoKeyframes("keyframes is empty. Click 'Edit keyframes (graphic)' on the node, or write one line per keyframe:  frame: x1 y1 x2 y2")
    return kf


def _interp(fs: np.ndarray, vs: np.ndarray, x: np.ndarray, mode: str) -> np.ndarray:
    if len(fs) == 1:
        return np.full(len(x), vs[0], dtype=np.float64)
    xc = np.clip(x, fs[0], fs[-1])
    if mode == "hold":
        i = np.clip(np.searchsorted(fs, xc, side="right") - 1, 0, len(fs) - 1)
        return vs[i].astype(np.float64)
    if mode == "smooth" and len(fs) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator
            return np.asarray(PchipInterpolator(fs, vs)(xc), dtype=np.float64)
        except Exception:
            pass
    return np.interp(xc, fs, vs)


def _gauss(vals: np.ndarray, window: int) -> np.ndarray:
    window = int(min(window, len(vals)))
    if window % 2 == 0:
        window += 1
    if window < 3 or len(vals) < 3:
        return vals
    pad = window // 2
    x = np.arange(window, dtype=np.float64) - pad
    k = np.exp(-(x ** 2) / (2.0 * max(window / 6.0, 0.5) ** 2))
    k /= k.sum()
    return np.convolve(np.pad(vals, pad, mode="reflect"), k, mode="valid")[: len(vals)]


def _affine_crop(img: torch.Tensor, box, cw: int, ch: int) -> torch.Tensor:
    """Sub-pixel crop+resize in one bilinear sample (same maths as the tracker's)."""
    x, y, bw, bh = box
    _, H, W, _ = img.shape
    src = img[..., :3].movedim(-1, 1).float()
    theta = torch.tensor(
        [[[bw / W, 0.0, (2.0 * x + bw) / W - 1.0],
          [0.0, bh / H, (2.0 * y + bh) / H - 1.0]]],
        dtype=torch.float32, device=src.device)
    grid = F.affine_grid(theta, (1, 3, int(ch), int(cw)), align_corners=False)
    out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.movedim(1, -1).to(img.dtype)


def _rect(img: torch.Tensor, x0, y0, x1, y1, rgb, t=2):
    H, W = img.shape[:2]
    x0, x1 = int(round(max(0, min(W - 1, x0)))), int(round(max(0, min(W, x1))))
    y0, y1 = int(round(max(0, min(H - 1, y0)))), int(round(max(0, min(H, y1))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    c = torch.tensor(rgb, dtype=img.dtype)
    img[y0:min(y0 + t, y1), x0:x1] = c
    img[max(y1 - t, y0):y1, x0:x1] = c
    img[y0:y1, x0:min(x0 + t, x1)] = c
    img[y0:y1, max(x1 - t, x0):x1] = c


# --------------------------------------------------------------------------- node
class H3FaceTrackManual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip":
                    "The video to work on (Load Video, optionally followed by Trim Video, Crop Video "
                    "and similar). Its frames are converted to `fps`; the converted frames, the audio "
                    "and the fps come out again on `images`, `audio`, `fps`."}),
                "keyframes": ("STRING", {"multiline": True, "default":
                    "# frame: x1 y1 x2 y2   (CROP box, source-frame pixels, origin top-left)\n"
                    "# filled by the 'Edit keyframes (graphic)' button\n",
                    "tooltip":
                    "The crop box on chosen frames, one per line: 'frame: x1 y1 x2 y2' in source pixels. "
                    "Between keyframes the box is interpolated. Filled by the graphic editor - no need to "
                    "type it. The first lines (# video / # frames) record which video it was made for, so "
                    "a different video clears it automatically."}),
                "crop_factor": ("FLOAT", {"default": 2.5, "min": 1.2, "max": 8.0, "step": 0.1,
                    "tooltip":
                    "How large your crop is compared to the face: crop height = face height x crop_factor "
                    "(2.5 = the face fills about 40% of the crop). It does not change the crop, you draw "
                    "that. It tells the node how big the face is, which sets: the denoise strength curve "
                    "of H3 Per-Frame Denoise, the face area for the paste mask / SAM prompt, the head area "
                    "the editor's tracker follows, and how far 'YOLO Calibrate' enlarges a too small box."}),
                "canvas_mode": (["manual", "auto_capped_768", "auto_no_downscale"], {"default": "manual",
                    "tooltip":
                    "manual: use canvas_width x canvas_height. auto_*: ignore them and pick a square "
                    "canvas from the largest crop (auto_capped_768 never above 768, auto_no_downscale "
                    "never smaller than the crop)."}),
                "canvas_width": ("INT", {"default": 768, "min": 128, "max": 1344, "step": 32, "tooltip":
                    "Width in pixels of every crop sent to H3 (rounded to a multiple of 32). Also sets the "
                    "aspect ratio of the crop box you draw."}),
                "canvas_height": ("INT", {"default": 768, "min": 128, "max": 1344, "step": 32, "tooltip":
                    "Height in pixels of every crop sent to H3 (rounded to a multiple of 32). More pixels = "
                    "more detail, more time and VRAM."}),
                "interpolation": (["smooth", "linear", "hold"], {"default": "smooth", "tooltip":
                    "How the crop box moves between keyframes. smooth: gentle curve without overshoot. "
                    "linear: straight lines. hold: stays on a keyframe until the next one, then jumps."}),
                "smooth_window": ("INT", {"default": 9, "min": 1, "max": 201, "step": 2, "tooltip":
                    "Extra smoothing of the box's movement over this many frames (inside each shot only). "
                    "Removes the jitter of tracked keyframes. 1 = off: the box passes exactly through "
                    "your keyframes."}),
            },
            "optional": {
                "cut_frames": ("STRING", {"default": "", "tooltip":
                    "Frames where a new shot starts (hard cuts), e.g. '96, 210'. Interpolation, smoothing "
                    "and tracking never cross them. Set with the CUTS row of the editor."}),
                "absent_ranges": ("STRING", {"default": "", "tooltip":
                    "Frame ranges where the subject is not in the picture, e.g. '120-150, 300'. Nothing is "
                    "pasted back there (the original pixels stay). Set with the ABSENT row of the editor."}),
                "fade_frames": ("INT", {"default": 9, "min": 1, "max": 99, "step": 2, "tooltip":
                    "Only matters with absent ranges: over how many frames the pasted face fades in/out at "
                    "the edge of an absent range."}),
                "face_width_ratio": ("FLOAT", {"default": 0.8, "min": 0.4, "max": 1.2, "step": 0.05,
                    "tooltip":
                    "Face width divided by face height. Gives the width of the face area (paste mask, "
                    "SAM prompt) and of the head area the tracker follows; the height comes from "
                    "crop_factor."}),
                "grid_fit": (["pad_end", "trim_end", "off"], {"default": "pad_end", "tooltip":
                    "H3 only renders 5, 22, 39 ... (17k+5) frames. When the count is off that grid, H3 "
                    "pads the end of its latent with reference content, which can smear the last frames.\n"
                    "pad_end: add the missing frames at the end of `crops` (a repeat of the last crop); "
                    "Stitch Back never pastes them.\n"
                    "trim_end: render fewer frames (the largest grid count below); the tail keeps its "
                    "original pixels.\noff: do nothing."}),
                "absent_mode": (["keep", "drop"], {"default": "keep", "tooltip":
                    "keep: frames inside absent ranges still go through H3 (nothing is pasted).\n"
                    "drop: those frames are removed from `crops`, so H3 does not render them (as "
                    "absent_shots does in the original pack). Enough of them are kept as padding to reach "
                    "H3's 17k+5 length. Audio is not trimmed - see 'H3 Audio For Kept Frames'."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 240.0, "step": 0.001, "tooltip":
                    "Frame rate the video is converted to (default 24): frames are dropped or repeated, the "
                    "duration and the audio stay the same. Keyframes, crops and the outputs all use the "
                    "converted frames. 0 = keep the source frame rate."}),
                "detector": (detector_list(), {"tooltip":
                    "YOLO face model used by the editor's 'YOLO Calibrate' button (the same models as in "
                    "H3 Face Refine: Impact Subpack's ultralytics folders). Not used when the graph runs."}),
                "confidence": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05, "tooltip":
                    "Minimum score for 'YOLO Calibrate' to accept a face. Lower it if faces are missed."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "IMAGE", "STRING", "INT", "INT", "INT",
                    "IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("crops", "transform", "preview", "report", "canvas_w", "canvas_h",
                    "frame_count", "images", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "The crops for H3 (canvas_width x canvas_height each). -> H3 Inject Video Latent.",
        "Where every crop came from and where the face is in it. -> H3 Per-Frame Denoise, H3 Face Stitch Back.",
        "The video with the crop boxes drawn on it (for checking only).",
        "Text summary: frame-rate conversion, crop sizes, grid padding and any warnings.",
        "Width of the crops. -> the H3 node's width.",
        "Height of the crops. -> the H3 node's height.",
        "Number of crops. -> the H3 node's length.",
        "The original frames after the frame-rate conversion. -> H3 Face Stitch Back (base images).",
        "The audio of the video, unchanged.",
        "The frame rate of `images` (the fps widget, or the source rate when it is 0).",
    )
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = (
        "Crops a face out of a video for H3 using a box YOU place. Press 'Edit keyframes (graphic)': "
        "place the crop box on a frame, track it forwards/backwards, mark cuts and frames where the "
        "subject is absent, and optionally let 'YOLO Calibrate' enlarge boxes that are too small for the "
        "face. The node converts the video to `fps`, cuts the crops, and produces the `transform` that "
        "H3 Face Stitch Back / Per-Frame Denoise need. Replaces H3 Face Track + Crop.")

    def run(self, video, keyframes, crop_factor, canvas_mode, canvas_width, canvas_height,
            interpolation, smooth_window, cut_frames="", absent_ranges="", fade_frames=9,
            face_width_ratio=0.8,
            absent_mode="keep", grid_fit="pad_end", fps=24.0, detector=None, confidence=0.35,
            unique_id=None):
        comps = video.get_components()                          # works for ANY video: a file, Trim Video, Crop Video, ...
        raw = comps.images
        n_src, src_fps = int(raw.shape[0]), float(comps.frame_rate)
        fps_out = float(fps) if convert_active(src_fps, fps) else src_fps
        images = raw
        if convert_active(src_fps, fps):                       # bring the video to the chosen rate
            sel = torch.as_tensor([source_index(k, n_src, src_fps, fps) for k in range(target_count(n_src, src_fps, fps))],
                                  dtype=torch.long)
            images = raw[sel]
        audio = comps.audio if comps.audio is not None else {"waveform": torch.zeros((1, 1, 1)),
                                                              "sample_rate": 44100}
        B, H, W, _ = images.shape
        # Which video were these keyframes made for? The editor writes a comment line at the top of
        # `keyframes` ("# video: <id>@<fps>"). The id is a fingerprint of the frames themselves, so it does
        # not depend on how the video is stored or which node produced it.
        sig_base = content_sig(raw)                             # of the source frames: a frame-rate change must not change it
        if unique_id is not None:                              # what the editor holds for this node is stale now?
            ent = _vs_files.get(str(unique_id))
            if ent and ent["sig"] != sig_base:
                clear_file(unique_id)
        cur = f"{sig_base}@{fps_out:.3f}"                       # video + frame rate the numbers refer to
        preview_only = "H3MANUAL_PREVIEW_ONLY" in (keyframes or "")   # sent by the editor to load the frames
        m = re.search(r"^[ \t]*#[ \t]*video:[ \t]*(.+?)[ \t]*$", keyframes or "", re.M)
        saved_sig, _, saved_fps = (m.group(1) if m else "").rpartition("@")
        cur_sig, _, cur_fps = (cur or "").rpartition("@")
        fps_changed = False
        try:
            fps_changed = bool(m and cur and saved_sig == cur_sig and abs(float(saved_fps) - float(cur_fps)) > 0.0015)
        except ValueError:
            pass
        if m and cur and saved_sig != cur_sig:               # a different FILE: the coordinates do not fit
            _notify(unique_id, "reset", "New video detected: the old keyframes, cuts and absent ranges "
                                        "were cleared. Open 'Edit keyframes (graphic)' to set new ones.")
            keyframes, cut_frames, absent_ranges = "", "", ""
        if preview_only:                                         # the editor asked for the frames: hold them, stop here
            if unique_id is not None:                            # hand the editor a file to read (saved only if needed)
                f = _editor_file(video, sig_base, n_src, W, H, src_fps)
                if f:
                    register_file(unique_id, f, sig_base)
                else:
                    register_tensor(unique_id, images, fps_out, sig_base)
            one = images[:1, ..., :3]
            return (one, {}, one, "frames ready for the editor", 0, 0, 0, one, audio, fps_out)
        clamped = []
        try:
            kf = _parse_keyframes(keyframes, B, clamped)
        except _NoKeyframes:
            blocked = _blocked(len(self.RETURN_TYPES))
            if blocked is None:                              # old ComfyUI without ExecutionBlocker
                raise
            _notify(unique_id, "info", "No keyframes yet - the nodes after this one were skipped. "
                                       "Open 'Edit keyframes (graphic)' and set the crop box.")
            return blocked
        warn = []
        if clamped:
            warn.append(f"WARNING: {len(clamped)} keyframe(s) were beyond the last frame ({B - 1}) and were moved onto it")
        mf = re.search(r"^[ \t]*#[ \t]*frames:[ \t]*(\d+)", keyframes or "", re.M)
        if mf and int(mf.group(1)) != B:
            warn.append(f"WARNING: the editor counted {mf.group(1)} frames but this node received {B}; "
                        f"keyframes may sit a little off - reopen the editor and check the end of the clip")
        if fps_changed:
            warn.append(f"WARNING: the frame rate changed since the keyframes were set "
                        f"({saved_fps} -> {cur_fps}); reopen the editor to rescale them")
        for w in warn:
            _notify(unique_id, "info", w)
        cuts = sorted({c for c in _ints(cut_frames) if 0 < c < B})
        bounds = [0] + cuts + [B]
        shots = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        # per-frame face centre / height / width
        cx = np.zeros(B); cy = np.zeros(B); fh = np.zeros(B); fw = np.zeros(B)
        kf_all = sorted(kf)
        for s, e in shots:
            ks = [f for f in kf_all if s <= f < e]
            if not ks:  # no keyframe in this shot: borrow the nearest one, report it
                near = min(kf_all, key=lambda f: min(abs(f - s), abs(f - (e - 1))))
                ks = [near]
                src = {near: kf[near]}
            else:
                src = {f: kf[f] for f in ks}
            fs = np.array(ks, dtype=np.float64)
            arr = np.array([src[f] for f in ks], dtype=np.float64)  # x1 y1 x2 y2
            x = np.arange(s, e, dtype=np.float64)
            vals = {
                "cx": (arr[:, 0] + arr[:, 2]) / 2, "cy": (arr[:, 1] + arr[:, 3]) / 2,
                "fh": arr[:, 3] - arr[:, 1], "fw": arr[:, 2] - arr[:, 0]}
            for name, tgt in (("cx", cx), ("cy", cy), ("fh", fh), ("fw", fw)):
                v = _interp(fs, vals[name], x, interpolation)
                tgt[s:e] = _gauss(v, smooth_window) if smooth_window > 1 else v

        # The keyframes now describe the CROP. The face size (paste mask, denoise curve) is
        # derived from it: face height = crop height / crop_factor.
        fh = fh / crop_factor
        fw = fh * face_width_ratio

        valid = np.ones(B, dtype=bool)
        for a, b in _ranges(absent_ranges, B):
            valid[a:b + 1] = False
        weights = np.zeros(B)
        for s, e in shots:
            v = valid[s:e].astype(np.float64)
            weights[s:e] = np.clip(_gauss(v, fade_frames), 0, 1) if fade_frames > 1 else v

        # canvas
        if canvas_mode != "manual":
            need = float(min(fh.max() * crop_factor, H))
            snapped = int(np.ceil(need / 32.0) * 32)
            if canvas_mode == "auto_capped_768":
                snapped = min(snapped, 768)
            snapped = max(512, min(snapped, 1344))  # same 512 floor as the tracker
            canvas_width = canvas_height = snapped
        canvas_width = max(32, int(round(canvas_width / 32.0)) * 32)
        canvas_height = max(32, int(round(canvas_height / 32.0)) * 32)
        aspect = canvas_width / float(canvas_height)

        # crop height per frame = the height of the box you drew (any size, no snapping)
        crop_h = fh * crop_factor

        # Frames inside absent ranges whose paste has fully faded out ("dead"). With
        # absent_mode=drop they are removed from the batch like the original pack's
        # absent_shots does: H3 needs a 17k+5 frame count, so as many dead frames as it takes
        # stay in as padding. The first frames of a fade (weight > 0) are never dropped.
        dead = (~valid) & (weights <= 0.02)
        weights = np.where(dead, 0.0, weights)
        kept = np.arange(B)
        drop_note = ""
        if absent_mode == "drop" and dead.any():
            present = np.nonzero(~dead)[0]
            spare = np.nonzero(dead)[0]
            need = len(present)
            if need >= 5:
                while need % 17 != 5:
                    need += 1
                pad = need - len(present)
                if pad <= len(spare):
                    kept = np.array(sorted(present.tolist() + spare[:pad].tolist()), dtype=int)
                    drop_note = (f"absent_mode=drop: dropped {B - len(kept)} of {int(dead.sum())} absent "
                                 f"frame(s); {len(kept)} rendered"
                                 + (f" ({pad} absent kept to reach H3's 17k+5 grid)" if pad else ""))
                else:
                    drop_note = (f"absent_mode=drop: {int(dead.sum())} absent frame(s) kept - dropping "
                                 f"them cannot reach H3's 17k+5 grid")
            else:
                drop_note = "absent_mode=drop: subject absent from almost the whole video; nothing dropped"
        # ---- H3's 17k+5 frame-count grid ---------------------------------------------------
        grid_note, pad = "", 0
        K0 = len(kept)
        if grid_fit != "off" and K0 % 17 != 5:
            if grid_fit == "trim_end" and K0 > 5:
                K1 = K0 - ((K0 - 5) % 17)
                kept = kept[:K1]
                grid_note = (f"grid_fit=trim_end: {K0} -> {K1} frames (H3 grid 17k+5); the last "
                             f"{K0 - K1} frame(s) are not refined and keep their original pixels")
            else:
                nxt = 5 if K0 < 5 else K0 + ((5 - K0) % 17)
                pad = nxt - K0
                grid_note = (f"grid_fit=pad_end: {K0} frames is off H3's 17k+5 grid -> added {pad} "
                             f"frame(s) at the end of `crops` ({nxt} total). They repeat the last crop; "
                             f"Stitch Back ignores them")
        pos = {int(i): j for j, i in enumerate(kept)}

        boxes, face_rect = [], []
        crops = torch.zeros((len(kept) + pad, canvas_height, canvas_width, 3), dtype=images.dtype)
        preview = images[..., :3].clone()
        for i in range(B):
            bh = float(crop_h[i])
            bw = bh * aspect
            if bw > W:
                bw, bh = float(W), W / aspect
            if bh > H:
                bh, bw = float(H), H * aspect
            x = min(max(cx[i] - bw / 2.0, 0.0), max(0.0, W - bw))
            y = min(max(cy[i] - bh / 2.0, 0.0), max(0.0, H - bh))
            box = (float(x), float(y), float(bw), float(bh))
            boxes.append(box)
            if i in pos:
                crops[pos[i]:pos[i] + 1] = _affine_crop(images[i:i + 1], box, canvas_width,
                                                        canvas_height).to(crops.dtype)
            # face rect in CANVAS px, from the real face position inside the (maybe clamped) crop
            face_rect.append((
                (cx[i] - fw[i] / 2.0 - x) / bw * canvas_width,
                (cy[i] - fh[i] / 2.0 - y) / bh * canvas_height,
                fw[i] / bw * canvas_width,
                fh[i] / bh * canvas_height))
            col = (1.0, 0.0, 1.0) if i in kf else ((0.0, 1.0, 0.0) if valid[i] else (1.0, 0.0, 0.0))
            _rect(preview[i], x, y, x + bw, y + bh, col)
            _rect(preview[i], cx[i] - fw[i] / 2, cy[i] - fh[i] / 2,
                  cx[i] + fw[i] / 2, cy[i] + fh[i] / 2, (0.0, 1.0, 1.0), 1)

        for j in range(pad):                       # grid padding: repeat the last real crop
            crops[len(kept) + j] = crops[len(kept) - 1]

        kl = [int(k) for k in kept]
        # shot boundaries remapped into crop index space (crops are a subset of the source frames)
        segs_kept = []
        for a2, b2 in shots:
            inside = [pos[f] for f in range(a2, b2) if f in pos]
            if inside:
                segs_kept.append((min(inside), max(inside) + 1))
        if not segs_kept:
            segs_kept = [(0, len(kl))]
        # Same schema as H3 Face Track + Crop: every per-frame list is per KEPT frame, in crop
        # order; `source` says which source frame each crop came from.
        transform = {
            "boxes": [boxes[k] for k in kl],
            "canvas": (int(canvas_width), int(canvas_height)),
            "src_size": (int(W), int(H)),
            "frames": len(kl),
            "source": kl,
            "source_frames": int(B),
            "segments": [(int(a2), int(b2)) for a2, b2 in segs_kept],
            "absent": [bool(dead[k]) for k in kl],
            "weights": [float(weights[k]) for k in kl],
            "detected": [bool(valid[k]) for k in kl],
            "face_rect": [tuple(float(v) for v in face_rect[k]) for k in kl],
            "crop_factor": float(crop_factor),
        }
        mags = [canvas_height / b[3] for b in boxes]
        hs = [b[3] for b in boxes]
        snapnote = (f"crop height {min(hs):.0f}-{max(hs):.0f} px (source), canvas {canvas_width}x{canvas_height}\n")
        K = len(kept) + pad
        dropnote = ((drop_note + "\n") if drop_note else "") + ((grid_note + "\n") if grid_note else "")
        kmin, kmax = min(kf), max(kf)
        cover = (f"keyframes cover frames {kmin}..{kmax} of 0..{B - 1}"
                 + (" (frames outside hold the nearest keyframe)" if kmin > 0 or kmax < B - 1 else "")
                 + "\n")
        convnote = (f"fps: {src_fps:.3f} -> {fps_out:.3f}, {int(comps.images.shape[0])} -> {B} frames "
                    f"(frames dropped/repeated, duration and audio unchanged)\n"
                    if convert_active(src_fps, fps) else f"fps: {fps_out:.3f} (source rate kept), {B} frames\n")
        report = ("\n".join(warn) + ("\n" if warn else "") + convnote + dropnote + cover + snapnote + f"manual track: {len(kf)} keyframes, {len(shots)} shot(s), "
                  f"{int((~valid).sum())} absent frame(s)\n"
                  f"derived face height min={fh.min():.0f}px max={fh.max():.0f}px\n"
                  f"magnification into {canvas_width}x{canvas_height}: "
                  f"min={min(mags):.2f}x max={max(mags):.2f}x"
                  + ("\n!! some crops are downscaled (<1.0x): raise canvas or lower crop_factor"
                     if min(mags) < 1.0 else ""))
        print("[H3FaceTrackManual] " + report.replace("\n", "\n[H3FaceTrackManual] "))
        return (crops, transform, preview, report, int(canvas_width), int(canvas_height),
                int(K), images, audio, fps_out)


NODE_CLASS_MAPPINGS = {"H3FaceTrackManual": H3FaceTrackManual}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FaceTrackManual": "H3 Face Track Manual"}
