"""Optional helper for `absent_mode = drop`.

The stock H3 Face Stitch Back and H3 Per-Frame Denoise already understand dropped frames
(through `transform["source"]` / `["absent"]`), so nothing else is needed for the picture.

Audio is the one thing the original pack does not touch: when frames are dropped the video
H3 renders is shorter than the soundtrack, and everything after the first dropped frame
would be lip-synced to the wrong audio. This node cuts the audio down to the frames that
are rendered.
"""
from __future__ import annotations

import torch


class H3AudioForKeptFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "transform": ("H3FACEXFORM", {"tooltip": "`transform` of the track node."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                  "tooltip": "Frame rate of the source video (the `fps` output of "
                                             "H3 Load Video + Face Select, or your loader)."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3/Face Refine"
    DESCRIPTION = ("Keep only the audio that belongs to the frames H3 renders, so lipsync stays "
                   "aligned after absent frames were dropped. Feed THIS to "
                   "MiniMaxH3NativeAudioLock; the original full audio still goes to the video "
                   "save node.")

    def run(self, audio, transform, fps):
        src = transform.get("source")
        if src:                                    # grid-padding frames repeat the last source frame
            src = [k for j, k in enumerate(src) if j == 0 or k != src[j - 1]]
        total = int(transform.get("source_frames", 0))
        if not src or len(src) == total:           # nothing was dropped
            return (audio,)
        wf, sr = audio["waveform"], int(audio["sample_rate"])
        T = wf.shape[-1]
        runs, a, prev = [], src[0], src[0]
        for k in src[1:]:
            if k != prev + 1:
                runs.append((a, prev)); a = k
            prev = k
        runs.append((a, prev))
        segs = []
        for a, b in runs:
            s0 = int(round(a / fps * sr)); s1 = min(T, int(round((b + 1) / fps * sr)))
            if s1 > s0:
                segs.append(wf[..., s0:s1])
        if not segs:
            raise ValueError("audio is shorter than the first kept frame; wrong fps or audio?")
        out = torch.cat(segs, dim=-1)
        print(f"[H3FaceTrackManual] audio: {len(runs)} kept run(s), {out.shape[-1] / sr:.2f}s "
              f"of {T / sr:.2f}s")
        return ({"waveform": out, "sample_rate": sr},)


class H3FaceTrackManualProbe:
    """Not meant to be placed by hand. When the editor needs the frames that come out of Trim Video /
    Crop Video / ..., it runs a tiny prompt made of those nodes, H3 Face Track Manual and this node.
    ComfyUI only runs a prompt that ends in an output node; this one is that end, so H3 Face Track
    Manual itself does not have to be an output node. It does nothing and saves nothing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    DEV_ONLY = True                                   # keeps it out of the node search
    CATEGORY = "MiniMax H3/Face Refine"

    def run(self, source):
        return ()


NODE_CLASS_MAPPINGS = {"H3AudioForKeptFrames": H3AudioForKeptFrames,
                       "H3FaceTrackManualProbe": H3FaceTrackManualProbe}
NODE_DISPLAY_NAME_MAPPINGS = {"H3AudioForKeptFrames": "H3 Audio For Kept Frames",
                              "H3FaceTrackManualProbe": "H3 Face Track Manual (editor helper)"}
