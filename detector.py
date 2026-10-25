"""YOLO face detectors, found the same way the H3 Face Refine pack finds them (Impact Subpack's
`ultralytics_bbox` / `ultralytics` model folders)."""
from __future__ import annotations

import os

import folder_paths

_CACHE: dict = {}


def detector_list() -> list:
    names, seen, out = [], set(), []
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            names.extend(folder_paths.get_filename_list(key))
        except Exception:
            pass
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out or ["face_yolov8m.pt"]


def load_detector(name: str):
    if name in _CACHE:
        return _CACHE[name]
    path = None
    for key in ("ultralytics_bbox", "ultralytics"):
        try:
            path = folder_paths.get_full_path(key, name)
        except Exception:
            path = None
        if path:
            break
    if path is None:                                   # fall back to the standard models tree
        base = getattr(folder_paths, "models_dir", "models")
        for sub in ("ultralytics/bbox", "ultralytics", "ultralytics/segm"):
            cand = os.path.join(base, *sub.split("/"), name)
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        raise FileNotFoundError(f"Face detector '{name}' not found in the ultralytics_bbox / "
                                f"ultralytics model folders.")
    from ultralytics import YOLO
    model = YOLO(path)
    _CACHE[name] = model
    return model
