from .track_manual import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _An
from .drop_helpers import NODE_CLASS_MAPPINGS as _B, NODE_DISPLAY_NAME_MAPPINGS as _Bn
from . import routes  # noqa: F401  (registers /h3manual/* endpoints)

NODE_CLASS_MAPPINGS = {**_A, **_B}
NODE_DISPLAY_NAME_MAPPINGS = {**_An, **_Bn}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
