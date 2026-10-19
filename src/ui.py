"""
User Interface and HUD Overlay Module
Renders bounding boxes, alert banners, and interactive checklist HUD.
"""

import numpy as np
from typing import List, Dict


def draw_hud(
    frame: np.ndarray,
    detections: List[Dict],
    tracker: any,
    fps: float,
    confirmation_threshold: int = 3
) -> np.ndarray:
    """Renders HUD overlay over the frame."""
    pass
