"""
User Interface and HUD Overlay Module
Renders bounding boxes, target labels, and a semi-transparent
interactive shopping checklist directly onto the OpenCV video frame.
"""

import cv2
import numpy as np
from typing import List, Dict
from src.tracker import ChecklistTracker, ChecklistItem


def draw_hud(
    frame: np.ndarray,
    detections: List[Dict],
    tracker: ChecklistTracker,
    fps: float,
    confirmation_threshold: int = 5
) -> np.ndarray:
    """
    Renders bounding boxes and an on-screen shopping checklist HUD over the frame.
    """
    h, w = frame.shape[:2]
    items = tracker.get_items()
    summary = tracker.get_summary()

    # 1. Draw Bounding Boxes for detected objects
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det["name"].lower()
        conf = det["confidence"]

        # Check status from tracker
        is_in_cart = False
        is_spotted = False
        consecutive = 0
        for item in items:
            if item.name in name or name in item.name or any(a in name or name in a for a in getattr(item, "aliases", [])):
                is_in_cart = item.found
                is_spotted = item.in_view
                consecutive = item.consecutive_frames
                break

        if is_in_cart:
            box_color = (0, 230, 0)       # Bright Green
            status_tag = "[IN CART]"
        elif is_spotted:
            box_color = (0, 215, 255)     # Glowing Gold
            status_tag = "[SPOTTED]"
        else:
            box_color = (255, 220, 50)    # Tentative Cyan
            status_tag = f"[{consecutive}/{confirmation_threshold}]"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Label pill
        label = f"{det['name']} {status_tag} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        y_text = max(y1 - 6, th + 6)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 4), box_color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text - 2), font, font_scale, (0, 0, 0), thickness)

    # 2. Draw Top Shelf Alert Banner if an item is spotted
    spotted_items = [item for item in items if item.in_view and not item.found]
    if spotted_items:
        banner_text = f"⚡ SPOTTED: {spotted_items[0].name.upper()}! Press SPACE or #{items.index(spotted_items[0])+1} to add to cart"
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx = (w - bw) // 2
        by = 35
        # Background alert pill
        cv2.rectangle(frame, (bx - 15, by - bh - 8), (bx + bw + 15, by + 8), (0, 180, 255), -1)
        cv2.rectangle(frame, (bx - 15, by - bh - 8), (bx + bw + 15, by + 8), (255, 255, 255), 2)
        cv2.putText(frame, banner_text, (bx, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    # 3. Draw Semi-Transparent Sidebar for Checklist HUD
    sidebar_w = 430
    sidebar_h = min(h - 20, 110 + 42 * len(items))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + sidebar_w, 15 + sidebar_h), (25, 25, 25), -1)
    alpha = 0.78
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + sidebar_w, 15 + sidebar_h), (80, 80, 80), 1)

    # Title & Stats
    cv2.putText(frame, "AI GROCERY ASSISTANT", (30, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 235, 255), 2)
    
    progress_text = f"In Cart: {summary['found']} / {summary['total']}  |  On Shelf: {summary.get('in_view', 0)}"
    cv2.putText(frame, progress_text, (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    # Checklist Items with Keyboard Shortcut Numbers (1-9)
    y_cursor = 110
    for idx, item in enumerate(items, 1):
        if item.found:
            box_sym = f"[{idx}] [X]"
            sym_color = (0, 255, 0)         # Green
            text_color = (180, 255, 180)
            status_suffix = " (In Cart)"
        elif item.in_view:
            box_sym = f"[{idx}] [⚡]"
            sym_color = (0, 215, 255)       # Gold
            text_color = (100, 230, 255)
            status_suffix = " [SPOTTED!]"
        else:
            box_sym = f"[{idx}] [ ]"
            sym_color = (150, 150, 150)
            text_color = (220, 220, 220)
            status_suffix = ""

        display_name = item.name[:22]
        cv2.putText(frame, box_sym, (30, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sym_color, 2)
        cv2.putText(frame, f"  {display_name}{status_suffix}", (88, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)
        y_cursor += 36

    # 4. Top-Right Telemetry & FPS Pill
    telemetry_w = 210
    telemetry_h = 42
    tx = w - telemetry_w - 15
    ty = 15

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (tx, ty), (tx + telemetry_w, ty + telemetry_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay2, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (tx, ty), (tx + telemetry_w, ty + telemetry_h), (80, 80, 80), 1)

    fps_color = (0, 255, 0) if fps >= 20.0 else (0, 180, 255)
    cv2.putText(frame, f"FPS: {fps:.1f} (M3 MPS)", (tx + 12, ty + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

    # 5. Interactive Controls Footer
    footer_text = "Press [1-9] to Toggle/Uncheck  |  SPACE to Add Spotted  |  'q' to Exit"
    cv2.rectangle(frame, (15, h - 35), (w - 15, h - 8), (20, 20, 20), -1)
    cv2.putText(frame, footer_text, (30, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 1)

    return frame

