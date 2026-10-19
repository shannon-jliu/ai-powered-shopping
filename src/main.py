"""
AI-Powered Grocery Shopping Assistant
- Recipe NLP extraction
- Open-vocabulary YOLO-World detection on Apple Silicon MPS
- GoPro Hero 8 real-time video capture via AVFoundation
- Temporal voting checklist state tracking
- Live OpenCV HUD display
"""

import argparse
import os
import sys
import time
import subprocess
import cv2

from src.camera import FreshFrameGrabber, list_available_cameras
from src.recipe_parser import parse_recipe_text, load_recipe_file
from src.detector import YOLOWorldDetector
from src.tracker import ChecklistTracker
from src.ui import draw_hud


DEFAULT_RECIPE = """
1 can crushed tomatoes
1 red onion
1 head garlic
1 bunch fresh basil
1 cup mozzarella cheese
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Grocery Shopping Assistant (CS 4701 MVP)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 for GoPro Hero 8, 1 for FaceTime HD)"
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Path to recipe text file (default: built-in sample recipe)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-worldv2.pt",
        help="YOLO-World model variant (default: yolov8s-worldv2.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Confidence threshold for object detection (default: 0.12)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=3,
        help="Consecutive frames required to confirm an ingredient (default: 3)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and list all connected camera device indices without starting"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Device Scanning Mode
    if args.scan:
        list_available_cameras()
        return

    print("==================================================")
    print("      AI-Powered Grocery Shopping Assistant       ")
    print("==================================================")

    # 2. Parse Recipe
    if args.recipe and os.path.exists(args.recipe):
        print(f"[Main] Loading recipe from: {args.recipe}")
        ingredients = load_recipe_file(args.recipe)
    else:
        sample_path = os.path.join(os.path.dirname(__file__), "..", "recipes", "sample_pasta.txt")
        if os.path.exists(sample_path):
            print(f"[Main] Loading sample recipe from: {sample_path}")
            ingredients = load_recipe_file(sample_path)
        else:
            print("[Main] Parsing default built-in recipe...")
            ingredients = parse_recipe_text(DEFAULT_RECIPE)

    if not ingredients:
        print("[Error] No valid ingredients parsed. Exiting.")
        sys.exit(1)

    print("\n[Main] Target Shopping Checklist:")
    for idx, item in enumerate(ingredients, 1):
        print(f"  {idx}. {item.name} (from: '{item.raw_text}')")

    # 3. Initialize Temporal Tracker
    tracker = ChecklistTracker(ingredients, confirmation_threshold=args.frames)

    # 4. Expand Visual Prompts with Food Synonyms / Packaging Aliases
    all_prompts = []
    seen = set()
    for item in tracker.get_items():
        for alias in [item.name] + getattr(item, "aliases", []):
            if alias not in seen:
                all_prompts.append(alias)
                seen.add(alias)

    print(f"\n[Main] Setting {len(all_prompts)} visual recognition prompts (with aliases):")
    print(f"  {all_prompts}")

    # 5. Initialize YOLO-World Detector
    detector = YOLOWorldDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        imgsz=640
    )
    detector.set_classes(all_prompts)

    # 6. Connect Camera
    try:
        grabber = FreshFrameGrabber(device_index=args.camera, width=1920, height=1080)
    except Exception as e:
        print(f"\n[Camera Error] {e}")
        print("Tip: Run with '--camera 1' to try the built-in FaceTime camera.")
        sys.exit(1)

    print("\n[Main] Starting live shopping assistant loop.")
    print(f"[Main] Detection Sensitivity: conf={args.conf}, confirmation={args.frames} frames")
    print("[Main] Display window will open. Press 'q' in the window to exit.\n")

    prev_time = time.time()
    fps = 0.0

    window_name = "AI-Powered Grocery Shopping Assistant (CS 4701)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = grabber.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Calculate smoothed FPS
            curr_time = time.time()
            instant_fps = 1.0 / max(curr_time - prev_time, 1e-5)
            fps = 0.85 * fps + 0.15 * instant_fps
            prev_time = curr_time

            # Run open-vocabulary detection
            detections = detector.detect(frame)
            if detections:
                summary_str = ", ".join([f"{d['name']} ({d['confidence']:.2f})" for d in detections])
                print(f"[Vision] Detected: {summary_str}", flush=True)

            detected_names = [d["name"] for d in detections]

            # Update temporal voting state & shelf alerts
            newly_spotted = tracker.update(detected_names)
            if newly_spotted:
                # Trigger pleasant macOS system audio alert in background
                try:
                    subprocess.Popen(
                        ["afplay", "/System/Library/Sounds/Tink.aiff"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception:
                    pass

            # Draw HUD overlay
            display_frame = draw_hud(
                frame=frame,
                detections=detections,
                tracker=tracker,
                fps=fps,
                confirmation_threshold=args.frames
            )

            # Show interactive window
            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[Main] User requested exit.")
                break
            elif ord('1') <= key <= ord('9'):
                item_idx = key - ord('0')
                toggled = tracker.toggle_item(item_idx)
            elif key == ord(' ') or key == ord('c'):
                # Add currently spotted shelf items to cart
                confirmed = tracker.confirm_spotted_to_cart()
                if confirmed:
                    try:
                        subprocess.Popen(
                            ["afplay", "/System/Library/Sounds/Ping.aiff"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        grabber.release()
        cv2.destroyAllWindows()

        # Print final shopping session report
        summary = tracker.get_summary()
        print("\n==================================================")
        print("           Shopping Session Summary               ")
        print("==================================================")
        print(f"Total Ingredients: {summary['total']}")
        print(f"Found / In Cart:   {summary['found']}")
        print(f"Remaining:         {summary['remaining']}")
        print("--------------------------------------------------")
        for item in tracker.get_items():
            status = "[FOUND]" if item.found else "[MISSING]"
            print(f" {status:9} {item.name}")
        print("==================================================\n")


if __name__ == "__main__":
    main()

