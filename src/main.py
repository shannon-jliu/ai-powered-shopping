"""
AI-Powered Grocery Shopping Assistant - Main Entry Point
CS 4701 - Cornell University
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Grocery Shopping Assistant (CS 4701)"
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
        help="Path to recipe text file"
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
        help="Consecutive frames required to spot an ingredient (default: 3)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and list all connected camera device indices without starting"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("AI-Powered Grocery Shopping Assistant (CS 4701)")
    # Pipeline integration to be added
    pass


if __name__ == "__main__":
    main()
