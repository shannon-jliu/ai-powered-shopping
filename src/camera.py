"""
Camera Abstraction Module
Handles real-time video capture from GoPro Hero 8 (via GoPro Webcam virtual device)
or built-in FaceTime HD camera using macOS AVFoundation.
"""

import cv2
from typing import Optional, Tuple


class FreshFrameGrabber:
    """
    Direct, low-latency camera stream wrapper for macOS AVFoundation.
    Avoids multi-threaded CoreMediaIO/Metal race conditions that cause
    SIGSEGV segmentation faults on Apple Silicon.
    """

    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height

        print(f"[Camera] Opening video device index {device_index} (AVFoundation)...")
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device index {device_index}. "
                "Ensure the GoPro Webcam app is running or camera permissions are granted."
            )

        # Configure target resolution & minimal buffer size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warmup capture session
        for _ in range(3):
            self.cap.read()

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Connected! Resolution: {actual_w}x{actual_h}")

    def read(self) -> Tuple[bool, Optional[any]]:
        """Reads the latest frame from the video stream."""
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        """Releases the camera hardware."""
        print("[Camera] Releasing video device...")
        if self.cap.isOpened():
            self.cap.release()
        print("[Camera] Device released.")


def list_available_cameras(max_devices: int = 4):
    """Utility to test and print available camera indices."""
    print("=== Scanning Camera Devices ===")
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f" -> Device {i}: ACTIVE ({w}x{h})")
                available.append(i)
            cap.release()
    return available


if __name__ == "__main__":
    cams = list_available_cameras()
    if cams:
        print(f"Active cameras: {cams}")
        grabber = FreshFrameGrabber(device_index=cams[0])
        ret, frame = grabber.read()
        if ret:
            print(f"Successfully grabbed test frame: {frame.shape}")
        grabber.release()
    else:
        print("No active cameras found.")

