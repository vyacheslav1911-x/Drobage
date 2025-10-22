#!/usr/bin/env python3
import sys
import os
import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import torch

# ---------------------------
# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------
# Load DepthAnything model
depth_model = DepthAnythingV2(encoder='vitl')
depth_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location=device))
depth_model.to(device)
depth_model.eval()

# ---------------------------
# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# ---------------------------
# DepthAI setup
jsonfile = "/Users/ivan/depthai/resources/184430101153051300_09_28_25_13_00.json"
calibData = dai.CalibrationHandler(jsonfile)

# Create pipeline
with dai.Pipeline() as pipeline_dai:
    pipeline_dai.setCalibrationData(calibData)
    cam = pipeline_dai.create(dai.node.Camera).build()
    videoQueue = cam.requestOutput((560, 560)).createOutputQueue()

    pipeline_dai.start()
    while pipeline_dai.isRunning():
        while True:
            try:
                videoIn = videoQueue.get()
                assert isinstance(videoIn, dai.ImgFrame)
            except ValueError:
                continue

            frame = videoIn.getCvFrame()
            frame_copy = frame.copy()

            # ---------------------------
            # YOLO detection
            results = yolo_model.predict(
                source=frame,
                show=False,
                classes=[47],
                max_det=1,
                save=False
            )
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                cv2.circle(frame_copy, (x_center, y_center), 5, (0, 255, 0), -1)

            # ---------------------------
            # DepthAnything inference
            # Resize to multiple of 14 (e.g., 560x560)
            resized_frame = cv2.resize(frame, (560, 560))

            frame_tensor = torch.from_numpy(resized_frame).float() / 255.0  # normalize
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                depth_map = depth_model(frame_tensor)
                depth_map = depth_map.squeeze().cpu().numpy()

            # Normalize and colorize depth map
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = np.uint8(depth_vis)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

            # ---------------------------
            # Display
            combined_streams = np.hstack([frame_copy, depth_colored])
            cv2.imshow("MERGED", combined_streams)

            if cv2.waitKey(1) == ord("q"):
                break
