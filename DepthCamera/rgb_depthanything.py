#!/usr/bin/env python3
import sys
import os
import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import requests

# ---------------------------
# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ---------------------------
# Load Hugging Face DepthAnything metric model (Indoor)
processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    use_fast=False
)
hf_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
).to(device)
hf_model.eval()

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

    # DepthAI intrinsics (for reference if using stereo disparity)
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
    f_x = intrinsics[0][0]
    B = 0.075  # meters

    # Camera setup
    cam = pipeline_dai.create(dai.node.Camera).build()
    videoQueue = cam.requestOutput((640, 480)).createOutputQueue()

    monoLeft = pipeline_dai.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline_dai.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline_dai.create(dai.node.StereoDepth)

    # Linking
    monoLeftOut = monoLeft.requestFullResolutionOutput()
    monoRightOut = monoRight.requestFullResolutionOutput()
    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    stereo.setOutputSize(640, 480)
    stereo.setRectification(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    syncedLeftQueue = stereo.syncedLeft.createOutputQueue()
    syncedRightQueue = stereo.syncedRight.createOutputQueue()
    disparityQueue = stereo.disparity.createOutputQueue()

    colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    colorMap[0] = [0, 0, 0]

    pipeline_dai.start()
    maxDisparity = 1

    while pipeline_dai.isRunning():
        try:
            videoIn = videoQueue.get()
            assert isinstance(videoIn, dai.ImgFrame)
            leftSynced = syncedLeftQueue.get()
            rightSynced = syncedRightQueue.get()
            disparity = disparityQueue.get()
        except ValueError:
            continue

        frame = videoIn.getCvFrame()

        # ---------------------------
        # YOLO detection
        results = yolo_model.predict(
            source=frame,
            show=False,
            classes=[47],
            max_det=1,
            save=False
        )

        x1 = y1 = x2 = y2 = None
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            x_center = int((x1 + x2) / 2)
            y_center = int((y1 + y2) / 2)
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), -1)

        # ---------------------------
        # Hugging Face DepthAnything metric inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)
            predicted_depth = outputs.predicted_depth  # meters

        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        # Colorize metric depth
        depth_vis = (depth_resized - np.min(depth_resized)) / (np.max(depth_resized) - np.min(depth_resized))
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        # ---------------------------
        # Optional: DepthAI disparity
        npDisparity = disparity.getFrame().astype(np.float32)
        resized_disparity = cv2.resize(npDisparity, (frame.shape[1], frame.shape[0]))
        maxDisparity = max(maxDisparity, np.max(resized_disparity))
        colorizedDisparity = cv2.applyColorMap(
            ((resized_disparity / maxDisparity) * 255).astype(np.uint8),
            colorMap
        )

        # Compute distance of object from metric depth
        if x1 is not None and y1 is not None:
            distance = depth_resized[y_center, x_center]
            print(f"Object distance DEPTHANYTHING(metric): {distance/2.5:.2f} m")


        TARGET_DISTANCE = 0.3
        integral_error = 0
        Kp = 1000
        Ki = 8
        ip_addr = "192.168.4.1"
        if x1 is not None and y1 is not None:
            disparity_value = resized_disparity[y_center, x_center]
            print("Disparity:",disparity_value)
            if disparity_value < 0.1:
                disparity_value = 0.1
            distance_m = (f_x * B) / disparity_value
            error = distance_m - TARGET_DISTANCE
            integral_error += error
            control_output = Kp * error + Ki * integral_error
            speed = max(0, min(255, float(control_output)))
            print("Error: ", error)
            print("Speed: ", speed)
            command = f'{{"T":11,"L":{int(speed)},"R":{int(speed)}}}'
            try:
                url = f"http://{ip_addr}/js?json={command}"
                requests.get(url, timeout=0.15)
            except Exception as ex:
                print("HTTP error:", ex)

            if distance_m <= TARGET_DISTANCE:
                # Stop the rover
                stop_cmd = '{"T":11,"L":0,"R":0}'
                try:
                    requests.get(f"http://{ip_addr}/js?json={stop_cmd}", timeout=0.15)
                except Exception as ex:
                    print("HTTP error while stopping:", ex)
                integral = 0
                print("✅ Target reached — stopping rover.")
            print(f"Object distance DEPTHAI(metric): {distance_m:.2f} m")
        # ---------------------------
        # Display

        combined_streams = np.hstack([frame, depth_colored, colorizedDisparity])
        cv2.imshow("MERGED", combined_streams)
        if cv2.waitKey(1) == ord("q"):
            break
