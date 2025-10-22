#!/usr/bin/env python3
import sys
import os
import cv2
import depthai as dai
from ultralytics import YOLO
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
import torch



jsonfile = "/Users/ivan/depthai/resources/184430101153051300_09_28_25_13_00.json"
calibData = dai.CalibrationHandler(jsonfile)

model = YOLO("yolov8n.pt")

# Create pipeline
with dai.Pipeline() as pipeline_dai:
    # Define source and output
    pipeline_dai.setCalibrationData(calibData)
    cam = pipeline_dai.create(dai.node.Camera).build()
    videoQueue = cam.requestOutput((640,400)).createOutputQueue()

    # Connect to device and start pipeline
    pipeline_dai.start()
    while pipeline_dai.isRunning():
        while True:
            try:
                videoIn = videoQueue.get()
                assert isinstance(videoIn, dai.ImgFrame)
            except ValueError:
                continue
            frame = videoIn.getCvFrame()
            results = model.predict(
                source=frame,
                show=False,
                classes=[47],
                max_det=1,
                save=False
            )
            result = results[0]  # take the first object for results list
            frame_copy = result.orig_img.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                # Take top-1 box
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                if result.boxes is not None and len(result.boxes) > 0:
                    # Take top-1 box
                    box = result.boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    cv2.circle(frame_copy, (x_center, y_center), 5, (0, 255, 0), -1)

            cv2.imshow("YOLO Livestream", frame_copy)

            if cv2.waitKey(1) == ord("q"):
                break
