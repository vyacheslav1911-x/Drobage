#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import sys
import os
from ament_index_python.packages import get_package_share_directory

# Imports for Depth-Anything
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image as PILImage # Use PILImage to avoid name conflict
import numpy as np

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node_r')
        package_share_directory = get_package_share_directory('my_yolo_package')
        default_model_path = os.path.join(package_share_directory, 'models', 'yolov8n.pt')

        # --- YOLO Parameters ---
        self.declare_parameter('model', default_model_path)
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('max_detections', 1)
        self.declare_parameter('class_detection', [47]) # apple
        self.declare_parameter('device', '0') # '0' for GPU, 'cpu' for CPU

        self.yolo_device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.class_detection = self.get_parameter('class_detection').get_parameter_value().integer_array_value

        # --- CV Bridge ---
        self.bridge = CvBridge()

        # --- Load YOLO Model ---
        try:
            self.model = YOLO(default_model_path)
            self.get_logger().info(f"Using YOLO model: {default_model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            sys.exit(1)

        # --- Load Depth-Anything Model ---
        self.get_logger().info("Loading Depth-Anything model...")
        if torch.cuda.is_available():
            self.hf_device = torch.device("cuda")
        else:
            self.hf_device = torch.device("cpu")
        self.get_logger().info(f"Using device for Depth model: {self.hf_device}")
        
        try:
            # You can change "Large" to "Base" or "Small" for faster inference
            # NOTE for Jetson Xavier: "Large" may be too slow. Start with "Base".
            # model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
            model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
            self.hf_model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.hf_device)
            self.hf_model.eval() # Set model to evaluation mode
            self.get_logger().info("Depth-Anything model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load HF model: {e}")
            sys.exit(1)

        # --- ROS Subscription ---
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        
        # --- ROS Publishers ---
        self.publisher_m = self.create_publisher(Int32MultiArray, 'yolo_detections_coords', 10)
        self.publisher_ = self.create_publisher(Detection2DArray, 'yolo_detections', 10)

    @torch.no_grad() # Disable gradient calculations for inference
    def image_callback(self, msg: Image):
        # 1. Convert ROS Image to CV Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img_height, img_width, _ = cv_image.shape
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
            
        # 2. Run YOLO Detection
        results = self.model(source=cv_image, device=self.yolo_device, conf=self.conf, 
                             classes=self.class_detection, max_det=self.max_detections, verbose=False)
        
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header
        
        # 3. Check if any objects were detected
        if not results or not results[0].boxes:
            # Still publish an empty detection message
            self.publisher_.publish(detections_msg)
            return

        # 4. Run Depth-Anything (only if objects are found)
        frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(frame_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.hf_device)

        outputs = self.hf_model(**inputs)
        predicted_depth = outputs.predicted_depth # This is in METERS

        # Resize depth map to match original image
        depth_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(img_height, img_width),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

        # 5. Process Detections and Publish Data
        for result in results:
            for box in result.boxes:
                # --- A. Populate Detection2D Message (for visualizer) ---
                detection = Detection2D()
                xywh = box.xywh[0]
                x_center = int(xywh[0])
                y_center = int(xywh[1])
                
                detection.bbox.center.x = float(x_center)
                detection.bbox.center.y = float(y_center)
                detection.bbox.size_x = float(xywh[2])
                detection.bbox.size_y = float(xywh[3])

                hypothesis = ObjectHypothesisWithPose()
                
                # --- ROS 2 Foxy Fix ---
                # The 'id' field in vision_msgs/ObjectHypothesisWithPose is 'int64' in Foxy.
                # We must pass the integer class ID (e.g., 47), not the string name ("apple").
                # Your visualizer node will need to map this int ID back to a name.
                hypothesis.id = int(box.cls[0]) 
                # --- End Foxy Fix ---
                
                hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)

                # --- B. Get Depth and Populate Coords Message ---
                # Look up the depth at the object's center
                distance_m = depth_resized[y_center, x_center]
                distance_mm = int(distance_m * 1000) # Convert meters to millimeters

                center_of_bb = Int32MultiArray()
                # *** THIS IS THE LINE YOU WANTED TO CHANGE ***
                center_of_bb.data = [x_center, distance_mm]
                # *** END OF CHANGE ***
                
                self.get_logger().info(f"Coords: [x={x_center}, depth={distance_mm}mm]")
                self.publisher_m.publish(center_of_bb)
                
        # 6. Publish all detections
        self.publisher_.publish(detections_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_node_r = YoloNode()
    try:
        rclpy.spin(yolo_node_r)
    except(SystemExit, KeyboardInterrupt):
        pass
    yolo_node_r.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()