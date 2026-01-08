"""
ROS 2 node for running YOLOv8 inference on incoming images.

This node:
- Subscribes to RGB images and depth frames
- Runs YOLOv8 inference using Ultralytics
- Publishes annotated images
- Publishes 2D object detections as vision_msgs/Detection2DArray
- Passes depth frames to the next node

Subscribed topics:
- image_raw (sensor_msgs/Image)
- depth_frame_to_inference (sensor_msgs/Image)

Published topics:
- annotated_image (sensor_msgs/Image)
- depth_frame (sensor_msgs/Image)
- detections (vision_msgs/Detection2DArray)
"""

from ultralytics import YOLO
import rclpy
import threading
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D,ObjectHypothesisWithPose
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import sys
import os
from ament_index_python.packages import get_package_share_directory

class YoloNode(Node):
    """
    ROS 2 node that performs YOLOv8 inference on incoming RGB images.

    The node receives RGB images, runs object detection using a TensorRT
    YOLOv8 model, publishes annotated images, republishes depth images form previous node and outputs detections using
    standard ROS vision messages.
    """
    def __init__(self) -> None:
        """
        Initialize the YOLO inference node.

        This method:
        - Declares and loads ROS parameters
        - Loads the YOLOv8 model defaults
        - Creates publishers and subscribers
        """
        print("Init")
        super().__init__('inference_node')
        self.cv_image = None
        self.latest_frame = None

        self.publisher_annotated = self.create_publisher(Image, "annotated_image", 5)
        self.publisher_depth = self.create_publisher(Image, "depth_frame", 5)
        self.publisher_detection = self.create_publisher(Detection2DArray, "detections", 5)
  
        pkg_share_directory = get_package_share_directory('my_yolo_package')
        defaults = {"model_path" : os.path.join(pkg_share_directory, "models", "yolov8n_raw.engine"),
                    "conf":0.5,
                    "max_detections":1,
                    "class_detection":[47],
                    "device" : 'cuda'
                    }
        self.params = {}

        for name, value in defaults.items():
            self.declare_parameter(name, value)
            self.params[name] = self.get_parameter(name).value

        self.model_path = self.params["model_path"]
        self.conf = self.params["conf"]
        self.max_detections = self.params["max_detections"]
        self.class_detection = self.params["class_detection"]
        self.device = self.params["device"]

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"{self.model} was loaded succsefully")
        except Exception as e:
            self.get_logger().error(f"Could not load {e}")

        self.subscription = self.create_subscription(
        Image, 
        'image_raw', 
        self.image_callback,
        5)

        self.subscription_depth = self.create_subscription(
        Image, 
        "depth_frame_to_inference",
        self.depth_callback,
        5)

        self.bridge = CvBridge()
        self.message_received = False

    def depth_callback(self, msg: Image) -> None:
        """
        Receive and republish depth frames.

        The depth image is converted to OpenCV format and immediately
        republished without modification. This allows downstream nodes
        to synchronize depth with detections.
        """
        self.depth_frame_inference = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_frame = self.bridge.cv2_to_imgmsg(self.depth_frame_inference, encoding = "passthrough")

        self.publisher_depth.publish(self.depth_frame)

    def image_callback(self, msg: Image) -> None:
        """
        Store the most recent RGB image for inference.

        The image is not processed directly in the callback to avoid
        blocking the ROS executor. Inference is handled in a separate
        processing thread.
        """
        self.latest_frame = msg


    def processing_thread(self) -> None:
        """
        Perform YOLO inference in a background thread.

        This method:
        - Waits for incoming images
        - Runs YOLOv8 inference
        - Publishes Detection2DArray messages
        - Publishes annotated images with bounding boxes
        """
        while rclpy.ok():
            if self.latest_frame is None:
                time.sleep(0.1)
                continue
            try:
                detection_2d_msg = Detection2DArray()
                detection_2d_msg.header = self.latest_frame.header
                detection_2d = Detection2D()
                hypothesis = ObjectHypothesisWithPose()

                self.get_logger().info("Message was received succesfully")

                self.cv_image = self.bridge.imgmsg_to_cv2(self.latest_frame, 'bgr8')

                infr_rslts = self.model(
                source=self.cv_image,
                device = self.device,
                conf = self.conf, 
                classes = self.class_detection, 
                max_det = self.max_detections)

                infr = infr_rslts[0]
                x1 = y1 = x2 = y2 = None
                for result in infr_rslts:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection_2d.bbox.center.x = ((x1 + x2) / 2)
                        detection_2d.bbox.center.y = ((y1 + y2) / 2)
                        detection_2d.bbox.size_x = float((x2 - x1))
                        detection_2d.bbox.size_y = float((y2 - y1))

                        hypothesis.id = str(box.cls[0].item())
                        hypothesis.score = float(box.conf[0])        

                        detection_2d.results.append(hypothesis)
                        detection_2d_msg.detections.append(detection_2d)


                if infr.boxes is not None and len(infr.boxes) > 0:
                    box = infr.boxes[0]
                    x1, y1,  x2, y2 = map(int, box.xyxy[0])
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    cv2.circle(self.cv_image, (x_center, y_center),5, (0, 255, 0), -1)
                    cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                self.annotated_image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding = "bgr8")
                self.annotated_image.header = self.latest_frame.header

                self.publisher_detection.publish(detection_2d_msg)
                self.publisher_annotated.publish(self.annotated_image)

            except Exception as e:
                self.get_logger().error(f"An exception occured: {e}")


def main(args=None) -> None:
    """
    ROS2 node entry point
    """
    print("Start")
    rclpy.init(args=args)
    inference_node = YoloNode()

    process_thread = threading.Thread(target=inference_node.processing_thread, daemon=True)
    process_thread.start()

    try:
        rclpy.spin(inference_node)
    except(SystemExit, KeyboardInterrupt):
        pass
    finally:
        inference_node.destroy_node()
        rclpy.shutdown()

if __name__ =='__main__':
    main()










                                                     
