#!/usr/bin/env python3
import depthai as dai
import torch
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D,ObjectHypothesisWithPose
from std_msgs.msg import Int32MultiArray
import cv2
import threading
import time
import numpy as np
from cv_bridge import CvBridge
import atexit
import os
from ament_index_python.packages import get_package_share_directory

class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera_node")

        cnfg_abs_path = get_package_share_directory("my_yolo_package")
        self.get_logger().info("Trying to load calibration file...")
        jsonfile = os.path.join(cnfg_abs_path, "config", "184430101153051300_09_28_25_13_00.json")
        self.get_logger().info("Calibration file loaded succesfully")

        self.declare_parameter('video_fps', 40)
        video_fps = self.get_parameter('video_fps').get_parameter_value().integer_value

        self.publisher_rgb = self.create_publisher(Image, 'image_raw', 5)
        self.publisher_depth = self.create_publisher(Image, 'depth_frame_to_inference', 5)
        self.bridge = CvBridge()

        self.get_logger().info("Configuring DepthAI pipeline...")

        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.create(dai.node.Camera).build()
        self.videoQueue = self.cam_rgb.requestOutput((640, 480)).createOutputQueue()
        calibData = dai.CalibrationHandler(jsonfile)
        self.pipeline.setCalibrationData(calibData)
        self.intrinsics  = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
        f_x = self.intrinsics[0][0]
        print(f_x)
        B = 0.075 #meters

        self.monoLeft = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        self.monoRight = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        self.monoLeftOut = self.monoLeft.requestFullResolutionOutput()
        self.monoRightOut = self.monoRight.requestFullResolutionOutput()

        self.monoLeftOut.link(self.stereo.left)
        self.monoRightOut.link(self.stereo.right)
    
        self.stereo.setOutputSize(640,480)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setRectification(True)
        self.stereo.setSubpixel(False)
        self.stereo.setExtendedDisparity(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        self.stereo.initialConfig.postProcessing.temporalFilter.enable = False
        self.stereo.initialConfig.postProcessing.spatialFilter.enable = True

        self.syncedLeftQueue = self.stereo.syncedLeft.createOutputQueue()
        self.syncedRightQueue = self.stereo.syncedRight.createOutputQueue()
        self.disparityQueue = self.stereo.disparity.createOutputQueue()


        # conecting and setting the device
        try:
            self.get_logger().info("Connecting to OAK-D Lite...")
            self.pipeline.start()
            self.get_logger().info("OAK-D Lite camera connected")
            time.sleep(0.5)
            self.videoIn = self.videoQueue.tryGet()
            self.disparity = self.disparityQueue.tryGet()

            assert isinstance(self.videoIn, dai.ImgFrame)
            assert isinstance(self.disparity, dai.ImgFrame)   
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OAK-D Lite: {e}")
            rclpy.shutdown()
            return

        #yolo    
        self.cv_image = None
        self.publisher_annotated = self.create_publisher(Image, "annotated_image", 5)
        self.publisher_depth = self.create_publisher(Image, "depth_frame", 5)
        self.publisher_detection = self.create_publisher(Detection2DArray, "detections", 5)

        pkg_share_directory = get_package_share_directory('my_yolo_package')
        defaults = {"model_path" : os.path.join(pkg_share_directory, "models", "yolov8n_raw.engine"),
                    "conf":0.5,
                    "max_detections":1,
                    "class_detection":[47],
                    "device" : '0'
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
 


        timer_period = float(1.0 / video_fps)
        self.timer = self.create_timer(timer_period, self.time_callback)
        atexit.register(self.cleanup)
        self.maxDisparity = 1

    def time_callback(self):
        rgb_in = self.videoQueue.tryGet()
        depth_in = self.disparityQueue.tryGet()
        if self.videoIn is None:
            self.get_logger.info("No frame received from OAK-D")
            return 

        if rgb_in is not None and depth_in is not None:

            rgb_frame = rgb_in.getCvFrame()
            self.cv_image = rgb_frame   
            self.npDisparity  = depth_in.getFrame()
            self.maxDisparity = max(self.maxDisparity, np.max(self.npDisparity))
            normalizedDisparity = ((self.npDisparity / self.maxDisparity) * 255).astype(np.uint8)
            self.get_logger().info("Message was received succesfully")
            infr_rslts = self.model(source=rgb_frame,
                                        device = "cuda",
                                        conf = self.conf, 
                                        classes = self.class_detection, 
                                        max_det = self.max_detections)
            infr = infr_rslts[0]
            x1 = y1 = x2 = y2 = None
            if infr.boxes is not None and len(infr.boxes) > 0:
                box = infr.boxes[0]
                x1, y1,  x2, y2 = map(int, box.xyxy[0])
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                cv2.circle(rgb_frame, (x_center, y_center),5, (0, 255, 0), -1)
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)




#---------------PUBLISHER
           # ros_image_rgb = self.bridge.cv2_to_imgmsg(rgb_frame, "bgr8")
           # ros_image_rgb.header.stamp = rclpy.time.Time(seconds=rgb_in.getTimestamp().total_seconds()).to_msg()
           # ros_image_rgb.header.frame_id = "rgb_frame"

           # ros_image_depth = self.bridge.cv2_to_imgmsg(normalizedDisparity, "mono8")
           # ros_image_depth.header.stamp = rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
           # ros_image_depth.header.frame_id = "depth_frame_to_inference"

           # self.publisher_rgb.publish(ros_image_rgb)
           # self.publisher_depth.publish(ros_image_depth)

        else:
            self.get_logger().warn("Failed to capture frame from OAK-D.")

       # self.subscription = self.create_subscription(Image, 
       #                                             'image_raw', 
       #                                              self.image_callback,
       #                                              5)

       # self.subscription_depth = self.create_subscription(Image, 
       #                                                    "depth_frame_to_inference",
       #                                                    self.depth_callback,
       #                                                    5)

       # self.bridge = CvBridge()
       # self.message_received = False

   # def depth_callback(self, msg: Image):
   #     self.depth_frame_inference = self.bridge.imgmsg_to_cv2(msg, "passthrough")
   #     self.depth_frame = self.bridge.cv2_to_imgmsg(self.depth_frame_inference, 
   #                                                  encoding = "passthrough")

   #     self.publisher_depth.publish(self.depth_frame)

#    def image_callback(self, msg: Image):
#        print("Callback")
#        detection_2d_msg = Detection2DArray()
#        detection_2d_msg.header = msg.header
#        detection_2d = Detection2D()
#        hypothesis = ObjectHypothesisWithPose()
#        try:
#            self.message_received = True
#            if self.message_received:
#                self.get_logger().info("Message was received succesfully")
#                self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#                infr_rslts = self.model(source=self.cv_image,
#                                        device = "cuda",
#                                        conf = self.conf, 
#                                        classes = self.class_detection, 
#                                        max_det = self.max_detections)
#                infr = infr_rslts[0]
#                x1 = y1 = x2 = y2 = None
#                for result in infr_rslts:
#                    for box in result.boxes:
#                        x1, y1, x2, y2 = map(int, box.xyxy[0])
#                        detection_2d.bbox.center.x = ((x1 + x2) / 2)
#                        detection_2d.bbox.center.y = ((y1 + y2) / 2)
#                        detection_2d.bbox.size_x = float((x2 - x1))
#                        detection_2d.bbox.size_y = float((y2 - y1))

#                       hypothesis.id = str(box.cls[0].item())
#                       hypothesis.score = float(box.conf[0])        

#                        detection_2d.results.append(hypothesis)
#                        detection_2d_msg.detections.append(detection_2d)


#                if infr.boxes is not None and len(infr.boxes) > 0:
#                    box = infr.boxes[0]
#                    x1, y1,  x2, y2 = map(int, box.xyxy[0])
#                    x_center = int((x1 + x2) / 2)
#                    y_center = int((y1 + y2) / 2)
#                    cv2.circle(self.cv_image, (x_center, y_center),5, (0, 255, 0), -1)
#                    cv2.rectangle(self.cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                self.annotated_image = self.bridge.cv2_to_imgmsg(self.cv_image, encoding = "bgr8")
#                self.annotated_image.header = msg.header
#                self.publisher_detection.publish(detection_2d_msg)
#                self.publisher_annotated.publish(self.annotated_image)

#        except Exception as e:
#            self.get_logger().error(f"An exception occured: {e}")

#    def visualize(self):
#        while rclpy.ok():
#            if self.cv_image is None:
#                continue
            #self.combined_streams = np.hstack([self.annotated_frame, self.depth_frame_colo>
#            cv2.imshow("Combined Stream", self.cv_image)

#            if cv2.waitKey(1) == ord("q"):
#                break  


    def cleanup(self):
        if hasattr(self, 'device'):
            self.device.close()
            self.get_logger().info("OAK-D device closed.")
    
def main(args=None):
    rclpy.init(args=args)
    camera_node = OakCameraNode()

#    vis_thread = threading.Thread(target=camera_node.visualize, daemon=True)
#    vis_thread.start()

    try:
        rclpy.spin(camera_node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:

        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

