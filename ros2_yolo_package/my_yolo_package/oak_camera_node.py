"""
ROS 2 node for configuration of the pipeline of OAK-D Lite camera and publishing of RGB and depth frames.

This node configures a DepthAI pipeline to:
- Get RGB and depth frames from queue
- Apply filters 
- Normalize disparity for visualization
- Publish both streams as ROS Image messages

Published topics:
- image_raw (sensor_msgs/Image, BGR8)
- depth_frame_to_inference (sensor_msgs/Image, mono8)
"""
import depthai as dai
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import atexit
import os
import time
from ament_index_python.packages import get_package_share_directory

class OakCameraNode(Node):
    """
    ROS 2 node that streams RGB and depth images from an OAK-D Lite camera.

    The node initializes a DepthAI pipeline, retrieves RGB and stereo
    disparity frames, converts them to ROS Image messages, and publishes
    them at approximately 30 FPS.
    """
    def __init__(self):
        """
        Initialize the OAK-D camera node.

        This method:
        - Loads camera calibration data
        - Creates ROS publishers
        - Configures the DepthAI pipeline
        - Retrieves first RGB and stereo disparity frames        
        - Starts a periodic timer callback
        """
        super().__init__("oak_camera_node")

        cnfg_abs_path = get_package_share_directory("my_yolo_package")

        self.get_logger().info("Trying to load calibration file...")
        jsonfile = os.path.join(cnfg_abs_path, "config", "184430101153051300_09_28_25_13_00.json")
        self.get_logger().info("Calibration file loaded succesfully")

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
        self.maxDisparity = 1

        try:
            self.get_logger().info("Connecting to OAK-D Lite...")
            self.pipeline.start()
            self.get_logger().info("OAK-D Lite camera connected")
            self.videoIn = self.videoQueue.get()
            self.disparity = self.disparityQueue.get()

            assert isinstance(self.videoIn, dai.ImgFrame)
            assert isinstance(self.disparity, dai.ImgFrame)
   
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OAK-D Lite: {e}")
            rclpy.shutdown()
            return


        timer_period = float(1.0 / 30)
        self.timer = self.create_timer(timer_period, self.time_callback)

        atexit.register(self.cleanup)

    def time_callback(self):
        """
        Timer callback that publishes RGB and depth images.

        This method:
        - Retrieves frames from DepthAI output queues
        - Normalizes disparity values for visualization
        - Converts frames to ROS Image messages
        - Publishes them on their respective topics
        """
        rgb_in = self.videoQueue.get()
        depth_in = self.disparityQueue.get()
        if rgb_in is None and depth_in is None:
            self.get_logger().info("No frame received from OAK-D")
            return 

        rgb_frame = rgb_in.getCvFrame()

        self.npDisparity  = depth_in.getFrame()
        self.maxDisparity = max(self.maxDisparity, np.max(self.npDisparity))
        normalizedDisparity = ((self.npDisparity / self.maxDisparity) * 255).astype(np.uint8)

        ros_image_rgb = self.bridge.cv2_to_imgmsg(rgb_frame, "bgr8")
        ros_image_rgb.header.stamp = rclpy.time.Time(seconds=rgb_in.getTimestamp().total_seconds()).to_msg>
        ros_image_rgb.header.frame_id = "rgb_frame"

        ros_image_depth = self.bridge.cv2_to_imgmsg(normalizedDisparity, "mono8")
        ros_image_depth.header.stamp = rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to>
        ros_image_depth.header.frame_id = "depth_frame_to_inference"

        self.publisher_rgb.publish(ros_image_rgb)
        self.publisher_depth.publish(ros_image_depth)

    def cleanup(self):
        if hasattr(self, 'device'):
            self.device.close()
            self.get_logger().info("OAK-D device closed.")
    
def main(args=None):
    """
    ROS2 node entry point
    """
    rclpy.init(args=args)
    camera_node = OakCameraNode()
    try:
        rclpy.spin(camera_node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()









