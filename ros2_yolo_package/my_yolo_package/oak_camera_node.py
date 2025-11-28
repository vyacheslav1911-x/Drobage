  GNU nano 4.8                                       oak_camera_node.py                                                  
#!/usr/bin/env python3
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
from ament_index_python.packages import get_package_share_directory

class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera_node")

        cnfg_abs_path = get_package_share_directory("my_yolo_package")
        self.get_logger().info("Trying to load calibration file...")
        jsonfile = os.path.join(cnfg_abs_path, "config", "184430101153051300_09_28_25_13_00.json")
        self.get_logger().info("Calibration file loaded succesfully")

        self.declare_parameter('video_fps', 30)
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

            self.videoIn = self.videoQueue.get()
            self.disparity = self.disparityQueue.get()

            assert isinstance(self.videoIn, dai.ImgFrame)
            assert isinstance(self.disparity, dai.ImgFrame)   
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OAK-D Lite: {e}")
            rclpy.shutdown()
            return


        timer_period = float(1.0 / video_fps)
        self.timer = self.create_timer(timer_period, self.time_callback)
        atexit.register(self.cleanup)
        self.maxDisparity = 1
    def time_callback(self):
        rgb_in = self.videoQueue.get()
        depth_in = self.disparityQueue.get()
        if rgb_in is None:
            self.get_logger.info("No frame received from OAK-D")
            return 

        if rgb_in is not None and depth_in is not None:

            rgb_frame = rgb_in.getCvFrame()

            self.npDisparity  = depth_in.getFrame()
            self.maxDisparity = max(self.maxDisparity, np.max(self.npDisparity))
            normalizedDisparity = ((self.npDisparity / self.maxDisparity) * 255).astype(np.uint8)

            ros_image_rgb = self.bridge.cv2_to_imgmsg(rgb_frame, "bgr8")
            ros_image_rgb.header.stamp = rclpy.time.Time(seconds=rgb_in.getTimestamp().total_seconds()).to_msg()
            ros_image_rgb.header.frame_id = "rgb_frame"

            ros_image_depth = self.bridge.cv2_to_imgmsg(normalizedDisparity, "mono8")
            ros_image_depth.header.stamp = rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
            ros_image_depth.header.frame_id = "depth_frame_to_inference"

            self.publisher_rgb.publish(ros_image_rgb)
            self.publisher_depth.publish(ros_image_depth)
        else:
            self.get_logger().warn("Failed to capture frame from OAK-D.")



    def cleanup(self):
        if hasattr(self, 'device'):
            self.device.close()
            self.get_logger().info("OAK-D device closed.")
    
def main(args=None):
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






