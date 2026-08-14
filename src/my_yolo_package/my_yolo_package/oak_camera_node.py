#!/usr/bin/env python3
import depthai as dai
import torch
import signal
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
import cv2
import numpy as np
from cv_bridge import CvBridge
import atexit
import os
import time
import requests
import json
import math
from geometry_msgs.msg import TransformStamped

from ament_index_python.packages import get_package_share_directory


class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera_node")
        cnfg_abs_path = "/home/wb/Desktop/Drobage/src/my_yolo_package/share"
        print(cnfg_abs_path)
        self.get_logger().info("Trying to load calibration file...")
        jsonfile = os.path.join(cnfg_abs_path, "config", "184430101153051300_09_28_25_13_00.json")
        print(jsonfile)
        self.get_logger().info("Calibration file loaded succesfully")

        self.cam_info = CameraInfo()
        self.cam_info.width = 640
        self.cam_info.height = 480
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.k =[457.798,0.0,313.368,0.0,456.957,224.113,0.0,0.0,1.0]
        self.cam_info.r = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        self.cam_info.p = [457.798,0.0,313.368,-34.334,0.0,456.957,224.113,0.0,0.0,0.0,1.0,0.0]
        self.cam_info.header.frame_id = "camera_link"
       
        self.publisher_rgb = self.create_publisher(Image, 'image_raw', 3)
        self.publisher_depth = self.create_publisher(Image, 'depth_frame_to_inference', 3)
        self.publisher_depth_original = self.create_publisher(Image, 'image_depth', 3)
        self.publisher_caminfo = self.create_publisher(CameraInfo, 'camera_info', 3)
####
        self.publisher_imu = self.create_publisher(Imu, "imu", 3)
        self.create_timer(float(1/20), self.imu_callback)
        self.imu = Imu()
        self.session = requests.Session()   
####

        self.bridge = CvBridge()

        self.get_logger().info("Configuring DepthAI pipeline...")

        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.create(dai.node.Camera).build()
        self.videoQueue = self.cam_rgb.requestOutput((640, 480), fps=10) 

        calibData = dai.CalibrationHandler(jsonfile)
        self.pipeline.setCalibrationData(calibData)
        self.intrinsics  = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)
       
        self.monoLeft = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        self.monoRight = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        self.monoLeftOut = self.monoLeft.requestFullResolutionOutput(fps=10)
        self.monoRightOut = self.monoRight.requestFullResolutionOutput(fps=10)

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


        self.disparityQueue = self.stereo.disparity.createOutputQueue()
        self.maxDisparity = 1

        self.sync = self.pipeline.create(dai.node.Sync)
        self.monoLeftOut.link(self.sync.inputs['self.monoLeft'])
        self.monoRightOut.link(self.sync.inputs["self.monoRight"])
        self.videoQueue.link(self.sync.inputs["self.cam_rgb"])
        self.queue = self.sync.out.createOutputQueue()


        try:
            self.get_logger().info("Connecting to OAK-D Lite...")
            self.pipeline.start()
            self.get_logger().info("OAK-D Lite camera connected")
            self.disparity = self.disparityQueue.get()
            
            messageGroup = self.queue.get()
            rgb_in = messageGroup['self.cam_rgb']

            assert isinstance(rgb_in, dai.ImgFrame)
            assert isinstance(self.disparity, dai.ImgFrame)
   
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OAK-D Lite: {e}")
            rclpy.shutdown()
            return


        timer_period = float(1.0 / 10)
        self.timer = self.create_timer(timer_period, self.time_callback)
#
        self.rgb_timer = self.create_timer(timer_period, self.publish_rgb_image)
        self.depth_timer = self.create_timer(timer_period, self.publish_depth_image)
        self.depth_original_timer = self.create_timer(timer_period, self.publish_depth_original_image)
        self.caminfo_timer = self.create_timer(timer_period, self.publish_caminfo)
        self.imu_timer = self.create_timer(timer_period, self.publish_imu)

        self.recording = True
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(f"/home/wb/Desktop/output_{self.get_clock().now()}.mp4", self.fourcc, 20.0, (640, 480))
        signal.signal(signal.SIGTERM, self.cleanup)
#        signal.signal(signal.SIGINT, self.cleanup) 
#
#        atexit.register(self.cleanup)

####
    def imu_callback(self):

        self.imu.orientation.x = 0.0
        self.imu.orientation.y = 0.0
        self.imu.orientation.z = 0.0
        self.imu.orientation.w = 1.0

        self.imu.orientation_covariance = [
        0.01,0.0,0.0,
        0.0,0.01,0.0,
        0.0,0.0,0.01
        ]

        self.imu.angular_velocity_covariance = [
        0.02,0.0,0.0,
        0.0,0.02,0.0,
        0.0,0.0,0.02
        ]

        self.imu.linear_acceleration_covariance = [
        0.04,0.0,0.0,
        0.0,0.04,0.0,
        0.0,0.0,0.04
        ]

        self.imu.linear_acceleration.x = 0.0
        self.imu.linear_acceleration.y = 0.0
        self.imu.linear_acceleration.z = 9.80665 

        self.imu.angular_velocity.x = 0.0
        self.imu.angular_velocity.y = 0.0
        self.imu.angular_velocity.z = 0.0

        self.imu.header.frame_id = "imu_link"


###

    def time_callback(self):
        print("alive")
        depth_in = self.disparityQueue.get() #tryGet()
        messageGroup = self.queue.get()  #tryGet()
        if depth_in is None:
            return
        rgb_in = messageGroup['self.cam_rgb']

        rgb_in_stamp = rgb_in.getTimestamp()
        depth_in_stamp = depth_in.getTimestamp()
        print(f"RGB timestamp:{rgb_in_stamp}")
        print(f"depth timestamp:{depth_in_stamp}")
        if rgb_in is None and depth_in is None:
            self.get_logger().info("No frame received from OAK-D")
            return 

        rgb_frame = rgb_in.getCvFrame()
###
        if self.recording and hasattr(self, "out") and self.out is not None:
            self.out.write(rgb_frame)
### 
        self.npDisparity  = depth_in.getFrame()
        self.maxDisparity = max(self.maxDisparity, np.max(self.npDisparity))
        normalizedDisparity = ((self.npDisparity / self.maxDisparity) * 255).astype(np.uint8)
        stamp = self.get_clock().now().to_msg()

        disp = self.npDisparity.astype(np.float32) / 34
        depth_m = np.zeros_like(disp, dtype=np.float32)
        valid = disp > 0
        depth_m[valid] = (457.798 * 0.075) / disp[valid]

        self.ros_image_rgb = self.bridge.cv2_to_imgmsg(rgb_frame, "bgr8")
        self.ros_image_rgb.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
        self.ros_image_rgb.header.frame_id = "camera_optical_frame"

        self.ros_image_depth_original = self.bridge.cv2_to_imgmsg(depth_m) #self.npDisparity*0.15
        self.ros_image_depth_original.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
        self.ros_image_depth_original.header.frame_id = "camera_optical_frame"

        self.ros_image_depth = self.bridge.cv2_to_imgmsg(normalizedDisparity, "mono8")
        self.ros_image_depth.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
        self.ros_image_depth.header.frame_id = "depth_frame_to_inference"

        self.cam_info.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
        self.cam_info.header.frame_id = "camera_optical_frame"
###
        self.imu.header.stamp = self.get_clock().now().to_msg() #rclpy.time.Time(seconds=depth_in.getTimestamp().total_seconds()).to_msg()
        self.imu.header.frame_id = "imu_link"
#        self.publisher_imu.publish(self.imu)

###
        self.publisher_rgb.publish(self.ros_image_rgb)
        self.publisher_depth.publish(self.ros_image_depth)
        self.publisher_depth_original.publish(self.ros_image_depth_original)
        self.publisher_caminfo.publish(self.cam_info)        
        self.publisher_imu.publish(self.imu)

    def publish_rgb_image(self):
        self.publisher_rgb.publish(self.ros_image_rgb)

    def publish_depth_image(self):
        self.publisher_depth.publish(self.ros_image_depth)

    def publish_depth_original_image(self):
        self.publisher_depth_original.publish(self.ros_image_depth_original)

    def publish_caminfo(self):
        self.publisher_caminfo.publish(self.cam_info) 

    def publish_imu(self):
        self.publisher_imu.publish(self.imu)

 


    def cleanup(self,signum,frame):
####
        try:
            if hasattr(self, "out") and self.out is not None:
                self.out.release()
                self.out = None
                self.recording = False
        except Exception as e:
            print(e)

###
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





