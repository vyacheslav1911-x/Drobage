import depthai as dai
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import atexit # To ensure clean shutdown

class OakCameraNode(Node):
    def __init__(self):
        super().__init__("oak_camera_node")
        self.declare_parameter('video_fps', 30)
        video_fps = self.get_parameter('video_fps').get_parameter_value().integer_value
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.bridge = CvBridge()

        #depthAI pipeline setup
        self.get_logger().info("Configuring DepthAI pipeline...")
        self.pipeline = dai.Pipeline()
        self.cam_rgb = self.pipeline.create(dai.node.Camera).build()
        self.videoQueue = self.cam_rgb.requestOutput((640, 480)).createOutputQueue()
        #rgb Cam_a
        #cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A) 
        #cam_rgb.setFps(video_fps)
        #cam_rgb.setInterleaved(False)

        # XLinkOut node to send frames to the host (this computer)
       #xout_rgb = self.pipeline.create(dai.node.XLinkOut)
       #xout_rgb.setStreamName("rgb")
       #cam_rgb.preview.link(xout_rgb.input)

        # conecting and setting the device
        try:
            self.get_logger().info("connecting to oak camera...")
            self.pipeline.start()
            self.get_logger().info("oak camera connected.")
            # output queue from the device
            while self.pipeline.isRunning():
                self.videoIn = self.videoQueue.get()
                assert isinstance(self.videoIn, dai.ImgFrame)
                cv2.imshow("Video", self.videoIn.getCvFrame())
                if cv2.waitKey(1) == ord("q"):
                    break
        except Exception as e:
            self.get_logger().error(f"Failed to connect to OAK-D: {e}")
            rclpy.shutdown()
            return
        # --- ROS Timer ---
        timer_period = float(1.0 / video_fps)
        self.timer = self.create_timer(timer_period, self.time_callback)
        self.get_logger().info(f"OAK-D node started, publishing to /image_raw at {video_fps} FPS.")

        # Ensure clean shutdown
        atexit.register(self.cleanup)

    def time_callback(self):
        # Get frame from OAK-D queue
        in_rgb = self.videoQueue.get()

        if in_rgb is not None:
            # Convert DepthAI frame to OpenCV format (BGR)
            frame = in_rgb.getCvFrame()

            # Convert the OpenCV image (BGR) to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")

            # Stamp the message with the current time
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "oak_camera_frame"

            # Publish the image message
            self.publisher_.publish(ros_image)
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
