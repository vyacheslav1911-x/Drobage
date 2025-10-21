import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def CameraNode(Node):
    def __init__(self):
        super.__init__("camera_node_r")
        self.declare_parameters('video_src', 0)
        self.declare_parameters('video_fps', 5)
        video_src = get_parameters('video_src').get_parameter_value().integer_value
        video_fps = get_parameters('video_fps').get_parameter_value().integer_value
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        timer_period = float(1/video_fps)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.cap because we want to store
        self.cap = cv2.VideoCapture(video_src)
        if not self.cap.isOpened():
            self.get_logger().error("Could not find the device")
            rclpy.shutdown()
        self.bridge = CvBridge()
        self.get_logger().info(f"Started at src: {video_src} at {video_fps} fps")
def time_callback():
    ret, frame = self.cap.read()
    if ret:
            # Convert the OpenCV image (BGR) to a ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            
            # Stamp the message with the current time
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = "camera_frame"
            
            # Publish the image message
        self.publisher_.publish(ros_image)
    else:
        self.get_logger().warn("Failed to capture frame.")
def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    
    # Cleanup
    camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
