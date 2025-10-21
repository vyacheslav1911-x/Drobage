import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import bridge

def CameraNode(Node):
    def __init__(self):
        super.__init__("camera_node_r")
        self.declare_parameters('video_src', 0)
        self.declare_parameters('video_fps', 5)
        video_src = get_parameters('video_src').get_parameter_value().integer_value
        video_fps = get_parameters('video_fps').get_parameter_value().integer_value
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)