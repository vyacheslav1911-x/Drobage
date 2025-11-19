  GNU nano 4.8                                      visualizer_node.py                                                 
import cv2
import rclpy 
import numpy as np
import threading
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

class Visualizer(Node):
    def __init__(self):
        super().__init__("visualizer_node")
        self.annotated_frame = None
        self.depth_frame_colorized = None
        self.get_logger().info("Initializing visualizer node...")
        self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.colorMap[0] = [0, 0, 0]
        self.sub_annotated = self.create_subscription(Image, "annotated_image", self.annotated_callback, 10)
        self.sub_depth =  self.create_subscription(Image, "depth_frame", self.depth_callback, 10)
        self.bridge = CvBridge()




    def annotated_callback(self, msg: Image):
        self.annotated_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    def depth_callback(self, msg: Image):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_frame = cv2.resize(self.depth_frame, (640, 480))
        self.depth_frame_colorized = cv2.applyColorMap(self.depth_frame, self.colorMap)
        
    def visualize(self): 
        while True:
            if self.annotated_frame is None or self.depth_frame_colorized is None:
                continue
            self.combined_streams = np.hstack([self.annotated_frame, self.depth_frame_colorized])
            cv2.imshow("Combined Stream", self.combined_streams)
            if cv2.waitKey(1) == ord("q"):
                break  

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = Visualizer()

    vis_thread = threading.Thread(target=visualizer_node.visualize, daemon=True)
    vis_thread.start()

    try:
        rclpy.spin(visualizer_node)
    except(KeyboardInterrupt, SystemExit):
        pass
    finally:
        visualizer_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()



