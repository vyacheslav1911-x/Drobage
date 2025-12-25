  GNU nano 4.8                                                           visualizer_node.py                                                                      
import cv2
import rclpy 
import numpy as np
import threading
import time
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32, Int16, Bool, Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

class Visualizer(Node):
    def __init__(self):
        super().__init__("visualizer_node")
        self.annotated_frame = None
        self.depth_frame_colorized = None

        self.get_logger().info("Initializing visualizer node...")

        self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.colorMap[0] = [0, 0, 0]
        self.image_center = (320, 240)

        self.bb_center = None    
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.detected = False

        self.f_x = 457.798
        self.B = 0.075

        self.sub_annotated = self.create_subscription(Image, "annotated_image", self.annotated_callback, 5)
        self.sub_depth =  self.create_subscription(Image, "depth_frame", self.depth_callback, 5)
        self.sub_detections = self.create_subscription(Detection2DArray, "detections", self.detection_callback, 5)

        self.publisher_frwd_dist = self.create_publisher(Float32, "forward_distance", 5) 
        self.publisher_err_x = self.create_publisher(Int16, "side_error", 5)        
        self.publisher_det = self.create_publisher(Bool, "detection", 5)
        self.publisher_bb_coords = self.create_publisher(Float32MultiArray, "coords", 5)

        self.create_timer(float(1/30), self.ROI_callback)       
        self.create_timer(float(1/30), self.side_error_callback)
        self.create_timer(float(1/30), self.detection)
        self.create_timer(float(1/30), self.coords)

        self.bridge = CvBridge()

    def annotated_callback(self, msg: Image):
        self.annotated_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8") 

    def depth_callback(self, msg: Image):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_frame = cv2.resize(self.depth_frame, (640, 480))
        self.depth_frame_colorized = cv2.applyColorMap(self.depth_frame, self.colorMap)

    def detection_callback(self, msg: Detection2DArray):
        if len(msg.detections) == 0:
            self.detected = False
            self.x1 = self.x2 = self.y1 = self.y2 = None
            return
        else:
            self.detected = True

        det =  msg.detections[0]
        self.x_center = det.bbox.center.x
        self.y_center = det.bbox.center.y
        self.size_x  = det.bbox.size_x
        self.size_y = det.bbox.size_y
        self.bb_center = (self.x_center, self.y_center)        

        self.x1 = int(self.x_center - self.size_x / 2)
        self.x2 = int(self.x_center + self.size_x / 2)
        self.y1 = int(self.y_center - self.size_y / 2)
        self.y2 = int(self.y_center + self.size_y / 2)
    
    def coords(self):
        msg = Float32MultiArray()
        if self.x1 is not None and self.y1 is not None:
            msg.data = list(map(float, [self.x1, self.y1, self.x2, self.y2]))
            self.publisher_bb_coords.publish(msg)

    def ROI_callback(self):
        msg = Float32() 
        if self.x1 is not None and self.y1 is not None and self.depth_frame is not None:
            self.region = self.depth_frame[int(self.y1-(self.y1*0.05)):int(self.y2-(self.y2*0.05)), self.x1:self.x2]
            self.valid_pixels = self.region[self.region > 0]
            self.disparity_value = np.mean(self.valid_pixels)
            if self.disparity_value < 0.1:
                self.disparity_value = 0.1
            self.distance_m = (self.f_x * self.B) / self.disparity_value
            print(f"{self.distance_m:.2f}")
            msg.data = round(self.distance_m, 2)
            self.publisher_frwd_dist.publish(msg)


   
    def side_error_callback(self):
        msg = Int16()
        if self.bb_center is None:
            return
        self.error_x = int(self.bb_center[0] - self.image_center[0])
        msg.data = self.error_x
        self.publisher_err_x.publish(msg)
    
    def detection(self):
        msg = Bool()
        msg.data = self.detected
        self.publisher_det.publish(msg) 

    def visualize(self):
        while rclpy.ok():
            if self.annotated_frame is None or self.depth_frame_colorized is None:
                time.sleep(0.01)
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












