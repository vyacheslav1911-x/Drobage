"""
ROS 2 node for visualizing depth and annotated RGB images while performing
object tracking with 1D forward and lateral Kalman filters.

This node:
- Subscribes to annotated RGB images, depth frames, and 2D object detections
- Computes forward distance from depth using disparity
- Tracks distance and lateral error using Kalman filters
- Publishes forward distance, side error, detection flags, and bounding box coordinates
- Visualizes combined RGB and depth streams

Subscribed topics:
- annotated_image (sensor_msgs/Image)
- depth_frame (sensor_msgs/Image)
- detections (vision_msgs/Detection2DArray)

Published topics:
- forward_distance (std_msgs/Float32)
- side_error (std_msgs/Int16)
- detection (std_msgs/Bool)
- coords (std_msgs/Float32MultiArray)
"""

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
from filterpy.kalman import KalmanFilter
#x - state vector
#z - measurement vector
#F - state transition matrix
#H - observation matrix
#R - measurement noise covariance
#Q - process noise covariance
#P - error covariance

class KalmanBox:
    """
    Kalman filter wrappers for smoothing object distance and lateral error.

    Contains two inner classes:
    - Forward: tracks object distance in 1D using [distance, velocity] state
    - Lateral: tracks lateral (x-axis) error relative to image center
    """
    class Forward:
        """
        1D Kalman filter for smoothing forward distance.
        State vector: [distance, velocity]
        """
        def __init__(self, initial_distance):
            self.kf = KalmanFilter(2,1)
            self.kf.F = np.array([[1, 1],
                                 [0, 1]])  
            self.kf.H = np.array([[1, 0]])  
            self.kf.x = np.array([[initial_distance], [0]])
            self.kf.R *= 0.05  
            self.kf.Q *= 0.00001  

        def predict(self) -> float:
            """
            Predict the next state of the filter.
            """
            self.kf.predict()
            return self.kf.x[0, 0]

        def update(self, distance: float) -> None:
            """
            Update the filter with a new measurement.
            """
            self.kf.update([[distance]])

        def get_distance(self) -> float:
            """
            Get the current estimated distance from the filter.
            """    
            return self.kf.x[0, 0]

    class Lateral:
        """
        1D Kalman filter for lateral error (x-axis) relative to image center.
        
        State vector: [side_error, velocity]
        """
        def __init__(self, side_error):
            self.kf = KalmanFilter(2,1)
            self.kf.F = np.array(
            [[1,1],
             [0,1]])

            self.kf.H = np.array(
            [[1,0]])

            self.kf.x = np.array([[side_error], [0]])

            self.kf.R *= 10
            self.kf.Q[[0,0],[1,1]] *= 0.00001

        def predict(self):
            """
            Predict the next state of the filter.
            """
            self.kf.predict()
            return self.get_side_error()

        def update(self, side_error):
            """
            Update the filter with a new measurement.
            """
            self.kf.update([[side_error]])

        def get_side_error(self):
            """
            Get the current estimated distance from the filter.
            """    
            return self.kf.x[0,0]



class Visualizer(Node):
    """
    ROS 2 node for visualizing RGB and depth frames and performing real-time
    forward and lateral object tracking using Kalman filters.
    """
    def __init__(self):
        super().__init__("visualizer_node")
        self.annotated_frame = None
        self.depth_frame_colorized = None

        self.get_logger().info("Initializing visualizer node...")

        self.colorMap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
        self.colorMap[0] = [0, 0, 0]
        self.image_center = (320, 240)

        self.tracker = None
        self.side_tracker = None

        self.error_predicted = None
        self.distance_m = None

        self.bbox = None
        self.bbox_predicted = None
        self.bb_center = None    
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.detected = False
        self.measurement_invalid = False

        self.disparity_spike = False
        self.prev_distance = None
        self.last_valid_distance = None
        self.recovery = False 

        self.f_x = 457.798
        self.B = 0.075
        self.rate_of_change = 0

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
        self.create_timer(float(1/30), self.tracking_loop)
        self.create_timer(float(1/30), self.side_tracking_loop)

        self.bridge = CvBridge()

    def annotated_callback(self, msg: Image):
        self.annotated_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8") 

    def depth_callback(self, msg: Image):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        self.depth_frame = cv2.resize(self.depth_frame, (640, 480))
        self.depth_frame_colorized = cv2.applyColorMap(self.depth_frame, self.colorMap)

    def detection_callback(self, msg: Detection2DArray):
        """
        Callback for 2D object detections.

        Extracts bounding box information and center coordinates for the
        first detection, and sets detection flags.
        """
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



        h, w = 480, 640

        self.x1 = max(0, int(self.x_center - self.size_x / 2))
        self.x2 = min(w, int(self.x_center + self.size_x / 2))
        self.y1 = max(0, int(self.y_center - self.size_y / 2)*0.99)
        self.y2 = min(h, int(self.y_center + self.size_y / 2)*0.99)
        self.bbox = [self.x1, self.y1, self.x2, self.y2]

    def is_disparity_spike(self, distance, prev_distance, threshold=0.25):
        """
        Check if the current distance measurement is a spike compared to previous.
        """
        if prev_distance is None:
            return False
        return distance > prev_distance*(1+threshold)        

    def recovered(self, actual, last, tol=0.15):        
        """
        Check if the measurement has recovered to a valid range after spike was detected.
        """
        print(f"DIFFERENCE: {abs(actual - last)}")
        return abs(actual - last) < tol

#    def derivative(self):
#        if self.distance_m is not None:
#            previous = self.prev_distance if self.prev_distance is not None else 0
#            self.rate_of_change = (self.distance_m - previous) / 0.033
#            self.prev_distance = self.distance_m
#            print(self.rate_of_change)    


    def ROI_callback(self):
        """
        Compute forward distance using depth frame and object bounding box.
        """
        msg = Float32() 
        if not self.detected and self.bbox is None:
            return
        if self.x1 is not None and self.y1 is not None and self.depth_frame is not None and self.detected:
            self.region = self.depth_frame[int(self.y1-(self.y1*0.05)):int(self.y2-(self.y2*0.05)), self.x1:self.x2]
        self.valid_pixels = self.region[self.region > 0]
        self.disparity_value = np.mean(self.valid_pixels)
        if self.disparity_value < 0.1:
            self.disparity_value = 0.1
        self.distance_m = (self.f_x * self.B) / self.disparity_value  
        self.disparity_spike = self.is_disparity_spike(self.distance_m, self.prev_distance)
      #  self.prev_distance = self.distance_m
      #  if self.disparity_spike:
      #      return
        if self.disparity_spike and not self.measurement_invalid:
            self.measurement_invalid = True
            self.recovery = False
      #      self.recovery = self.recovered(self.distance_m, self.last_valid_distance)
        if self.measurement_invalid:
            self.recovery = self.recovered(self.distance_m, self.last_valid_distance)     
            if self.recovery:
                self.measurement_invalid = False

        if not self.measurement_invalid: 
            distance_out = self.distance_m
            self.last_valid_distance = self.distance_m
        else: 
            distance_out = self.last_valid_distance
        self.prev_distance = self.distance_m     
        print(f"{self.distance_m:.2f}")
        print(f"SPIKE: {self.disparity_spike}")
        print(f"MEASUREMENT INVALID: {self.measurement_invalid}")
        print(f"RECOVERY: {self.recovery}")
        print(f"LAST VALID DISTANCE: {self.last_valid_distance}")
        msg.data = round(distance_out, 2)
        print(msg)
        self.publisher_frwd_dist.publish(msg)
 
    def tracking_loop(self):
        """
        Update the forward distance Kalman filter.
        """
        if self.distance_m is None:
            return
        if self.tracker is None and self.distance_m is not None:
            self.tracker = KalmanBox.Forward(self.distance_m)
        self.tracker.predict()
        if self.distance_m is not None and not self.disparity_spike and not self.measurement_invalid:
            self.tracker.update(self.distance_m)
        self.distance_predicted = self.tracker.get_distance()
        print(f"PREDICTED: {self.distance_predicted}")
        print(f"ACTUAL: {self.distance_m}")


    def side_tracking_loop(self):
        """
        Update the lateral error Kalman filter.
        """
        if self.bbox is None:
            return
        if self.side_tracker is None and self.bbox is not None and self.error_x is not None:
            self.side_tracker = KalmanBox.Lateral(self.error_x)
        if self.detected:
            self.side_tracker.update(self.error_x)
        self.side_tracker.predict()
        self.error_predicted = self.side_tracker.get_side_error()

   
    def side_error_callback(self):
        """
        Compute and publish lateral error relative to image center.
        """
        msg = Int16()
        if self.bb_center is None:
            return
        if self.x1 is not None and self.y1 is not None and self.depth_frame is not None and self.detected:
            self.error_x = int(self.bb_center[0] - self.image_center[0])
        elif not self.detected and self.error_predicted is not None:
            self.error_x = int(self.error_predicted)
        msg.data = max(-320, min(self.error_x, 320))
        print(self.error_x)
        self.publisher_err_x.publish(msg)
    
    def detection(self):
        """
        Publish detection as a Boolean flag
        """
        msg = Bool()
        msg.data = self.detected
        self.publisher_det.publish(msg) 

    def visualize(self):
        """
        Visualize the frames using cv window
        """
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

          
