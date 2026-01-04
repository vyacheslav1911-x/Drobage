mport cv2
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
    class Forward:
        def __init__(self, bbox):
            self.kf = KalmanFilter(7, 4)
            self.kf.F = np.array(
            [[1,0,0,0,0,0,0], 
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])

            self.kf.H = np.array(
            [[1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]]) 

            self.kf.R[:2,:2] *= 10  
            self.kf.Q *= 0.001 
            self.kf.Q[4:,4:] *= 0.00001
            self.kf.P *= 10 
            self.kf.P[4:, 4:] *= 10

            self.kf.x[:4] = self.convert_bb_to_z(bbox)
            self.time_since_det = 0

        def convert_bb_to_z(self, bbox):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            cx = bbox[0] + w/2
            cy = bbox[1] + h/2
            s = w * h
            r = w/h
            z = [cx, cy, s, r]
            return np.array(z).reshape((4,1))

        def convert_x_to_bb(self, x):
            w = np.sqrt(x[2] * x[3])
            h = x[2]/w
            self.x1_predicted = x[0] - w/2
            self.y1_predicted = x[1] - h/2
            self.x2_predicted = x[0] + w/2
            self.y2_predicted = x[1] + h/2
            bb_predicted = [self.x1_predicted, self.y1_predicted, self.x2_predicted, self.y2_predicted]
            return np.array(bb_predicted).reshape((1,4))

        def predict(self):
            self.kf.predict()
            self.time_since_det += 1
            return self.get_bbox()

        def update(self, bbox):
            self.kf.update(self.convert_bb_to_z(bbox))
            self.time_since_det = 0

        def get_bbox(self):
            return self.convert_x_to_bb(self.kf.x)[0] 

    class Lateral:
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
            self.kf.predict()
            return self.get_side_error()

        def update(self, side_error):
            self.kf.update([[side_error]])

        def get_side_error(self):
            return self.kf.x[0,0]



class Visualizer(Node):
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

        self.bbox = None
        self.bbox_predicted = None
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
        self.bbox = [self.x1, self.y1, self.x2, self.y2]

    def tracking_loop(self):
        if self.bbox is None:
            return
        if self.tracker is None and self.bbox is not None:
            self.tracker = KalmanBox.Forward(self.bbox)

        if self.detected:
            self.tracker.update(self.bbox)
        self.tracker.predict()
        self.bbox_predicted = self.tracker.get_bbox()
        print(f"PREDICTED: {self.bbox_predicted}")
        print(f"ACTUAL: {self.bbox}")

    def ROI_callback(self):
        msg = Float32() 
        if not self.detected and self.bbox is None:
            return
        if self.x1 is not None and self.y1 is not None and self.depth_frame is not None and self.detected:
            self.region = self.depth_frame[int(self.y1-(self.y1*0.05)):int(self.y2-(self.y2*0.05)), self.x1:self.x2]
        elif not self.detected and self.bbox_predicted is not None:
            x1, y1, x2, y2, = self.bbox_predicted
            if np.isnan([x1, y1, x2, y2]).any():
                return
            self.region = self.depth_frame[int(self.bbox_predicted[1]-(self.bbox_predicted[1]*0.05)):int(self.bbox_predicted[3]-(self.bbox_predicted[3]*0.05)), >
        self.valid_pixels = self.region[self.region > 0]
        self.disparity_value = np.mean(self.valid_pixels)
        if self.disparity_value < 0.1:
            self.disparity_value = 0.1
        self.distance_m = (self.f_x * self.B) / self.disparity_value
        print(f"{self.distance_m:.2f}")
        msg.data = round(self.distance_m, 2)
        print(msg)
        self.publisher_frwd_dist.publish(msg)

    def side_tracking_loop(self):
        if self.bbox is None:
            return
        if self.side_tracker is None and self.bbox is not None and self.error_x is not None:
            self.side_tracker = KalmanBox.Lateral(self.error_x)
        if self.detected:
            self.side_tracker.update(self.error_x)
        self.side_tracker.predict()
        self.error_predicted = self.side_tracker.get_side_error()

   
    def side_error_callback(self):
        msg = Int16()
        if self.bb_center is None:
            return
        if self.x1 is not None and self.y1 is not None and self.depth_frame is not None and self.detected:
            self.error_x = int(self.bb_center[0] - self.image_center[0])
        elif not self.detected and self.error_predicted is not None:
            self.error_x = int(self.error_predicted)
        msg.data = self.error_x
        print(self.error_x)
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











