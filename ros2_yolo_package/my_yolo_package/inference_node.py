from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D,ObjectHypothesisWithPose
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
import sys
import os
from ament_index_python.packages import get_package_share_directory

class YoloNode(Node):
    def __init__(self):
        print("Init")
        super().__init__('inference_node')
        self.publisher = self.create_publisher(Detection2DArray, "/bb", 10)
        pkg_share_directory = get_package_share_directory('my_yolo_package')
        defaults = {"model_path" : os.path.join(pkg_share_directory, "models", "yolov8>
                    "conf":0.5,
                    "max_detections":1,
                    "class_detection":[47],
                    "device" : '0'
                    }
        self.params = {}
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            self.params[name] = self.get_parameter(name).value

        self.model_path = self.params["model_path"]
        self.conf = self.params["conf"]
        self.max_detections = self.params["max_detections"]
        self.class_detection = self.params["class_detection"]
        self.device = self.params["device"]
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"{self.model} was loaded succsefully")
        except Exception as e:
            self.get_logger().error(f"Could not load {e}")

        self.subscription = self.create_subscription(Image, 
                                                    'image_raw',
                                                     self.image_callback,
                                                     10)

        self.bridge = CvBridge()
        self.message_received = False        

    def image_callback(self, msg: Image):
        print("Callback")
        detection_2d_msg = Detection2DArray()
        detection_2d_msg.header = msg.header
        detection_2d = Detection2D()
        hypothesis = ObjectHypothesisWithPose()
        try:
            #converting img  to np array via bridge
            self.message_received = True
            if self.message_received:
                self.get_logger().info("Message was received succesfully")
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                infr_rslts = self.model(source=cv_image,
                                        device = self.device,
                                        conf = self.conf, 
                                        classes = self.class_detection, 
                                        max_det = self.max_detections)
                infr = infr_rslts[0]
                x1 = y1 = x2 = y2 = None
                for result in infr_rslts:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detection_2d.bbox.center.x = ((x1 + x2) / 2)
                        detection_2d.bbox.center.y = ((y1 + y2) / 2)
                        detection_2d.bbox.size_x =float((x2 - x1))
                        detection_2d.bbox.size_y =float((y2 - y1))

                        hypothesis.id = str(box.cls[0].item())
                        hypothesis.score = float(box.conf[0])        

                        detection_2d.results.append(hypothesis)
                        detection_2d_msg.detections.append(detection_2d)
                    self.publisher.publish(detection_2d_msg)

                if infr.boxes is not None and len(infr.boxes) > 0:
                    box = infr.boxes[0]
                    x1, y1,  x2, y2 = map(int, box.xyxy[0])
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    cv2.circle(cv_image, (x_center, y_center),5, (0, 255, 0), -1)
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)       >
                cv2.imshow("Stream", cv_image)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"An exception occured: {e}")



def main(args=None):
    print("Start")
    rclpy.init(args=args)
    inference_node = YoloNode()
    try:
        rclpy.spin(inference_node)
    except(SystemExit, KeyboardInterrupt):
        pass
    inference_node.destroy_node()
    rclpy.shutdown()
if __name__ =='__main__':
    main()

                                                     
