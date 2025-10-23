import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D,ObjectHypothesisWithPose
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import sys
import os
from ament_index_python.packages import get_package_share_directory

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node_r')
        package_share_directory = get_package_share_directory('my_yolo_package')
        default_model_path = os.path.join(package_share_directory, 'models','yolov8n.pt')

        self.declare_parameter('model', default_model_path)#model
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('max_detections', 1)
        self.declare_parameter('class_detection', [47])#apple by default (class is set acording to coco dataset)
        self.declare_parameter('device', '0')#gpu by default
        #setters
        model_name = self.get_parameter('model').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.class_detection= self.get_parameter('class_detection').get_parameter_value().integer_array_value

        self.bridge = CvBridge()
        try:
            self.model = YOLO(model_name)
            self.get_logger().info(f"Using YOLO model: {model_name}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            sys.exit(1)
        self.subscription = self.create_subscription(Image,'image_raw',self.image_callback, 10)
        self.publisher_m = self.create_publisher(Int32MultiArray, 'yolo_detections_coords', 10)
        self.publisher_ = self.create_publisher(Detection2DArray, 'yolo_detections', 10)

    def image_callback(self, msg: Image):
        #converting img  to np array via bridge
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(source= cv_image, device = self.device, conf = self.conf, classes = self.class_detection,max_det = self.max_detections)
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header  # Use the same header as the image
        for result in results:
            for box in result.boxes:
                detection = Detection2D()
                xywh = box.xywh[0]
                detection.bbox.center.x = float(xywh[0])
                detection.bbox.center.y = float(xywh[1])
                detection.bbox.size_x = float(xywh[2])
                detection.bbox.size_y = float(xywh[3])



                # --- CORRECTED SECTION for ROS 2 Foxy ---
                # The ObjectHypothesisWithPose message in Foxy has 'id' and 'score' directly.
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = str(self.model.names[int(box.cls[0])])
                hypothesis.score = float(box.conf[0])
                # --- END CORRECTION ---
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)
                center_of_bb = Int32MultiArray()
                center_of_bb.data = [int(xywh[0]), int(0)]
                self.get_logger().info(f"Cords: {center_of_bb}")
                self.publisher_m.publish(center_of_bb)
            self.publisher_.publish(detections_msg)
            '''original_cords = result.boxes.data
            if original_cords.shape[0] > 0:
                
                x_center = ((original_cords[0][0] + original_cords[0][2] ) / 2)
                ##y_center = ((original_cords[0][1] + original_cords[0][3] ) / 2)
                center_of_bb = Int32MultiArray()
                center_of_bb.data = [int(x_center.item()), int(0)]
                self.get_logger().info(f"Cords: {center_of_bb}")
                self.publisher_.publish(center_of_bb)'''
def main(args=None):
    rclpy.init(args=args)
    yolo_node_r = YoloNode()
    try:
        rclpy.spin(yolo_node_r)
    except(SystemExit, KeyboardInterrupt):
        pass
    yolo_node_r.destroy_node()
    rclpy.shutdown()
if __name__ =='__main__':
    main()