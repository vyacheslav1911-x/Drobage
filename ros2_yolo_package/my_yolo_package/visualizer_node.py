# visualizer_node.py
#LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ros2 run my_yolo_package visualizer_node 
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import message_filters


class VisualizerNode(Node):
    """
    A node that subscribes to image and detection topics, draws bounding
    boxes on the image, and displays it in a window.
    """

    def __init__(self):
        super().__init__('visualizer_node')

        self.bridge = CvBridge()
        self.get_logger().info("Visualizer node started.")

        self.image_sub = message_filters.Subscriber(self, Image, 'image_raw')
        self.detection_sub = message_filters.Subscriber(self, Detection2DArray, 'yolo_detections')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

    def synced_callback(self, image_msg: Image, detections_msg: Detection2DArray):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

            for detection in detections_msg.detections:
                center_x = int(detection.bbox.center.x)
                center_y = int(detection.bbox.center.y)

                width = int(detection.bbox.size_x)
                height = int(detection.bbox.size_y)

                x1 = center_x - width // 2
                y1 = center_y - height // 2
                x2 = x1 + width
                y2 = y1 + height

                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- CORRECTED SECTION for ROS 2 Foxy ---
                # Read from the 'id' and 'score' fields directly.
                if detection.results:
                    class_id = detection.results[0].id
                    score = detection.results[0].score
                    label = f"{class_id}: {score:.2f}"
                    cv2.putText(cv_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # --- END CORRECTION ---

            cv2.imshow("YOLO Detections", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in visualizer callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    visualizer_node = VisualizerNode()
    try:
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    visualizer_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

