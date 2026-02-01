import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Float32
import csv

class TrackingLogger(Node):
    def __init__(self):
        super().__init__('tracking_logger')

        self.lat_err = None
        self.fwd_err = None

        self.prev_time = None

        self.file = open('tracking_log4.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'time_s',
            'dt_s',
            'lateral_error',
            'forward_error'
        ])

        self.create_subscription(Int16, 'side_error', self.lat_cb, 10)
        self.create_subscription(Float32, 'forward_distance', self.fwd_cb, 10)

    def lat_cb(self, msg):
        self.lat_err = float(msg.data)
        self.try_log()

    def fwd_cb(self, msg):
        self.fwd_err = float(msg.data)
        self.try_log()

    def try_log(self):
        if self.lat_err is None or self.fwd_err is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9  # seconds

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now - self.prev_time

        self.prev_time = now

        self.writer.writerow([
            now,
            dt,
            self.lat_err,
            self.fwd_err
        ])
        self.file.flush()

    def destroy_node(self):
        self.file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = TrackingLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
