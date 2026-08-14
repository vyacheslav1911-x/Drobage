import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
import sys

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_MAGNETOMETER,
    BNO_REPORT_ROTATION_VECTOR
)

class BNO08XNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.mag_pub = self.create_publisher(MagneticField, 'imu/mag', 10)

        self.get_logger().info("Initializing BNO08X on I2C Bus 1...")
        try:
            self.i2c = I2C(1)
            self.bno = BNO08X_I2C(self.i2c, address=0x4a)

            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

            self.get_logger().info("BNO08X Initialized Successfully! Publishing data")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize BNO08X: {e}")
            sys.exit(1)

        self.timer = self.create_timer(0.05, self.publish_data)

    def publish_data(self):
        try:
            accel_x, accel_y, accel_z = self.bno.acceleration
            gyro_x, gyro_y, gyro_z = self.bno.gyro
            mag_x, mag_y, mag_z = self.bno.magnetic
            quat_i, quat_j, quat_k, quat_real = self.bno.quaternion

            now = self.get_clock().now().to_msg()

            imu_msg = Imu()
            imu_msg.header.stamp = now
            imu_msg.header.frame_id = 'imu_link'

            imu_msg.orientation.x = float(quat_i)
            imu_msg.orientation.y = float(quat_j)
            imu_msg.orientation.z = float(quat_k)
            imu_msg.orientation.w = float(quat_real)

            imu_msg.angular_velocity.x = float(gyro_x)
            imu_msg.angular_velocity.y = float(gyro_y)
            imu_msg.angular_velocity.z = float(gyro_z)

            imu_msg.linear_acceleration.x = float(accel_x)
            imu_msg.linear_acceleration.y = float(accel_y)
            imu_msg.linear_acceleration.z = float(accel_z)

            self.imu_pub.publish(imu_msg)

            mag_msg = MagneticField()
            mag_msg.header.stamp = now
            mag_msg.header.frame_id = 'imu_link'
            mag_msg.magnetic_field.x = float(mag_x) * 1e-6
            mag_msg.magnetic_field.y = float(mag_y) * 1e-6
            mag_msg.magnetic_field.z = float(mag_z) * 1e-6

            self.mag_pub.publish(mag_msg)

        except Exception as e:
            self.get_logger().warning(f"I2C Read Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    imu_node = BNO08XNode()
    try:
        rclpy.spin(imu_node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down BNO08X node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

