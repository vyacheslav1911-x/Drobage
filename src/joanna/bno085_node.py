# link to repository with bno085 driver for ros2
# (can be useful in the future)
# https://github.com/bnbhat/bno08x_ros2_driver

# there is a possibility that some subscribers would need more published data


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
        super().__init__('bno08x_node')
        
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.mag_pub = self.create_publisher(MagneticField, 'imu/mag', 10)
        
        self.get_logger().info("Initializing BNO08X on I2C Bus 1...")
###
        self.sample = 1
        self.sum_quat_i = 0
        self.sum_quat_j = 0
        self.sum_quat_k = 0
        self.sum_quat_w = 0
 
        self.sum_gyro_x = 0
        self.sum_gyro_y = 0
        self.sum_gyro_z = 0

        self.sum_accel_x = 0
        self.sum_accel_y = 0
        self.sum_accel_z = 0 

        
###
        try:
            self.i2c = I2C(1)
            self.bno = BNO08X_I2C(self.i2c, address=0x4a)
            
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            
            self.get_logger().info("BNO08X Initialized Successfully! Publishing at 50Hz...")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize BNO08X: {e}")
            sys.exit(1)
            
        self.timer = self.create_timer(0.02, self.publish_data)


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
            
            imu_msg.orientation_covariance = [0.05, 0.0, 0.0,
                                              0.0, 0.05, 0.0,
                                              0.0, 0.0, 0.05]

            imu_msg.angular_velocity.x = float(gyro_x)
            imu_msg.angular_velocity.y = float(gyro_y)
            imu_msg.angular_velocity.z = float(gyro_z)
            
            imu_msg.angular_velocity_covariance = [0.1, 0.0, 0.0,
                                                   0.0, 0.1, 0.0,
                                                   0.0, 0.0, 0.1]

            imu_msg.linear_acceleration.x = float(accel_x)
            imu_msg.linear_acceleration.y = float(accel_y)
            imu_msg.linear_acceleration.z = float(accel_z)
            
            self.imu_pub.publish(imu_msg)

            mag_msg = MagneticField()
            mag_msg.header.stamp = now
            mag_msg.header.frame_id = 'mag_link'
            mag_msg.magnetic_field.x = float(mag_x) * 1e-6
            mag_msg.magnetic_field.y = float(mag_y) * 1e-6
            mag_msg.magnetic_field.z = float(mag_z) * 1e-6
            
            self.mag_pub.publish(mag_msg)
###
            self.sum_quat_i += float(quat_i) 
            self.sum_quat_j += float(quat_j) 
            self.sum_quat_k += float(quat_k) 
            self.sum_quat_w += float(quat_real) 
 
            self.sum_gyro_x += float(gyro_x) 
            self.sum_gyro_y += float(gyro_y) 
            self.sum_gyro_z += float(gyro_z) 

            self.sum_accel_x += float(accel_x) 
            self.sum_accel_y += float(accel_y)
            self.sum_accel_z += float(accel_z) 

            self.sample += 1

            self.calculate_bias(self.sample, self.sum_quat_i, self.sum_quat_j, self.sum_quat_k, self.sum_quat_w, self.sum_gyro_x, self.sum_gyro_y, self.sum_gyro_z, self.sum_accel_x, self.sum_accel_y, self.sum_accel_z)
#            print(f"Bias quat i: {self.sum_quat_i}, bias quat j: {self.sum_quat_j}, bias quat k: {self.sum_quat_k}, bias quat w: {self.sum_quat_w}")
#            print(f"Bias gyro x: {self.sum_gyro_x}, bias gyro y: {self.sum_gyro_y}, bias gyro z: {self.sum_gyro_z}")
#            print(f"Bias accel x: {self.sum_accel_x}, bias accel y: {self.sum_accel_y}, bias accel z: {self.sum_accel_z}")

        except Exception as e:
            self.get_logger().warning(f"I2C Read Error: {e}")

    def calculate_bias(self, sample, sum_quat_i, sum_quat_j, sum_quat_k, sum_quat_w, sum_gyro_x, sum_gyro_y, sum_gyro_z, sum_accel_x, sum_accel_y, sum_accel_z):
        bias_quat_i = sum_quat_i / sample
        bias_quat_j = sum_quat_j / sample
        bias_quat_k = sum_quat_k / sample
        bias_quat_w = sum_quat_w / sample

        bias_gyro_x = sum_gyro_x / sample
        bias_gyro_y = sum_gyro_y / sample
        bias_gyro_z = sum_gyro_z / sample

        bias_accel_x = sum_accel_x / sample
        bias_accel_y = sum_accel_y / sample
        bias_accel_z = sum_accel_z / sample

        print(f"Bias quat i: {bias_quat_i}, bias quat j: {bias_quat_j}, bias quat k: {bias_quat_k}, bias quat w: {bias_quat_w}")
        print(f"Bias gyro x: {bias_gyro_x}, bias gyro y: {bias_gyro_y}, bias gyro z: {bias_gyro_z}")
        print(f"Bias accel x: {bias_accel_x}, bias accel y: {bias_accel_y}, bias accel z: {bias_accel_z}")



def main(args=None):
    rclpy.init(args=args)
    node = BNO08XNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down BNO08X node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
