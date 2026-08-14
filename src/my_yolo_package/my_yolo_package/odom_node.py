import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from tf_transformations import quaternion_from_euler
import json
import requests
import math
import transforms3d as tf3d
from nav_msgs.msg import Odometry
import time
from geometry_msgs.msg import Quaternion, Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import serial
import threading
import os

class OdomNode(Node):
    def __init__(self):
        super().__init__("odom_node")
        self.publisher_odom = self.create_publisher(Odometry, "odometry", 5)
        self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 20)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.t = TransformStamped()
        self.q = Quaternion()
        self.linear_velocity = None
        self.angular_velocity = None

        self.create_timer(float(1/50), self.twist_callback)
        self.create_timer(float(1/50), self.odom_callback)
#        self.create_timer(float(1/50), self.publish_dynamic_transform)
        self.create_timer(float(1/50), self.publish_odometry)

#        self.session = requests.Session()
        self.ports = ['/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3']
        for port in self.ports:
            if os.path.exists(port):
                self.ser = serial.Serial(port, baudrate=115200, timeout=0.1)

        self.odom = Odometry()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vr_filtered = 0
        self.vl_filtered = 0

        self.last_time = time.time()
        self.timestamp = None
        # Wheel parameters
        self.wheel_base = 0.154  # meters
        self.left_wheel_speed = 0.0  # m/s
        self.right_wheel_speed = 0.0   
        self.subscription = self.create_subscription(
        Image, 
        'image_depth', 
        self.depth_callback,
        5)   
        self.current_time = self.get_clock().now().to_msg()
        self.zero_vel_count = 0


    def depth_callback(self, msg: Image):
        self.timestamp = msg.header.stamp
        print(self.timestamp)

    def cmd_vel_callback(self, msg):
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z

    def twist_callback(self):
        if self.linear_velocity is None or self.angular_velocity is None:
            return
        # Apply minimum threshold to angular velocity if linear velocity is zero
#        if self.linear_velocity == 0.0:
#            if 0.0 < self.angular_velocity < 0.2:
#                self.angular_velocity = 0.2
#            elif -0.2 < self.angular_velocity < 0.0:
#                self.angular_velocity = -0.2

        # Send the velocity data to the UGV as a JSON string
        data = f'{{"T":{13},"X":{self.linear_velocity},"Z":{self.angular_velocity}}}'

#        data = json.dumps({'T': 13, 'X': self.linear_velocity, 'Z': self.angular_velocity}) + "\n"
        print(data)
        self.send_command(data)

    def send_command(self, data) -> None:
        json_command = data
        try:
            self.ser.write((json_command + "\n").encode("utf-8"))
            print("SUCCESS")
            self.ser.flush()
#            self.session.get(json_command, timeout=0.4)     
        except Exception as e:
            print("HTTP error: ", e)

#####
    def odom_callback(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now 
#        json_command = f'"T":{130}'
        self.current_time = self.get_clock().now().to_msg()
        try:
            data_encoded = self.ser.readline().decode('utf-8').strip()
            data = json.loads(data_encoded)
            print(data)
#            if(-10 < data["M1"] < 10  or -10 < data["M2"] < 10 or -10 < data["M3"] < 10 or -10 < data["M4"] < 10):
#                data["M1"] = 0
#                data["M2"] = 0
#                data["M3"] = 0
#                data["M4"] = 0
            vl_clean = 0.1 * (data["M1"] + data["M4"])
            vr_clean = 0.1 * (data["M2"] + data["M3"])

            
            rpm_to_ms = (1 / 60) * (2.0 * math.pi * 0.03725) #0.018625

            vl = vl_clean / 2 * rpm_to_ms #/ 2.0
            vr = vr_clean / 2 * rpm_to_ms #/ 2.0

            alpha = 0.2
            self.vr_filtered = alpha*vr + (1 - alpha) * self.vr_filtered
            self.vl_filtered = alpha*vl + (1 - alpha) * self.vl_filtered   

        # Differential drive kinematics
            v = (self.vr_filtered + self.vl_filtered) / 2 
            omega = (self.vr_filtered - self.vl_filtered) / self.wheel_base 
        # Update pose
            self.theta += omega * dt #omega/20 * dt
            self.x += v * math.cos(self.theta) * dt # + 3.14159) * dt #v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt # + 3.14159) * dt
            print(f"velocity:{v}, omega:{omega}, theta:{self.theta}, x:{self.x}, y:{self.y}, dt: {dt}")

        # Construct message
            self.odom = Odometry()
            self.odom.header.frame_id = "odom"
            self.odom.child_frame_id = "base_link"

            self.odom.pose.pose.position.x = self.x
            self.odom.pose.pose.position.y = self.y
            self.odom.pose.pose.position.z = 0.0
            self.odom.header.stamp = self.current_time #self.get_clock().now().to_msg()

            # CHANGED
            #half_theta = self.theta / 2.0
            #self.q.x = 0.0
            #self.q.y = 0.0
            #self.q.z = math.sin(half_theta)
            #self.q.w = math.cos(half_theta)
            # END
            self.q.w = math.cos(self.theta) # + 3.14159 / 2)
            self.q.x = 0.0
            self.q.y = 0.0
            self.q.z = math.sin(self.theta) # + 3.14159 / 2)            

            self.odom.pose.pose.orientation = self.q
            self.odom.pose.covariance = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                                         0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 
                                         0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.2, 0.0, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.2, 0.0,
                                         0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
#            if(self.linear_velocity is None or self.angular_velocity is None):
#                return
            self.odom.twist.twist.linear.x = float(v) #v
            self.odom.twist.twist.angular.z = float(omega) #omega
            self.odom.twist.covariance = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
                                              0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 
                                              0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.2, 0.0, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.2, 0.0,
                                              0.0, 0.0, 0.0, 0.0, 0.0, 0.2]


        except Exception as e:
            self.get_logger().error(f"An exception occured: {e}") 

#    def publish_dynamic_transform(self):
#        self.t.header.stamp = self.get_clock().now().to_msg() #self.current_time #self.get_clock().now().to_msg()
#        self.t.header.frame_id = "odom"
#        self.t.child_frame_id = "base_link"
#        self.t.transform.translation.x = self.x
#        self.t.transform.translation.y = self.y
#        self.t.transform.translation.z = 0.0 
#        self.t.transform.rotation.x = 0.0 #self.q
#        self.t.transform.rotation.y = 0.0
#        self.t.transform.rotation.z = math.sin(self.theta)
#        self.t.transform.rotation.w = math.cos(self.theta)

#        self.tf_broadcaster.sendTransform(self.t)

    def publish_odometry(self):
        self.publisher_odom.publish(self.odom)

def main(args=None):
    rclpy.init(args=args)
    odom_node = OdomNode()
    twist_thread = threading.Thread(target=odom_node.twist_callback)
    twist_thread.start()
    odom_thread = threading.Thread(target=odom_node.odom_callback)
    odom_thread.start()
#    tf_thread = threading.Thread(target=odom_node.publish_dynamic_transform)
#    tf_thread.start()
    publisher_thread = threading.Thread(target=odom_node.publish_odometry)
    publisher_thread.start()
    try:
        rclpy.spin(odom_node)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        odom_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
