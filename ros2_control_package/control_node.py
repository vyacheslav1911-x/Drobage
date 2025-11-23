import cv2
import rclpy
import time
import requests 
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32


class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp = 400
        self.Ki = 8
        self.ip = "192.168.4.1"
        self.integral_error = 0
        self.last_sample = time.monotonic()
        self.is_moving = False
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_err>
        

    def forward_error_callback(self, msg):
        print("inside")
        self.distance_m = msg.data
        TARGET_DISTANCE = 0.25
        should_move = self.distance_m > TARGET_DISTANCE
        if should_move:
            error = self.distance_m - TARGET_DISTANCE
 
            current_sample = time.monotonic()
            dt = current_sample - self.last_sample
            self.last_sample = current_sample

            self.integral_error += error * dt
            control_output = self.Kp * error + self.Ki * self.integral_error
            speed = max(60, min(255, control_output))

            command_move = f'{{"T":11,"L":{int(speed)},"R":{int(speed)}}}'        
            json_move = f"http://{self.ip}/js?json={command_move}"
            try:
                requests.get(json_move, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)
            self.is_moving = True
        elif not should_move and self.is_moving:
            command_stop = '{"T":11,"L":0,"R":0}'
            json_stop =  f"http://{self.ip}/js?json={command_stop}"
            try:
                requests.get(json_stop, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)
            self.is_moving = False
            print("Target reached - stopping the rover")
         
        print(f"{self.distance_m:.2f}")

def main():
    rclpy.init()
    control_node = Control()
    try:
        rclpy.spin(control_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

