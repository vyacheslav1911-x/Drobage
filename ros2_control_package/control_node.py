import cv2
import rclpy
import time
import requests 
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32

class PIController:
    def __init__(self, kp, ki):
        self.kp = kp
        self.ki = ki
        self.integral_error = 0 
        self.last_sample = None

    def update(self, error):
        self.current_sample = time.monotonic()
        if self.last_sample is None:
            dt = 0.05
        else:
            dt = max(0.001,  self.current_sample - self.last_sample)
        self.last_sample = self.current_sample

        self.integral_error += error * dt
        self.integral_error = max(-255, min(255, self.integral_error))

        control_output = self.kp*error + self.ki * self.integral_error
        return max(-255, min(255, control_output))


class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp = 400
        self.Ki = 8
        self.ip = "192.168.4.1"
        self.is_moving = False
        self.controller = PIController(self.Kp, self.Ki) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 10)
        self.distance_m = None
        self.create_timer(0.05, self.control_loop)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data


    def control_loop(self):
        if self.distance_m is None:
            return
        TARGET_DISTANCE = 0.25
        should_move = self.distance_m > TARGET_DISTANCE

        if should_move:
            error = self.distance_m - TARGET_DISTANCE

            control_output = self.controller.update(error)
            speed = max(80, min(255, control_output))

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





