import cv2
import rclpy
import time
import requests 
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int16

class PIController:
    class Forward:
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
    
    class Side:
        def __init__(self, kp_side, ki_side):
            self.kp_side = kp_side
            self.ki_side = ki_side
            self.integral_error_side = 0
            self.last_sample_side = None


        def update(self, error_side):
            self.current_sample_side = time.monotonic()
            if self.last_sample_side is None:
                dt = 0.05
            else:
                dt = max(0.001, self.current_sample_side - self.last_sample_side)
            self.last_sample_side = self.current_sample_side

            self.integral_error_side += error_side * dt
            self.integral_error_side = max(-120, min(120, self.integral_error_side))

            control_output = self.kp_side * error_side + self.ki_side * self.integral_error_side
            return max(-255, min(255, control_output))

class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp_frwd = 400
        self.Ki_frwd = 8

        self.Kp_side = 1.0
        self.Ki_side = 0.6

        self.ip = "192.168.4.1"
        self.is_moving = False
        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 10)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 10) 

        self.distance_m = None
        self.side_error = None    

        self.create_timer(0.05, self.control_loop_frwd)
        self.create_timer(0.05, self.control_loop_side)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data


    def control_loop_frwd(self):
        if self.distance_m is None:
            return
        TARGET_DISTANCE = 0.25
        should_move = self.distance_m > TARGET_DISTANCE

        if should_move:
            error = self.distance_m - TARGET_DISTANCE

            control_output = self.frwd_controller.update(error)
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

    def control_loop_side(self):
        if self.side_error is None:
            return
        control_output = self.side_controller.update(self.side_error)
        if control_output < 0:
            command_turn_left = f'{{"T":11,"L":0,"R":{int(abs(control_output))}}}'
            json_turn_left = f"http://{self.ip}/js?json={command_turn_left}"
            try:
                requests.get(json_turn_left, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)
        elif control_output > 0:
            command_turn_right = f'{{"T":11,"L":{control_output},"R":0}}'
            json_turn_right = f"http://{self.ip}/js?json={command_turn_right}"
            try:
                requests.get(json_turn_right, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)
        elif -10 < self.side_error < 10:
            command_stop = '{"T":11,"L":0,"R":0}'
            json_stop =  f"http://{self.ip}/js?json={command_stop}"
            try:
                requests.get(json_stop, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)

        print(f"Side error: {self.side_error}")
        print(f"Control output: {control_output}")
        control_output = 0

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







