  GNU nano 4.8                                                             control_node.py                                                                       
import cv2
import rclpy
import time
import requests
import math 
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int16, Bool

class PIController:
    class Forward:
        def __init__(self, kp, ki):
            self.kp = kp
            self.ki = ki
            self.integral_error = 0 
            self.last_sample = None
            self.last_error_frwd = None
            self.k_decay = 200.0

        def update(self, error):
            self.current_sample = time.monotonic()
            if self.last_sample is None:
                dt = 0.05
            else:
                dt = max(0.001,  self.current_sample - self.last_sample)
            self.last_sample = self.current_sample

            alpha = math.exp(-self.k_decay * dt)
            print(dt)
            self.integral_error += error * 0.033
            self.integral_error = max(-160, min(160, self.integral_error))
            prev_error = self.last_error_frwd if self.last_error_frwd is not None else 0
            rate_of_change_frwd = (error - prev_error) / 0.033
            self.last_error_frwd = error
            control_output = self.kp*error + self.ki * self.integral_error
            print(f"Integral Error Forward: {self.integral_error}")
            print(f"Derivative Forward: {rate_of_change_frwd}")
            print(f"Control output forward: {control_output}")
            return max(-255, min(255, control_output)), rate_of_change_frwd, alpha
    
    class Side:
        def __init__(self, kp_side, ki_side):
            self.kp_side = kp_side
            self.ki_side = ki_side
            self.integral_error_side = 0
            self.last_sample_side = None
            self.last_error_side = None


        def update(self, error_side):
            self.current_sample_side = time.monotonic()
            if self.last_sample_side is None:
                self.dt = 0.1
            else:
                self.dt = max(0.001, self.current_sample_side - self.last_sample_side)
            self.last_sample_side = self.current_sample_side

            self.integral_error_side += error_side * 0.033
            self.integral_error_side = max(-100, min(100, self.integral_error_side))
            control_output = self.kp_side * error_side + self.ki_side * self.integral_error_side 
            prev_error = self.last_error_side if self.last_error_side is not None else 0
            rate_of_change_side = (error_side - prev_error) / 0.033
            self.last_error_side = error_side 
            print(f"Derivative Side: {rate_of_change_side}")

            return max(-255, min(255, control_output)), rate_of_change_side

class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp_frwd = 35
        self.Ki_frwd = 40

        self.Kp_side = 1.2
        self.Ki_side = 1.5
        self.TARGET_DISTANCE = 0.3

        self.ip = "192.168.4.1"
        self.is_moving = False
        self.should_stop = False
        self.should_stop_moving = False
        self.should_back = False
        self.turning = False
        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 5)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 5) 
        self.subscriber_det = self.create_subscription(Bool, "detection", self.detection_callback, 5)
        self.rate_of_change = None
        self.exp_decay = None       
        self.previous = None
        self.distance_m = None
        self.side_error = None    
        self.spike_lock = None       

        self.create_timer(0.1, self.control_loop_frwd)
#        self.create_timer(0.05, self.control_loop_side)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data

    def detection_callback(self, msg):
        self.detected = msg.data
        print(self.detected)        

    def control_loop_frwd(self):
        should_turn = True
        if self.distance_m is None or self.side_error is None:
            return
        if self.spike_lock is not None and time.time() < self.spike_lock:
            print("Going to sleep...")
            print(f"Inside lock: {time.time()}, {self.spike_lock}")
            return
        distance_m_prev = self.previous if self.previous is not None else 0
        if self.distance_m > distance_m_prev*1.25 and distance_m_prev != 0:
            self.spike_lock = time.time() + 0.1
            print(f"Inside condition: {self.spike_lock}")
            return

        self.previous = self.distance_m 
        should_move = self.distance_m > self.TARGET_DISTANCE + 0.03   
        control_output_side, self.rate_of_change = self.side_controller.update(self.side_error)
        if should_move and not self.turning and self.detected:
            self.should_back = False
            error = self.distance_m - self.TARGET_DISTANCE
            control_output_frwd, self.rate_of_change_frwd, self.exp_decay  = self.frwd_controller.update(error)
#            if error < self.distance_m * 0.25:
#                control_output_frwd *= 0.005
            speed = max(20, min(255, control_output_frwd))
            print(f"Speed: {speed}")
            if control_output_side < 0:
                command_frwd = f'{{"T":11,"L":{int(speed+40)},"R":{int(speed+40)}}}'
                json_frwd =  f"http://{self.ip}/js?json={command_frwd}"
            elif control_output_side > 0:
                command_frwd = f'{{"T":11,"L":{int(speed)+40},"R":{int(speed)+40}}}'
                json_frwd =  f"http://{self.ip}/js?json={command_frwd}"
            try:
                requests.get(json_frwd, timeout=0.1)
            except Exception as e:
                print("HTTP error1: ", e)

        if not should_move: 
            if abs(self.side_error) > 10:
                self.should_stop = False
            if -10 < self.side_error < 10:
                command_stop = '{"T":11,"L":0,"R":0}'
                json_stop =  f"http://{self.ip}/js?json={command_stop}"
                self.should_stop = True
                try:
                    requests.get(json_stop, timeout=0.1)
                except Exception as e:
                    print("HTTP error2: ", e)

            if not self.detected:
                command_stop = '{"T":11,"L":0,"R":0}'
                json_stop =  f"http://{self.ip}/js?json={command_stop}"
                self.should_stop_moving = True
                try:

            if should_turn and not self.should_stop:
                self.turning = True   
                if control_output_side < 0:
                    command_move = f'{{"T":11,"L":{-120},"R":{120}}}'        
                    json_move = f"http://{self.ip}/js?json={command_move}"
                    try:
                        requests.get(json_move, timeout=0.1)

                    except Exception as e:
                        print("HTTP error4: ", e)           
                if control_output_side > 0:
                    command_move = f'{{"T":11,"L":{120},"R":{-120}}}'        
                    json_move = f"http://{self.ip}/js?json={command_move}"
                    try:
                        requests.get(json_move, timeout=0.1)
                    except Exception as e:
                        print("HTTP error5: ", e)

            self.turning = False    
        print(f"Distance: {self.distance_m:.2f}")
        print(f"Side control magnitude: {control_output_side}")
        print(f"Should stop: {self.should_stop_moving}")   
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
















