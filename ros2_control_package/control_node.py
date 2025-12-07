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
        def __init__(self, kp: float, ki: float):
            self.kp = kp
            self.ki = ki
            self.integral_error = 0 
            self.last_sample = None
            self.last_error_frwd = None
            
        def update(self, error: float, detected: bool):
#            self.current_sample = time.monotonic()
#            if self.last_sample is None:
#                dt = 0.05
#            else:
#                dt = max(0.001,  self.current_sample - self.last_sample)
#            self.last_sample = self.current_sample
            self.integral_error += error * 0.033
            self.integral_error = max(-120, min(120, self.integral_error))
            prev_error = self.last_error_frwd if self.last_error_frwd is not None else 0
            rate_of_change_frwd = (error - prev_error) / 0.033
            self.last_error_frwd = error
            control_output = self.kp*error + self.ki * self.integral_error
            print(f"Integral Error Forward: {self.integral_error}")
            print(f"Derivative Forward: {rate_of_change_frwd}")
            print(f"Control output forward: {control_output}")
            return max(-255, min(255, control_output)), rate_of_change_frwd
    
    class Side:
        def __init__(self, kp_side: float, ki_side: float):
            self.kp_side = kp_side
            self.ki_side = ki_side
            self.integral_error_side = 0
            self.last_sample_side = None
            self.last_error_side = None


        def update(self, error_side: float):
#            self.current_sample_side = time.monotonic()
#            if self.last_sample_side is None:
#                self.dt = 0.1
#            else:
#                self.dt = max(0.001, self.current_sample_side - self.last_sample_side)
#            self.last_sample_side = self.current_sample_side

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

        self.Kp_side = 0.3  #12 V
        self.Ki_side = 0.2

        self.TARGET_DISTANCE = 0.3
        self.feedforward = 40

        self.ip = "192.168.4.1"
        self.speed = None

        self.should_turn = True
        self.should_stop = False
        self.turning = False

        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 5)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 5) 
        self.subscriber_det = self.create_subscription(Bool, "detection", self.detection_callback, 5)

        self.rate_of_change = None
        self.previous = None
        self.distance_m = None
        self.side_error = None
        self.detected = None    
        self.spike_lock = None       

        self.create_timer(0.033, self.control_loop)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data

    def detection_callback(self, msg):
        self.detected = msg.data
        if not self.detected:
            self.send_command(11,0,0)
        print(f"DETECTED OR NOT: {self.detected}")        

    def send_command(self, T: int, L_speed: int, R_speed: int) -> None:
        json_command = f'{{"T":{T},"L":{L_speed},"R":{R_speed}}}'
        json_send =  f"http://{self.ip}/js?json={json_command}"
        try:
            requests.get(json_send, timeout=0.1)
        except Exception as e:
            print("HTTP error: ", e)

    def lock(self) -> Bool:
         if self.spike_lock is not None and time.time() < self.spike_lock:
             print("Going to sleep...")
             print(f"Inside lock: {time.time()}, {self.spike_lock}")
             return True
         distance_m_prev = self.previous if self.previous is not None else 0
         if self.distance_m > distance_m_prev*1.25 and distance_m_prev != 0:

             self.spike_lock = time.time() + 0.01
             print(f"Inside condition: {self.spike_lock}")
             return True           
         self.previous = self.distance_m 
         return False

    def control_loop(self):
        if self.distance_m is None or self.side_error is None:
            return
#        if self.lock():
#            return

        should_move = self.distance_m > self.TARGET_DISTANCE + 0.03   
        control_output_side, self.rate_of_change = self.side_controller.update(self.side_error)
        if should_move and not self.turning and self.detected:
            error = self.distance_m - self.TARGET_DISTANCE
            control_output_frwd, self.rate_of_change_frwd  = self.frwd_controller.update(error, self.detected)
            self.speed = max(20, min(255, control_output_frwd))
            print(f"Speed: {self.speed}")
            if control_output_side < 0:
                 L_speed = self.speed + self.feedforward
                 R_speed = self.speed + self.feedforward
            elif control_output_side > 0:
                 L_speed = self.speed + self.feedforward
                 R_speed = self.speed + self.feedforward
            self.send_command(11, L_speed, R_speed)

        if not should_move: 
            if abs(self.side_error) > 10:
                self.should_stop = False
            if -10 < self.side_error < 10:
                self.should_stop = True
                self.send_command(11, 0, 0)
#            if not self.detected:
#                print("INSIDE")
#                self.send_command(11, 0, 0)

            if self.should_turn and not self.should_stop:
                self.turning = True   
                if control_output_side < 0:
                    L_speed = -120
                    R_speed = 120
                    self.send_command(11, L_speed, R_speed)
                if control_output_side > 0:
                    L_speed = 120
                    R_speed = -120
                    self.send_command(11, L_speed, R_speed)
                               
            self.turning = False  
         
        print(f"Distance: {self.distance_m:.2f}")
        print(f"Side control magnitude: {control_output_side}")
#        print(f"Should stop: {self.should_stop_moving}")   

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






















