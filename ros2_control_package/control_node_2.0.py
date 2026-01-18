import rclpy
import time
import requests
import math 
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int16, Bool, Float32MultiArray
from enum import Enum, auto

class PIController:
    class Forward:
        def __init__(self, kp: float, ki: float):
            self.kp = kp
            self.ki = ki
            self.integral_error = 0 
            self.last_sample = None
            self.last_error_frwd = None

        def update(self, error: float, dt:float) -> (float, float):
            self.integral_error += error * dt
            self.integral_error = max(-120, min(120, self.integral_error))
            prev_error = self.last_error_frwd if self.last_error_frwd is not None else 0
            rate_of_change_frwd = (error - prev_error) / dt
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


        def update(self, error_side: float, dt: float) -> (float, float):
            self.integral_error_side += error_side * dt
            self.integral_error_side = max(-100, min(100, self.integral_error_side))
            control_output = self.kp_side * error_side + self.ki_side * self.integral_error_side 
            prev_error = self.last_error_side if self.last_error_side is not None else 0
            rate_of_change_side = (error_side - prev_error) / dt
            self.last_error_side = error_side 
            print(f"Derivative Side: {rate_of_change_side}")
            return max(-255, min(255, control_output)), rate_of_change_side


class States(Enum):
    APPROACH = auto()
    MOVING = auto()
    CENTERING = auto()
    SEARCH = auto()
    LOCKED = auto()
    STOP = auto()

class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp_frwd = 40
        self.Ki_frwd = 45

        self.Kp_side = 0.3  #12 V
        self.Ki_side = 0.2

        self.TARGET_DISTANCE = 0.3
        self.d_dec = 0.34
        self.feedforward = 25
        self.detection_count = 0
        self.confidence = 0
        self.control_output_side = 0        

        self.ip = "192.168.4.1"
        self.speed = None
        self.counter = 0
        self.lock_counter = 0
        self.stuck = 0

        self.left_counter_search = 0
        self.right_counter_search = 0
        self.center_counter_search = 0 
        self.search_counter = 0

        self.dt = 0.033
        self.last_time = 0

        self.should_turn = True
        self.should_stop = False
        self.turning = False
        self.done = False
        self.search_mode = False
        self.approach_mode = False
        self.was_locked = False
        self.stop_all = False

        self.state = States.APPROACH

        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 5)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 5) 
        self.subscriber_det = self.create_subscription(Bool, "detection", self.detection_callback, 5)        

        self.rate_of_change = None
        self.rate_of_change_frwd = None
        self.previous = None
        self.distance_m = None
        self.side_error = None
        self.detected = None    
        self.spike_lock = None       

        self.create_timer(0.033, self.control_loop)
        self.create_timer(0.033, self.first_detection)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data
        self.side_error = max(-320, min(self.side_error, 320))
        print(self.side_error)

    def detection_callback(self, msg):
        self.detected = msg.data
        if self.detection_count == 0 and not self.detected:
            print(self.detection_count)
            self.approach_mode = True
            print(self.approach_mode)
            self.send_command(11, 70, 70)
        print(f"DETECTED OR NOT: {self.detected}")


    def send_command(self, T: int, L_speed: int, R_speed: int) -> None:
        json_command = f'{{"T":{T},"L":{L_speed},"R":{R_speed}}}'
        json_send =  f"http://{self.ip}/js?json={json_command}"
        try:
            requests.get(json_send, timeout=0.1)
        except Exception as e:
            print("HTTP error: ", e)

    def first_detection(self) -> None:
        if self.detected:
            self.detection_count += 1

    def update_state(self):
        if self.search_mode and not self.approach_mode:
            self.state = States.SEARCH
 
        if self.lock():
            self.lock_counter += 1
            self.state = States.LOCKED

        if self.state == States.LOCKED:
            self.stuck += 1  
 
        if self.stuck > 150:
            self.search_mode = True
        if self.lock_counter > 30:
            self.search_mode = True

        if self.detected:
            self.approach_mode = False
            self.search_mode = False
            self.detection_count += 1       
            self.lock_counter = 0
            self.stuck = 0

        if self.approach_mode:
            self.state = States.APPROACH
        should_move = self.distance_m > self.TARGET_DISTANCE + 0.03
        self.control_output_side, self.rate_of_change = self.side_controller.update(self.side_error, self.dt)
        if abs(self.side_error) > 30:
            self.should_stop = False
#        if self.should_stop and should_move and not self.detected and not self.stop_all:
#            self.approach_mode = True
        if self.should_stop:
            self.turning = False      
#        if should_move and not self.turning and self.detected:
#            self.stop_all = False
#            self.state = States.MOVING

        if self.detected and not self.should_stop or not should_move and self.detected and not self.lock():
            self.state = States.CENTERING
        if not should_move and self.should_stop:
            self.state = States.STOP
            self.stop_all  = True
        if self.stop_all and not self.detected or self.stop_all and self.detected:
            self.state = States.STOP
        if should_move and not self.turning and self.detected:
            self.stop_all = False
            self.state = States.MOVING


    def lock(self) -> Bool: 
        if self.spike_lock is not None:
            print("Going to sleep...")
            if self.counter < self.spike_lock:
                self.counter = self.counter + 1
                self.send_command(11, 0, 0)
                print(self.rate_of_change_frwd)
                print(f"COUNTER: {self.counter}")
                return True
            self.previous = self.distance_m
        self.derivative = self.rate_of_change_frwd
        print(self.derivative)
        distance_m_prev = self.previous if self.previous is not None else 0
        if self.distance_m > distance_m_prev*1.25 and distance_m_prev != 0  or (self.derivative is not None and abs(self.derivative) > 1):
            self.spike_lock = 10
            self.counter = 0 
            print(self.distance_m)
            print(distance_m_prev)
            print(self.previous) 
            print(f"Inside condition: {self.spike_lock}")
            return True

        self.counter = 0
        self.spike_lock = None           
        self.previous = self.distance_m
        print(self.distance_m)
        print(distance_m_prev) 
        return False

    def approach(self):
        self.send_command(11, 70, 70)

    def moving(self):
        error = self.distance_m - self.TARGET_DISTANCE
        control_output_frwd, self.rate_of_change_frwd  = self.frwd_controller.update(error, self.dt)
        print(f"FRWD DERIVATIVE:{self.rate_of_change_frwd}")
        if self.stop_all:
            control_output_frwd = 20  
        self.speed = max(20, min(90, control_output_frwd))
        print(f"SPEED: {self.speed}")
        if self.distance_m <= self.d_dec or self.rate_of_change_frwd > 1:
            self.speed = 0

        L_speed = self.speed + self.feedforward
        R_speed = self.speed + self.feedforward
        self.send_command(11, L_speed, R_speed)


    def search(self):
        if self.left_counter_search < 60:
            self.send_command(11, -100, 100)
            self.left_counter_search += 1
        if self.left_counter_search == 60:
            if self.right_counter_search <= 120:
                self.send_command(11, 100, -100)  
                self.right_counter_search += 1
        if self.right_counter_search == 120:
            if self.center_counter_search < 60:
                self.send_command(11, -100, 100)
                self.center_counter_search += 1

            print(self.left_counter_search) 

    def centering(self):   
        print(f"SIDE ERROR:{self.side_error}")
        control_output_side, self.rate_of_change = self.side_controller.update(self.side_error, self.dt)

        if -30 < self.side_error < 30:
            self.should_stop = True
        if self.should_turn and not self.should_stop:
            self.turning = True   
            if self.control_output_side < 0:
                L_speed = -100
                R_speed = 100
                self.send_command(11, L_speed, R_speed)
            if self.control_output_side > 0:
                L_speed = 100
                R_speed = -100
                self.send_command(11, L_speed, R_speed)

    def stop(self):
         self.send_command(11, 0, 0)

    def control_loop(self):
         now = time.monotonic()
         self.dt = now - self.last_time
         self.last_time = now
         if self.distance_m is None or self.side_error is None:
             return
         print(f"LOCK COUNTR:{self.lock_counter}")
         print(self.confidence)
         print(self.state)
         self.update_state()
         if self.state == States.LOCKED:
             return
         elif self.state == States.APPROACH:
             self.approach()
         elif self.state == States.MOVING:
             self.moving()
         elif self.state == States.SEARCH:
             self.search()
         elif self.state == States.CENTERING:
             self.centering()
         elif self.state == States.STOP:
             self.stop()
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






