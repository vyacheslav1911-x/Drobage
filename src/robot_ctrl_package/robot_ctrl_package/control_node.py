import rclpy
import time
import requests
import math 
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int16, Bool, Float32MultiArray
from enum import Enum, auto
import threading
import serial
import json
import os

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
            control_output = self.kp_side * error_side
            prev_error = self.last_error_side if self.last_error_side is not None else 0
            rate_of_change_side = (error_side - prev_error) / dt
            self.last_error_side = error_side 
            return max(-255, min(255, control_output)), rate_of_change_side


class States(Enum):
    IDLE = auto()
    MOVING = auto()
    CENTERING = auto()
    STOP = auto()

class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp_frwd = 45
        self.Ki_frwd = 40

        self.Kp_side = 0.35  #12 V
        self.Ki_side = 0.2

        self.TARGET_DISTANCE = 0.31
        self.feedforward = 25
        self.detection_count = 0
        self.control_output_side = 0 

        self.ip = "192.168.4.1"
        self.session = requests.Session()
#####
        self.SERIAL_DEV =["/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyACM3"]
        self.ser = None
        self.using_serial = False
        for dev in self.SERIAL_DEV:
            if os.path.exists(dev):
                self.ser = serial.Serial(dev, baudrate=115200)
                self.get_logger().info("Using Serial")
                self.using_serial = True
#####

        self.speed = None

        self.dt = 0.1
        self.last_time = time.monotonic()

        self.should_turn = True
        self.should_stop = False
        self.turning = False
        self.approach_mode = False
        self.full_stop = False
        self.executing = False

        self.state = States.IDLE #APPROACH

        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 1)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 1) 
        self.subscriber_det = self.create_subscription(Bool, "detection", self.detection_callback, 1)        
        self.subscriber_meas_invalid = self.create_subscription(Bool, "meas_invalid", self.invalid_callback, 1)         
        self.subscriber_recovery = self.create_subscription(Bool, "recovery", self.recovery_callback, 1)
        self.subscriber_exec = self.create_subscription(Bool, "executing_xyz",  self.exec_callback, 1)

        self.stop_publisher = self.create_publisher(Bool, "stop", 5)
        self.det_publisher = self.create_publisher(Bool, "detection_to_arm", 5)

        self.rate_of_change = None
        self.rate_of_change_frwd = None
        self.distance_m = None
        self.side_error = None
        self.detected = False #None    

        self.create_timer(float(1/10), self.control_loop)
        self.create_timer(float(1/10), self.first_detection)
        self.create_timer(float(1/10), self.print_timer)
        self.create_timer(float(1/10), self.detection)
        self.create_timer(float(1/10), self.serial_check)

    def serial_check(self):
        for dev in self.SERIAL_DEV:
            try:
                if os.path.exists(dev):
                    self.ser = serial.Serial(dev, baudrate=115200)
                    self.get_logger().info("Using Serial")
                    self.using_serial = True
                else:
                    self.using_serial = False
            except PermissionError as e:
                self.get_logger().info(e)


    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data
        if self.side_error is not None:
            self.side_error = max(-320, min(self.side_error, 320))

    def invalid_callback(self, msg):
        self.meas_invalid = msg.data

    def recovery_callback(self, msg):
        self.recovery = msg.data

    def detection_callback(self, msg):
        self.detected = msg.data

    def exec_callback(self, msg):
        self.executing = msg.data
        print(self.executing)

    def send_command(self, T: int, L_speed: int, R_speed: int) -> None:
        json_command = f'{{"T":{T},"L":{L_speed},"R":{R_speed}}}'
        json_send =  f"http://{self.ip}/js?json={json_command}"
#        json_ser = json.dumps(json_command) ###
        try:
            if self.using_serial:
                self.ser.write((json_command + "\n").encode("utf-8"))
                self.ser.flush()
            if not self.using_serial: 
                self.session.get(json_send, timeout=0.4)
        except Exception as e:
            print("HTTP error: ", e)

    def first_detection(self) -> None:
        if self.detected:
            self.detection_count += 1

    def update_state(self):
        if self.detection_count == 0 and not self.detected:
            self.state = States.IDLE
            
        if self.distance_m is None or self.side_error is None:
            return

        should_move = self.distance_m > self.TARGET_DISTANCE #+ 0.03
        self.control_output_side, self.rate_of_change = self.side_controller.update(self.side_error, self.dt)
 
        if not self.executing:
            self.full_stop = False

        if self.detected:
#            self.approach_mode = False
            self.detection_count += 1       

#        if abs(self.side_error) > 20:
#            self.should_stop = False
#        if self.should_stop:
#            self.turning = False      
        if should_move: #and not self.turning:
            self.state = States.MOVING

#        if not self.should_stop:
#            self.state = States.CENTERING

        if not should_move: #and self.should_stop:
            self.state = States.STOP
            self.full_stop = True 
            self.speed = 20


    def approach(self):
        self.send_command(1, 0, 0)

    def moving(self):
        error = self.distance_m - self.TARGET_DISTANCE
        control_output_frwd, self.rate_of_change_frwd  = self.frwd_controller.update(error, self.dt)
        self.speed = max(20, min(90, control_output_frwd))

        L_speed = self.speed + self.feedforward
        R_speed = self.speed + self.feedforward
        self.send_command(1, L_speed, R_speed)


#    def centering(self):   

#        if -20 < self.side_error < 20:
#            self.should_stop = True
#        if self.should_turn and not self.should_stop:
#            self.turning = True   
#            if self.side_error < 0:
#                L_speed = -abs(self.control_output_side)
#                R_speed = abs(self.control_output_side)
#                self.send_command(1, L_speed, R_speed)
#            if self.side_error > 0:
#                L_speed = abs(self.control_output_side)
#                R_speed = -abs(self.control_output_side)

#                self.send_command(1, L_speed, R_speed)

    def stop(self):
         self.send_command(1, 0, 0)

    def control_loop(self):
         now = time.monotonic()
         self.dt = now - self.last_time
         self.last_time = now
###
         if self.executing:
             print(self.executing)
             return
###
         self.update_state()
         if self.state == States.IDLE:
             self.approach()
         if self.state == States.MOVING:
             self.moving()
         elif self.state == States.CENTERING:
             self.centering()
         elif self.state == States.STOP:
             self.stop()
         try:
             msg_stop = Bool()
             msg_stop.data = self.full_stop
             self.stop_publisher.publish(msg_stop)
         except AssertionError as e:
             self.get_logger().info(e)


    def detection(self):
        msg_det = Bool()
        msg_det.data = bool(self.detected)
        self.det_publisher.publish(msg_det) 

     
    def print_timer(self):
        self.get_logger().info(f"SPEED:{self.speed}")
        self.get_logger().info(f"STATE:{self.state}")
        self.get_logger().info(f"DISTANCE:{self.distance_m}")
        self.get_logger().info(f"SIDE ERROR:{self.side_error}")
        self.get_logger().info(f"CONTROL OUTPUT SIDE:{self.control_output_side}")
        self.get_logger().info(f"DETECTED:{self.detected}")
        print(self.dt)

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
























