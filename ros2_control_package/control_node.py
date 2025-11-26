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
            self.last_error_side = None


        def update(self, error_side):
            self.current_sample_side = time.monotonic()
            if self.last_sample_side is None:
                self.dt = 0.1
            else:
                self.dt = max(0.001, self.current_sample_side - self.last_sample_side)
            self.last_sample_side = self.current_sample_side

            self.integral_error_side += error_side * self.dt
           self.integral_error_side = max(-200, min(200, self.integral_error_side))
            control_output = self.kp_side * error_side + self.ki_side * self.integral_error_side 
            self.prev_error = self.last_error_side if self.last_error_side is not None else 0
            self.rate_of_change = (error_side - self.prev_error) / self.dt
            self.last_error_side = error_side 
            print(f"Derivative: {self.rate_of_change}")
            return max(-255, min(255, control_output)), self.rate_of_change

class Control(Node):
    def __init__(self):
        super().__init__("control_node")
        self.Kp_frwd = 1.5
        self.Ki_frwd = 1

        self.Kp_side = 0.7
        self.Ki_side = 0.5
        self.TARGET_DISTANCE = 0.25

        self.ip = "192.168.4.1"
        self.is_moving = False
        self.should_turn = True
        self.should_stop = False
        self.frwd_controller = PIController.Forward(self.Kp_frwd, self.Ki_frwd)
        self.side_controller = PIController.Side(self.Kp_side, self.Ki_side) 
        self.subscriber_frwd_dist = self.create_subscription(Float32, "forward_distance", self.forward_error_callback, 5)
        self.subscriber_side_error = self.create_subscription(Int16, "side_error", self.side_error_callback, 5) 
        self.rate_of_change = None       

        self.distance_m = None
        self.side_error = None    

        self.create_timer(0.1, self.control_loop_frwd)
#        self.create_timer(0.05, self.control_loop_side)

    def forward_error_callback(self, msg):
        self.distance_m = msg.data

    def side_error_callback(self, msg):
        self.side_error = msg.data


    def control_loop_frwd(self):
        should_turn = True

        if self.distance_m is None or self.side_error is None:
            return
        should_move = self.distance_m > self.TARGET_DISTANCE
        control_output_side, self.rate_of_change = self.side_controller.update(self.side_error)
        if should_move:
            error = self.distance_m - self.TARGET_DISTANCE
            control_output_frwd = self.frwd_controller.update(error)
            speed = max(80, min(255, control_output_frwd))
            if control_output_side < 0:
                command_frwd = f'{{"T":11,"L":{int(speed)},"R":{int(speed)}}}'
                json_frwd =  f"http://{self.ip}/js?json={command_frwd}"
            elif control_output_side > 0:
                command_frwd = f'{{"T":11,"L":{int(speed)},"R":{int(speed)}}}'
                json_frwd =  f"http://{self.ip}/js?json={command_frwd}"
            try:
                requests.get(json_frwd, timeout=0.1)
            except Exception as e:
                print("HTTP error: ", e)
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
                    print("HTTP error: ", e)

            if should_turn and not self.should_stop:    
                if control_output_side < 0:
                    command_move = f'{{"T":11,"L":{-180},"R":{180}}}'        
                    json_move = f"http://{self.ip}/js?json={command_move}"
                    try:
                        requests.get(json_move, timeout=0.1)
                    except Exception as e:
                        print("HTTP error: ", e)

                if control_output_side > 0:
                    command_move = f'{{"T":11,"L":{110},"R":{-110}}}'        
                    json_move = f"http://{self.ip}/js?json={command_move}"
                    try:
                        requests.get(json_move, timeout=0.1)
                    except Exception as e:
                        print("HTTP error: ", e)



#        elif (not should_move and self.is_moving):
#            command_stop = '{"T":11,"L":0,"R":0}'
#            json_stop =  f"http://{self.ip}/js?json={command_stop}"
#            try:
#                requests.get(json_stop, timeout=0.1)
#            except Exception as e:
#                print("HTTP error: ", e)
#            self.is_moving = False
#            self.is_stopped = True
#            print("Target reached - stopping the rover")
#            if self.is_stopped:
#                print(self.rate_of_change)
#                if (-11 < self.rate_of_change < 11) and (-5 < self.side_error < 5):
#                    return
#                if self.side_error < 0:
#                    command_reverse_left = f'{{"T":11,"L":{-150},"R":{150}}}'
#                    json_reverse_left = f"http://{self.ip}/js?json={command_reverse_left}"
#                    try:
#                        requests.get(json_reverse_left, timeout=0.1)
#                    except Exception as e:
#                        print("HTTP error: ", e)
#                elif self.side_error > 0:
#                    command_reverse_right = f'{{"T":11,"L":{150},"R":{-150}}}'
#                    json_reverse_right = f"http://{self.ip}/js?json={command_reverse_right}"
#                    try:
#                        requests.get(json_reverse_right, timeout=0.1)
#                    except Exception as e:
#                        print("HTTP error: ", e)
  

        print(f"Distance: {self.distance_m:.2f}")
        print(f"Side control magnitude: {control_output_side}")
#    def control_loop_side(self):
#        if self.side_error is None:
#            return
#        control_output = self.side_controller.update(self.side_error)
#        if control_output < 0:
#            command_turn_left = f'{{"T":11,"L":0,"R":{int(abs(control_output))}}}'
#            json_turn_left = f"http://{self.ip}/js?json={command_turn_left}"
#            try:
#                requests.get(json_turn_left, timeout=0.1)
#            except Exception as e:
#                print("HTTP error: ", e)
#        elif control_output > 0:
#            command_turn_right = f'{{"T":11,"L":{control_output},"R":0}}'
#            json_turn_right = f"http://{self.ip}/js?json={command_turn_right}"
#            try:
#                requests.get(json_turn_right, timeout=0.1)
#            except Exception as e:
#                print("HTTP error: ", e)
#        elif -10 < self.side_error < 10:
#            command_stop = '{"T":11,"L":0,"R":0}'
#            json_stop =  f"http://{self.ip}/js?json={command_stop}"
#            try:
#                requests.get(json_stop, timeout=0.1)
#            except Exception as e:
#                print("HTTP error: ", e)
#        
#        print(f"Side error: {self.side_error}")
#        print(f"Control output: {control_output}")
#        control_output = 0

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










