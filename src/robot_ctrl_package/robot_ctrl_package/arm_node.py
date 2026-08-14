from ultralytics import YOLO
import rclpy
import time
import requests
import math 
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Int16, Bool, Float32MultiArray
import sys
import serial
import threading
import json
import cv2
import os

def lerp(a, b, t):
    return a + (b - a) * t
    
def interpolate_stages(stage_a, stage_b, t):
    keys = ["b", "s", "e", "t_joint", "r", "g"]
    return {k: lerp(stage_a[k], stage_b[k], t) for k in keys}


class Arm(Node):
    def __init__(self):
        super().__init__("arm_node")
        self.subscriber_stop = self.create_subscription(Bool, "stop", self.stop_callback, 5)
        self.subscriber_det = self.create_subscription(Bool, "detection_to_arm", self.detection_callback, 5)  

        self.publisher_exec = self.create_publisher(Bool, "executing", 5)

        self.full_stop = False
        self.stop_execution = False
        self.executing = False
        self.detected = False
        self.ready_to_pick = False

        self.frame = None

        self.ser = serial.Serial("/dev/ttyUSB0", baudrate=115200)

        self.lock = threading.Lock()

        self.steps = 20
        self.spd =  0
        self.acc = 10
        self.move_time = 1.0
        self.stages_file = "/home/wb/Desktop/stages.json"
        self.current_stage = 0 
        self.current_step = 0
####
        pkg_share_directory = "/home/wb/Desktop/Drobage/src/my_yolo_package/share"
        defaults = {"model_path" : os.path.join(pkg_share_directory, "models", "best.pt"),
                    "conf":0.2,
                    "max_detections":1,
                    "class_detection":[17], #47
                    "device" : 'cuda'
                    }
        self.params = {}

        for name, value in defaults.items():
            self.declare_parameter(name, value)
            self.params[name] = self.get_parameter(name).value

        self.model_path = self.params["model_path"]
        self.conf = self.params["conf"]
        self.max_detections = self.params["max_detections"]
        self.class_detection = self.params["class_detection"]
        self.device = self.params["device"]

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"{self.model} was loaded succsefully")
        except Exception as e:
            self.get_logger().error(f"Could not load {e}")

####             
        self.timer = self.create_timer(float(self.move_time / self.steps), self.execute_step)
        self.camera_timer = self.create_timer(float(1/10), self.arm_camera)
        with open(self.stages_file) as f:
            self.stages = json.load(f)["stages"]
###
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().info("Cannot open camera")
            exit()

        arm_camera_thread_vis = threading.Thread(target=self.arm_camera_vis, daemon=False)
        arm_camera_thread_vis.start()

        process_thread = threading.Thread(target=self.processing_thread, daemon=True)
        process_thread.start()


#        arm_camera_thread = threading.Thread(target=self.arm_camera, daemon=False)
#        arm_camera_thread.start()


###

    def stop_callback(self, msg):
        self.full_stop = msg.data
        print(msg)

    def detection_callback(self, msg):
        self.detected = msg.data

    def send_command(self, ser, joints, spd, acc):
        cmd = {
            "T": 102,
            "base": joints["b"],
            "shoulder": joints["s"],
            "elbow": joints["e"],
            "wrist": joints["t_joint"],
            "roll": joints["r"],
            "hand": joints["g"],
            "spd": spd,
            "acc": acc,
        }
        self.ser.write(json.dumps(cmd).encode() + b"\n")
    

#    def read_serial(self):
#        while rclpy.ok():
#            data = self.ser.readline().decode('utf-8')
#            if data:
#                print(f"Received: {data}, {len(self.stages)}", end='')
###
    def arm_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Can't receive frame. Exiting...")
            exit()
        with self.lock:
            self.frame = frame.copy()

    def processing_thread(self):
         while rclpy.ok():
             with self.lock:
                 if self.frame is None:
                     continue
                 try:
                     infr_rslts = self.model(
                     source = self.frame,
                     device = self.device,
                     conf = self.conf,
                     classes = self.class_detection,
                     max_det = self.max_detections)

                     infr = infr_rslts[0]
                     x1 = y1 = x2 = y2 = None

                     if infr.boxes is not None and len(infr.boxes) > 0:
                         box = infr.boxes[0]
                         x1, y1,  x2, y2 = map(int, box.xyxy[0])

                         x_center = int((x1 + x2) / 2)
                         y_center = int((y1 + y2) / 2)
                         cv2.circle(self.frame, (x_center, y_center),5, (0, 255, 0), -1)
                         cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                 except Exception as e:
                     self.get_logger().error(f"An exception occured: {e}")

         
        
    def arm_camera_vis(self):
         while rclpy.ok():
             with self.lock:
                 if self.frame is None:
                     frame_to_show = None
                 else:
                     frame_to_show = self.frame.copy()
             if frame_to_show is None:
                 time.sleep(0.01)
                 continue
#             if self.frame is None:
#                 time.sleep(0.01)
#                 continue
#             gray = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2GRAY)
             
             cv2.imshow("frame", frame_to_show)
             if cv2.waitKey(1) == ord("q"):
                 break

###
    def execute_step(self):
        if self.full_stop and self.detected:
            self.ready_to_pick = True ##
        if self.ready_to_pick: ##
            self.executing = True  #
            if self.current_stage >= len(self.stages) or self.stop_execution:
                self.get_logger().info("Trajectory complete")
                self.executing = False #
                self.ready_to_pick = False
                self.current_stage = 0 ##
                self.current_step = 0 ##
                return
        
            if self.current_stage == 0:
                stage = self.stages[self.current_stage]
                self.send_command(self.ser, stage, self.spd, self.acc)
                self.current_stage += 1
                time.sleep(0.5 + self.move_time)
                return

            t = self.current_step / self.steps
            prev = self.stages[self.current_stage - 1]
            curr = self.stages[self.current_stage]
            interpolated = interpolate_stages(prev, curr, t)            
            self.send_command(self.ser, interpolated, self.spd, self.acc)
            self.current_step += 1        

            if self.current_step > self.steps:
                self.current_step = 0
                self.current_stage += 1

        msg_exec = Bool() #
        msg_exec.data = self.executing  #
        self.publisher_exec.publish(msg_exec)   #
        print(f"FULL STOP:{self.full_stop}")
        print(f"DETECTED:{self.detected}")

def main(args=None):
    rclpy.init(args=args)
    arm_node = Arm()

#    arm_camera_thread_vis = threading.Thread(target=arm_node.arm_camera_vis, daemon=False)
#    arm_camera_thread_vis.start()

#    arm_camera_thread = threading.Thread(target=arm_node.arm_camera, daemon=False)
#    arm_camera_thread.start()

#   ser_thread = threading.Thread(target=arm_node.read_serial, daemon=True)
#   ser_thread.start()

    try:
        rclpy.spin(arm_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        arm_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
