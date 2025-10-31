#ros2 run robot_ctrl_package http_esp_node
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import math
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
def reduce_side_error(pwm, ip):
    if(pwm>0):
                command = f'{{"T":11,"L":0,"R":{(abs(pwm))}}}'
    elif(pwm <0):
                command = f'{{"T":11,"L":{(abs(pwm))},"R":0}}'
    else:
                command = f'{{"T":11,"L":0,"R":0}}'
    url = "http://" + ip + "/js?json=" + command
    try:

        response = requests.get(url, timeout=0.5)

        response.raise_for_status()
        content = response.text
        return content


    except (RequestException, Timeout, ConnectionError) as e:
        # this will catch timeouts, connection failures
        return f"HTTP Error: {e}"

class HTTPNode(Node):
    def __init__(self):
        super().__init__('http_esp_node')
        self.declare_parameter('ip_addr', '192.168.4.1')
        self.ip_addr = self.get_parameter('ip_addr').get_parameter_value().string_value
        self.subscription = self.create_subscription(Int32,'side_ctrl', self.send_side_callback, 10)
        self.subscription = self.create_subscription(Int32,'frwrd_back_ctrl', self.send_frwrd_back_callback, 10)
    def send_side_callback(self, msg:Int32):
        self.side_cmd = msg.data
        content= reduce_side_error(self.side_cmd, self.ip_addr)
        self.get_logger().info(f"Response: {content}")
    def send_frwrd_back_callback(self, msg:Int32):
        self.frwrd_cmd = msg.data
def main(args=None):
    rclpy.init(args=args)
    http_esp_node = HTTPNode()
    try:
        rclpy.spin(http_esp_node)
    except(SystemExit, KeyboardInterrupt):
        pass
    http_esp_node.destroy_node()
    rclpy.shutdown()
if __name__ =='__main__':
    main()