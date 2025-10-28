import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import math

#################### ADD ROUND IN PI CMD_NODE


import requests
def reduce_side_error(pwm, ip):
    if(pwm<0):
                command = f'{{"T":11,"L":0,"R":{(abs(pwm))}}}' 
    elif(pwm >0):
                command = f'{{"T":11,"L":{(abs(pwm))},"R":0}}' 
    else:
                command = f'{{"T":11,"L":0,"R":0}}' 
    url = "http://" + ip + "/js?json=" + command
    response = requests.get(url)
    content = response.text
    return content

class HTTPNode(Node):
    super().__init__('http_esp_node')
    self.declare_parameter('ip_addr', '192.168.4.1')
    self.ip_addr = self.get_parameter('ip_addr').get_parameter_value().string_value
    self.subscription = self.create_subscription(Int32,'IDONTREMEMBER', self.send_side_callback, 10)
    self.subscription = self.create_subscription(Int32,'IDONTR', self.send_frwrd_back_callback, 10)
    def send_side_callback(self, msg:Int32):
        self.side_cmd = msg
        content= reduce_side_error(self.side_cmd)
        self.get_logger().info(f"Response: {content}")
    def send_frwrd_back_callback(self, msg:Int32):
        self.frwrd_cmd = msg
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
