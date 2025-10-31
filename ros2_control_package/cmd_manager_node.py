import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32
import time
#todo
TIME_PERIOD = 0.2
I = 0
_time = 0
last_error = 0
const_limit = 255
# PI control
def PI(K, Ki,Kd, x_desired, x_actual):

    global I, _time, last_error, const_limit
    now = time.monotonic()
    if (_time == 0):
        dt = 0.05
    else:
        dt = max(1e-3, (now - _time))
    _time = now
    e = (x_desired - x_actual)

    if (e > const_limit):
        e = const_limit
    if (e < -const_limit):
        e = -const_limit
    if (e < 5 and e > -5):
        e = 0
        I = 0


    # коли збільшення "газу" не спричинить халепи , то тільки тоді ми його збільшуємо
    # when increasing the tork does not cause a trouble, only then we change the

    u_ = Ki * I + K * e
    # checking if the theoretical PI sygnal is bigger that possible signal if so variable would_sat = True
    would_sat = abs(u_) > const_limit
    # checking if the sing of the error and theoretical PI sygnal are the same. It is bad for us in case of big tork , wich is checked in would_sat
    push_worse = (u_ * e) > 0
    if not (would_sat and push_worse):
        I += e * dt
    if (dt>0):
        e_d = (e - last_error)/dt
    else:
        e_d =0
    D = Kd*e_d


    u = Ki * I + K * e + D
    u = max(-const_limit,min(const_limit, u))
    last_error = e

    return u


class ManagerNode(Node):
    def __init__(self):
        super().__init__('robot_ctrl_manager')
        #coefficient of scaling the error message
        self.declare_parameter('side_k', 0.9)
        self.declare_parameter('side_ki', 0.1)
        self.declare_parameter('side_kd', 0.7)
        self.k_s = self.get_parameter('side_k').get_parameter_value().double_value
        self.ki_s = self.get_parameter('side_ki').get_parameter_value().double_value
        self.kd_s = self.get_parameter('side_kd').get_parameter_value().double_value

        self.pending_message = False
        self.timer = self.create_timer(TIME_PERIOD, self.timer_callback)
        self.subscription_ = self.create_subscription(Int32MultiArray, 'yolo_detections_coords', self.listener_callback, 10)
        self.publisher_side = self.create_publisher(Int32, 'side_ctrl', 10)
        self.publisher_fwd = self.create_publisher(Int32, 'frwrd_back_ctrl', 10)

    def timer_callback(self):
        if(self.pending_message == True):
            self.pending_message = False
        else:
            self.side_int = int(round(PI(0, 0, 0, 0, 0)))
            side_msg = Int32()
            side_msg.data = self.side_int
            self.publisher_side.publish(side_msg)
            self.get_logger().info(f"Forward: {1488} \n Side: {self.side_int}")

    def listener_callback(self, msg: Int32MultiArray):

        #msg is an array with 2 int values which we send in yolo_node_r node
        #we are normalizing that int so it fits to our robot acceptable command scale(-255 to 255)
        self.pending_message = True
        self.side_int = int(round(PI(self.k_s,self.ki_s,self.kd_s, 0, int((msg.data[0]-320)))))#if minus error-> turn left if + right

        #moving forward or backward error
        #TO/DO
        self.frwrd_back_int = msg.data[1]
        side_msg = Int32()
        frwrd_back_msg = Int32()
        frwrd_back_msg.data = self.frwrd_back_int
        side_msg.data = self.side_int
        self.get_logger().info(f"Forward: {self.frwrd_back_int} \n Side: {self.side_int}")
        #.data because the Int32 is not a python Int it
        self.publisher_side.publish(side_msg)
        self.publisher_fwd.publish(frwrd_back_msg)

def main(args=None):
    rclpy.init(args=args)
    robot_ctrl_manager = ManagerNode()
    try:
        rclpy.spin(robot_ctrl_manager)
    except(SystemExit, KeyboardInterrupt):
        pass
    robot_ctrl_manager.destroy_node()
    rclpy.shutdown()
if __name__ =='__main__':
    main()