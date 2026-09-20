#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from roarm_msgs.srv import MoveToXYZ
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    RobotState,
)
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState  #change
import threading
import math
import time

class MoveToXYZServer(Node):
    def __init__(self):
        super().__init__('xyz_control')
        self.last_xyz = {"x":0.0, "y":0.0, "z":0.0}
###
        self.x = None
        self.y = None
        self.z = None

        self.last_x = None
        self.last_y = None
        self.last_z = None

        self.executing = False 
        self.detected = False
        self.full_stop = False
        self.counter = 0
###
        self.cb_group = ReentrantCallbackGroup()
###
        self.subscriber_object_coords = self.create_subscription(Point, "object_coords", self.object_coords_callback, 1)
        self.subscriber_det = self.create_subscription(Bool, "detection", self.detection_callback,  1)
        self.subscriber_stop = self.create_subscription(Bool, "stop", self.stop_callback, 1)

        # IK seed. Without this KDL starts from an arbitrary guess and can  #change
        # return a valid but wildly different branch (arm folded backwards,  #change
        # base and wrist each 180 degrees round), which shows up as the arm  #change
        # unwinding a full turn between two nearby targets.  #change
        self.current_joint_state = None  #change
        self.subscriber_joints = self.create_subscription(  #change
            JointState, "joint_states", self.joint_state_callback, 10)  #change

        self.publisher_exec = self.create_publisher(Bool, "executing_xyz", 5)

        self.create_timer(0.5, self.control_loop)
        self.create_timer(0.5, self.publisher_exec_timer)
###
        self.move_action_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.cb_group
        )

        self.get_logger().info('Waiting for MoveGroup action server...')
        self.move_action_client.wait_for_server()
        self.get_logger().info('Connected to move_group')
       
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            callback_group = self.cb_group
        )

        self.get_logger().info('Waiting for GripperCommand action server...')
        self.gripper_client.wait_for_server()
        self.get_logger().info('Connected to GripperCommand')

        self.ik_client = self.create_client(
            GetPositionIK,
            'compute_ik',
            callback_group=self.cb_group
        )

        self.get_logger().info('Waiting for compute_ik service...')  #change
        self.ik_client.wait_for_service()  #change
        self.get_logger().info('Connected to compute_ik')  #change

        self.pick_srv = self.create_service(
            MoveToXYZ,
            'pick_sequential',
            self.handle_pick_sequential,
            callback_group=self.cb_group
        )
        self.get_logger().info('Service /pick_seqential  ready')

        # Single move, no grasp — lets you command an XYZ target directly:  #change
        #   ros2 service call /move_to_xyz roarm_msgs/srv/MoveToXYZ "{x: 0.2, y: 0.0, z: 0.15}"  #change
        self.move_srv = self.create_service(  #change
            MoveToXYZ,  #change
            'move_to_xyz',  #change
            self.handle_move_request,  #change
            callback_group=self.cb_group  #change
        )  #change
        self.get_logger().info('Service /move_to_xyz ready')  #change
###
    def object_coords_callback(self, msg: Point):
        self.x = msg.x
        self.y = -msg.y
        self.z = msg.z
        print(self.x, self.y, self.z)

    def joint_state_callback(self, msg: JointState):  #change
        self.current_joint_state = msg  #change

    def detection_callback(self, msg: Bool):
        self.detected = bool(msg.data)

    def stop_callback(self, msg):
        self.full_stop = msg.data


    def publisher_exec_timer(self):
        msg_exec = Bool()
        msg_exec.data = self.executing
        self.publisher_exec.publish(msg_exec)

    def control_loop(self):
#
        if self.full_stop:
           self.counter+=1

        if not self.full_stop:
           self.counter = 0
#
        if self.counter > 3 and self.detected:
#            time.sleep(1.0)
            if self.executing is True or self.x is None or self.y is None or self.z is None:
                return
            self.executing = True
            req = MoveToXYZ.Request()
            if self.last_x is not None:       
                if self.last_x - 0.015 <= self.x <= self.last_x + 0.015 and self.last_y - 0.005 <= self.y <= self.last_y + 0.005 and self.last_z - 0.002 <= self.z <= self.last_z + 0.002:
#                    self.x = self.last_x
#                    self.y = self.last_y
#                    self.z = self.last_z
                    self.executing = False
                    return                   
                if not self.detected:
                    self.x = self.last_x
                    self.y = self.last_y
                    self.z = self.last_z
                    self.executing = False
                    return                
            req.x = self.x
            req.y = self.y
            req.z = self.z

            self.last_x = self.x
            self.last_y = self.y
            self.last_z = self.z

            service_call_obj = self.create_client(MoveToXYZ, "pick_sequential").call_async(req)
            service_call_obj.add_done_callback(self.done_callback)
        else:
            return 

    def done_callback(self, service_call_obj):
        try:
           result = service_call_obj.result()
        except Exception as e:
           self.get_logger().error(f"{e}")
        self.executing = False
###
    def close_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 3.14    # swapped
        goal.command.max_effort = 0.0
        event = threading.Event()
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(lambda f: event.set())
        event.wait()

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected')
            return False

        event = threading.Event()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: event.set())
        event.wait()

        self.get_logger().info('Gripper closed')
        return True

    def open_gripper(self):
        goal = GripperCommand.Goal()
        goal.command.position = 1.5708   # swapped
        goal.command.max_effort = 0.0

        event = threading.Event()
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(lambda f: event.set())
        event.wait()

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        event = threading.Event()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: event.set())
        event.wait()

        self.get_logger().info('Gripper opened')
        return True

    def handle_pick_sequential(self, request,response):
        self.open_gripper()
        obj_x, obj_y, obj_z = request.x, request.y, request.z
        self.get_logger().info(f'Sequential pick: x={obj_x}, y={obj_y}, z={obj_z}')
        base_angle = math.atan2(obj_y, obj_x)

        dist = math.sqrt(obj_x*obj_x + obj_y*obj_y)
        #rotatitng base to the object
        safe_dist = 0.06  # 6cm from base, retracted
        safe_x = safe_dist * math.cos(base_angle)
        safe_y = safe_dist * math.sin(base_angle)
        safe_z = 0.05 # safe height above ground
        self.get_logger().info(f'Step 1: Rotating to face object (base={math.degrees(base_angle):.1f}°)')
        step1_req = MoveToXYZ.Request()
        step1_req.x = safe_x
        step1_req.y = safe_y
        step1_req.z = safe_z
        step1_resp = MoveToXYZ.Response()
        self.handle_move_request(step1_req, step1_resp)
        time.sleep(0.4)
        if not step1_resp.success:
            response.success = False
            response.message = ' Step 1 failed could not rotate'
            return response
        self.get_logger().info(f'Step 2: Moving to target ({obj_x}, {obj_y}, {obj_z})')
        step2_req = MoveToXYZ.Request()
        step2_req.x = obj_x
        step2_req.y = obj_y
        step2_req.z = obj_z
        step2_resp = MoveToXYZ.Response()

        self.handle_move_request(step2_req, step2_resp)
        self.close_gripper()
        time.sleep(1)
        if not step2_resp.success:
            response.success = False
            response.message = 'Step 2 failed: could not reach target'
            return response

        response.success = True

        #{x: -0.34, y: 0.02, z: 0.16}"
        self.get_logger().info(f'Step 3: Moving to trash drop')
        step3_req = MoveToXYZ.Request()
        step3_req.x = -0.34
        step3_req.y = 0.0
        step3_req.z = 0.16
        step3_resp = MoveToXYZ.Response()

        self.handle_move_request(step3_req, step3_resp)
        if not step3_resp.success:
            response.success = False
            response.message = ' Step 3 failed could not drop'
            return response
        self.open_gripper()
        time.sleep(0.5)
        self.get_logger().info(f'Step 4: Moving to init')
        step4_req = MoveToXYZ.Request()
        step4_req.x = 0.1
        step4_req.y = 0.0
        step4_req.z = 0.15
        step4_resp = MoveToXYZ.Response()

        self.handle_move_request(step4_req, step4_resp)
        if not step3_resp.success:
            response.success = False
            response.message = ' Step 4 failed could not got to init'
            return response
        self.close_gripper()

        return response


    def solve_ik(self, x, y, z):
        # Tried in order, first solution wins. Pitch is the angle between the  #change
        # gripper's approach axis and the horizontal: 90 = straight down.  #change
        # Top-down costs reach, so fall back to shallower angles when 90 fails.  #change
        pitch_values = [90, 85, 80, 70, 60, 50, 40, 30, 20, 18, 15, 13, 0, 120, 150, 180]  #change
        result = None  #change

        # Yaw that points the arm at the object. This goes into the requested  #change
        # orientation rather than being written over the base joint afterwards:  #change
        # KDL may return a solution whose base is offset by pi with the rest of  #change
        # the arm flipped to match, and overwriting the base in that case  #change
        # mirrors the tool to the opposite side of the robot.  #change
        yaw = math.atan2(y, x)  #change
        # removed: dist / link1_x / link1_y / link1_z -- the target used to be
        # collapsed onto the y=0 plane before being sent to IK  #change

        for pdeg in pitch_values:
            pitch = math.radians(pdeg)

            # Rz(yaw) * Ry(pitch)  #change
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)  #change
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)  #change
            qx, qy, qz, qw = -sy * sp, cy * sp, sy * cp, cy * cp  #change

            ik_request = GetPositionIK.Request()
            ik_request.ik_request.group_name = 'hand'
            ik_request.ik_request.pose_stamped = PoseStamped()
            ik_request.ik_request.pose_stamped.header.frame_id = 'world'
            ik_request.ik_request.pose_stamped.pose.position = Point(
                x=x, y=y, z=z  #change
            )
            ik_request.ik_request.pose_stamped.pose.orientation = Quaternion(
                x=qx, y=qy, z=qz, w=qw
            )
            ik_request.ik_request.avoid_collisions = True

            # Start the solver from where the arm actually is, so it returns  #change
            # the nearest solution instead of an arbitrary one.  #change
            if self.current_joint_state is not None:  #change
                ik_request.ik_request.robot_state.joint_state = self.current_joint_state  #change

            event = threading.Event()
            future = self.ik_client.call_async(ik_request)
            future.add_done_callback(lambda f: event.set())
            event.wait()

            result = future.result()
            if result is not None and result.error_code.val == 1:  #change
                names = list(result.solution.joint_state.name)
                positions = list(result.solution.joint_state.position)
                base = positions[names.index('base_link_to_link1')]  #change
                # removed: positions[base_idx] = base_angle -- overwriting the
                # base joint of a pi-flipped IK branch mirrored the tool to the
                # wrong side of the robot  #change

                # A sane solution points the base at the object. If it is off  #change
                # by more than a right angle the arm is reaching backwards and  #change
                # folding over itself to hit the same point -- correct pose,  #change
                # absurd path. Skip it and try the next pitch.  #change
                off = abs(math.atan2(math.sin(base - yaw), math.cos(base - yaw)))  #change
                if off > math.pi / 2:  #change
                    self.get_logger().warn(  #change
                        f'REJECTED: pitch={pdeg}° base={math.degrees(base):.1f}° '  #change
                        f'is {math.degrees(off):.0f}° from yaw={math.degrees(yaw):.1f}°'  #change
                    )  #change
                    continue  #change

                self.get_logger().info(  #change
                    f'IK SOLVEEEEED: pitch={pdeg}° yaw={math.degrees(yaw):.1f}° '  #change
                    f'base={math.degrees(base):.1f}°'  #change
                )  #change
                return result

            self.get_logger().warn(f'FAILED: pitch={pdeg}°')

        self.get_logger().error(f'IK failed at every pitch for ({x}, {y}, {z})')  #change
        return result
    
    def handle_move_request(self, request, response):
        self.get_logger().info(
            f'Received: x={request.x}, y={request.y}, z={request.z}'
        )

        ik_result = self.solve_ik(request.x, request.y, request.z)

        if ik_result is None:  #change
            response.success = False  #change
            response.message = 'IK service returned no result'  #change
            self.get_logger().error(response.message)  #change
            return response  #change

        if ik_result.error_code.val != 1:
            response.success = False
            response.message = f'IK failed with error code: {ik_result.error_code.val}'
            self.get_logger().error(response.message)
            return response

        joint_names = ik_result.solution.joint_state.name
        joint_positions = ik_result.solution.joint_state.position

        self.get_logger().info(f'IK solution: {list(zip(joint_names, joint_positions))}')

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = 'hand'
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 1.0
        goal.request.max_acceleration_scaling_factor = 1.0

        constraints = Constraints()
        for name, position in zip(joint_names, joint_positions):
            if name == 'link5_to_gripper_link':
                continue
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = position
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraints)

        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        event = threading.Event()
        future = self.move_action_client.send_goal_async(goal)
        future.add_done_callback(lambda f: event.set())
        event.wait()

        goal_handle = future.result()
        if not goal_handle.accepted:
            response.success = False
            response.message = 'Goal rejected by move_group'
            self.get_logger().error(response.message)
            return response

        self.get_logger().info('Goal accepted, executing...')

        event = threading.Event()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: event.set())
        event.wait()

        result = result_future.result().result
        if result.error_code.val == 1:
            response.success = True
            response.message = f'Moved to ({request.x}, {request.y}, {request.z})'
            self.get_logger().info(response.message)
        else:
            response.success = False
            response.message = f'Planning failed with error code: {result.error_code.val}'
            self.get_logger().error(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = MoveToXYZServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

