from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource 
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription, TimerAction

def generate_launch_description():

    planner_node = Node(
       package='nav2_planner',
       executable='planner_server',
       arguments=['--ros-args', '--params-file', '/home/wb/Desktop/Drobage/src/navigation_package/share/config/nav2_demo_params.yaml'])

    controller_node = Node(
       package='nav2_controller',
       executable='controller_server',
       arguments=['--ros-args', '--params-file', '/home/wb/Desktop/Drobage/src/navigation_package/share/config/nav2_demo_params.yaml'])

    navigator_node = Node(
       package='nav2_bt_navigator',
       executable='bt_navigator',
       arguments=['--ros-args', '--params-file', '/home/wb/Desktop/Drobage/src/navigation_package/share/config/nav2_demo_params.yaml'])

    behaviors_node = Node(
       package='nav2_behaviors',
       executable='behavior_server',
       arguments=['--ros-args', '--params-file', '/home/wb/Desktop/Drobage/src/navigation_package/share/config/nav2_demo_params.yaml'])

#    lifecycle_node = Node(
#       package='nav2_lifecycle_manager',
#       executable='lifecycle_manager',
#       arguments=[ {'use_sim_time':False}, {'autostart':True}, {'node_names':["controller_server", "planner_server","bt_navigator","behavior_server"]}])

#    delayed_lifecycle = TimerAction(period=3.0, actions=[lifecycle_node])
    return LaunchDescription([
        planner_node,
        controller_node,
        navigator_node,
        behaviors_node,
#        delayed_lifecycle
    ])

