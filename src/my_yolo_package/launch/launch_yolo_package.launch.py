from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource 
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription, TimerAction

def generate_launch_description():
    camera_node = Node(
        package='my_yolo_package',
        executable='oak_camera_node',
        name='camera'
    )
    inference = Node(
        package='my_yolo_package',
        executable='inference_node',
        name='inference'
    )
    visualizer = Node(
        package='my_yolo_package',
        executable='visualizer_node',
        name='visualizer'
    )
    control = Node(
        package='robot_ctrl_package',
        executable='control_node',
        name='control'
    )
   


    return LaunchDescription([
        camera_node,
        inference,
        visualizer,
        control
    ])
