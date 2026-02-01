from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_ctrl_package',
            executable='control_node',
            name='control'
        )
    ])
