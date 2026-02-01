from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_yolo_package',
            executable='oak_camera_node',
            name='camera'
        ),
        Node(
            package='my_yolo_package',
            executable='inference_node',
            name='inference'
        ),
        Node(
            package='my_yolo_package',
            executable='visualizer_node',
            name='visualizer'
        )
    ])
