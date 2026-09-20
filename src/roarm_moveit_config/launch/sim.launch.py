"""Full RViz simulation bring-up for the RoArm M3.

Starts the mock-hardware ros2_control stack, MoveIt2, RViz and the
xyz_control node, so an XYZ target can be commanded with no real arm
attached:

    ros2 launch roarm_moveit_config sim.launch.py
    ros2 service call /move_to_xyz roarm_msgs/srv/MoveToXYZ "{x: 0.2, y: 0.0, z: 0.15}"
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    full_path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    moveit_config_dir = get_package_share_directory('roarm_moveit_config')

    use_rviz = LaunchConfiguration('use_rviz')
    use_xyz_control = LaunchConfiguration('use_xyz_control')

    xacro_file = os.path.join(moveit_config_dir, 'config', 'roarm_m3.urdf.xacro')
    robot_description = {'robot_description': ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str
    )}

    srdf_file = os.path.join(moveit_config_dir, 'config', 'roarm_m3.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    kinematics_yaml = load_yaml('roarm_moveit_config', 'config/kinematics.yaml')
    robot_description_kinematics = {'robot_description_kinematics': kinematics_yaml}

    joint_limits_yaml = load_yaml('roarm_moveit_config', 'config/joint_limits.yaml')
    sensors_3d_yaml = load_yaml('roarm_moveit_config', 'config/sensors_3d.yaml')
    ompl_planning_yaml = load_yaml('roarm_moveit_config', 'config/ompl_planning.yaml')
    planning_pipelines = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': ompl_planning_yaml,
    }
    moveit_controllers_yaml = load_yaml('roarm_moveit_config', 'config/moveit_controllers.yaml')
    ros2_controllers_file = os.path.join(moveit_config_dir, 'config', 'ros2_controllers.yaml')

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # mock_components/GenericSystem stands in for the serial-driven arm.
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_file],
        remappings=[('/controller_manager/robot_description', '/robot_description')],
        output='screen',
    )

    def spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='screen',
        )

    joint_state_broadcaster_spawner = spawner('joint_state_broadcaster')
    hand_controller_spawner = spawner('hand_controller')
    gripper_controller_spawner = spawner('gripper_controller')

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {'robot_description_planning': joint_limits_yaml},
            moveit_controllers_yaml,
            sensors_3d_yaml,
            planning_pipelines,
            {'use_sim_time': False},
            {'moveit_manage_controllers': True},
            {'trajectory_execution.controller_manager_name': '/controller_manager'},
            {'trajectory_execution.allowed_execution_duration_scaling': 1.2},
            {'trajectory_execution.allowed_goal_duration_margin': 0.5},
            {'trajectory_execution.allowed_start_tolerance': 0.01},
            {'trajectory_execution.execution_duration_monitoring': False},
            {'publish_robot_description_semantic': True},
        ],
    )

    rviz_config = os.path.join(moveit_config_dir, 'config', 'moveit.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
        arguments=['-d', rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            planning_pipelines,
        ],
    )

    # Blocks in its constructor on move_action / gripper_cmd / compute_ik,
    # so it is only started once move_group is up.
    xyz_control_node = Node(
        package='roarm_moveit_cmd',
        executable='xyz_control',
        output='screen',
        condition=IfCondition(use_xyz_control),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('use_xyz_control', default_value='true'),

        robot_state_publisher,
        ros2_control_node,

        # Controllers must not be spawned before the controller_manager exists,
        # and the trajectory controllers need joint_state_broadcaster first.
        RegisterEventHandler(OnProcessStart(
            target_action=ros2_control_node,
            on_start=[joint_state_broadcaster_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[hand_controller_spawner],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=hand_controller_spawner,
            on_exit=[gripper_controller_spawner, move_group_node, rviz_node],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[xyz_control_node],
        )),
    ])
