from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch Gazebo
    launch_gazebo = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("kobuki_gazebo"), "launch", "aws_small_house.launch.py"],
        ),
        launch_arguments={
            "spawn_kobuki": "False",
        }.items(),
    )

    # Launch Gazebo RL env controller
    launch_test_map = Node(
        package="gazebo_rl_env",
        executable="test_map.py",
        name="test_map",
        output="screen",
    )

    ld = LaunchDescription()
    # ld.add_action(launch_gazebo)
    ld.add_action(launch_test_map)

    return ld
