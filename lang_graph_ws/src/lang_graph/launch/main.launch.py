from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

def generate_launch_description():

    ############################################################
    # Set the configuration file parse by argparser
    config_dir = PathJoinSubstitution([
        FindPackageShare("lang_graph"), 'config'
    ])
    config_filename = 'config.yaml'

    ############################################################
    # Set the image directory
    image_dir = PathJoinSubstitution([
        FindPackageShare("lang_graph"), 'data/img'
    ])

    ############################################################
    # Main node
    main_node = Node(
        package='lang_graph',
        executable='main.py',
        name='lang_graph',
        output='screen',
        parameters=[{
            'config_file': PathJoinSubstitution([config_dir, config_filename]),
            'image_folder': image_dir
        }]
    )

    return LaunchDescription([
        main_node
    ])