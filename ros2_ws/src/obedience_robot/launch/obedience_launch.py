"""
Launch file for Obedience bipedal robot system.

Launches:
    - Integrated Robot Node: MuJoCo simulation + Walking controller + ROS2 interface

Usage:
    ros2 launch obedience_robot obedience_launch.py
    ros2 launch obedience_robot obedience_launch.py scene_xml:=/path/to/scene.xml use_viewer:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    scene_xml_arg = DeclareLaunchArgument(
        'scene_xml',
        default_value='hospital_scene.xml',
        description='MuJoCo XML scene file path'
    )
    
    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer',
        default_value='true',
        description='Whether to show MuJoCo viewer'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='50.0',
        description='Telemetry publishing rate in Hz'
    )
    
    # Integrated Robot Node (MuJoCo + Walking Controller + ROS2)
    robot_node = Node(
        package='obedience_robot',
        executable='robot',
        name='obedience_robot',
        output='screen',
        parameters=[{
            'scene_xml': LaunchConfiguration('scene_xml'),
            'use_viewer': LaunchConfiguration('use_viewer'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )
    
    return LaunchDescription([
        scene_xml_arg,
        use_viewer_arg,
        publish_rate_arg,
        robot_node,
    ])
