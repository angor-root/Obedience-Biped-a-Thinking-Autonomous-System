"""
Full System Launch File for Obedience Bipedal Robot.

Launches the complete autonomous system:
    1. Robot Node: MuJoCo simulation + Walking controller + Sensor publishers
    2. Thinking Node: High-level autonomy (mission planning, decision making)
    3. Health Node: ISHM fault detection and monitoring

Usage:
    # Full system:
    ros2 launch obedience_robot full_system_launch.py

    # With fault injection GUI:
    ros2 launch obedience_robot full_system_launch.py enable_gui:=true
    
    # Custom scene:
    ros2 launch obedience_robot full_system_launch.py scene_xml:=/path/to/scene.xml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    # ========== LAUNCH ARGUMENTS ==========
    
    scene_xml_arg = DeclareLaunchArgument(
        'scene_xml',
        default_value='/root/Obedience-Biped-a-Thinking-Autonomous-System/models/xml/hospital_scene.xml',
        description='MuJoCo XML scene file path'
    )
    
    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer',
        default_value='true',
        description='Whether to show MuJoCo viewer'
    )
    
    enable_gui_arg = DeclareLaunchArgument(
        'enable_gui',
        default_value='false',
        description='Whether to launch fault injection GUI'
    )
    
    auto_start_arg = DeclareLaunchArgument(
        'auto_start_mission',
        default_value='true',
        description='Auto-start mission after initialization'
    )
    
    # ========== NODE DEFINITIONS ==========
    
    # 1. Robot Node - MuJoCo Simulation + Walking Controller
    # This is the core - runs physics and low-level control
    robot_node = Node(
        package='obedience_robot',
        executable='robot',
        name='obedience_robot',
        output='screen',
        parameters=[{
            'scene_xml': LaunchConfiguration('scene_xml'),
            'use_viewer': LaunchConfiguration('use_viewer'),
            'publish_rate': 50.0,  # Hz - sensor data rate
        }],
        # Topics provided:
        # Publishers: /imu, /joint_states, /robot_position, /battery_level, /contacts
        # Subscribers: /nav_target, /cmd_vel
    )
    
    # 2. Health Node - Fault Detection (ISHM)
    # Starts 2 seconds after robot to ensure sensors are ready
    health_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='obedience_robot',
                executable='health_node',
                name='health_node',
                output='screen',
                parameters=[{
                    'update_rate': 20.0,  # Hz - fast monitoring
                    'enable_fault_injection': True,
                }],
                # Topics provided:
                # Publishers: /health_status, /active_faults, /recovery_action, /health_alert
                # Subscribers: /imu, /joint_states, /contacts, /battery_level, /inject_fault, /clear_fault
            )
        ]
    )
    
    # 3. Thinking Node - High-Level Autonomy
    # Starts 3 seconds after robot to ensure health monitoring is ready
    thinking_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='obedience_robot',
                executable='thinking_node',
                name='thinking_node',
                output='screen',
                parameters=[{
                    'update_rate': 10.0,  # Hz - deliberation rate
                    'auto_start_mission': LaunchConfiguration('auto_start_mission'),
                }],
                # Topics provided:
                # Publishers: /nav_target, /mission_status, /system_status, /current_action
                # Subscribers: /imu, /joint_states, /health_status, /robot_position, /mission_command
            )
        ]
    )
    
    # 4. Fault Injection GUI (optional)
    # Only launches if enable_gui:=true
    fault_gui_node = Node(
        package='obedience_robot',
        executable='fault_injection_gui',
        name='fault_injection_gui',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_gui')),
        # Topics provided:
        # Publishers: /inject_fault, /clear_fault
    )
    
    # ========== LAUNCH DESCRIPTION ==========
    
    return LaunchDescription([
        # Arguments
        scene_xml_arg,
        use_viewer_arg,
        enable_gui_arg,
        auto_start_arg,
        
        # Nodes (in order)
        robot_node,       # t=0s: Start simulation
        health_node,      # t=2s: Start health monitoring
        thinking_node,    # t=3s: Start autonomy
        fault_gui_node,   # t=0s: GUI (if enabled)
    ])
