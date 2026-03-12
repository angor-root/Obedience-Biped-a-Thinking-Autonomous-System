#!/bin/bash
# =============================================================================
# OBEDIENCE - ROS2 System Launch Script
# =============================================================================
# This script sets up the environment and launches the bipedal robot system.
#
# Usage:
#   ./launch_ros2.sh                    # Default launch with viewer
#   ./launch_ros2.sh --no-viewer        # Launch without MuJoCo viewer
#   ./launch_ros2.sh --fast             # Run simulation as fast as possible
#
# After launch, in another terminal:
#   ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.3}, angular: {z: 0.0}}"
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR/ros2_ws"

# Parse arguments
USE_VIEWER="true"
REALTIME="1.0"

for arg in "$@"; do
    case $arg in
        --no-viewer)
            USE_VIEWER="false"
            ;;
        --fast)
            REALTIME="0.0"
            ;;
        --help|-h)
            echo "Usage: $0 [--no-viewer] [--fast]"
            echo ""
            echo "Options:"
            echo "  --no-viewer    Run without MuJoCo visualization"
            echo "  --fast         Run simulation as fast as possible"
            exit 0
            ;;
    esac
done

echo "============================================================"
echo "      OBEDIENCE - Autonomous Bipedal Robot System"
echo "============================================================"
echo ""

# Source ROS2
echo "[1/3] Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# Source workspace
echo "[2/3] Sourcing workspace..."
source "$WS_DIR/install/setup.bash"

# Check MuJoCo Python
echo "[3/3] Checking MuJoCo..."
PYTHON_PATH="/root/Bipedal_walking_capture_point/bipedal_capture_point/bin/python"
if [ -f "$PYTHON_PATH" ]; then
    export PYTHONPATH="$($PYTHON_PATH -c 'import sys; print(":".join(sys.path))')":$PYTHONPATH
fi

echo ""
echo "Configuration:"
echo "  - Viewer: $USE_VIEWER"
echo "  - Realtime: $REALTIME"
echo ""
echo "Launching... (Ctrl+C to stop)"
echo ""

# Launch
ros2 launch obedience_robot obedience_launch.py \
    scene_xml:=hospital_scene.xml \
    use_viewer:=$USE_VIEWER \
    realtime_factor:=$REALTIME
