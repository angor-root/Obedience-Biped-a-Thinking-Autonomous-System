#!/bin/bash
# ============================================================
# OBEDIENCE Mission Control - Launch Script
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR"

echo "=============================================="
echo "  OBEDIENCE Mission Control - Launch System"
echo "=============================================="
echo ""
echo "  Options:"
echo "    --viewer    Enable MuJoCo 3D visualization"
echo "    --no-gui    Disable Mission Control GUI"
echo ""

# Parse arguments
USE_VIEWER=true  # Por defecto mostrar MuJoCo viewer
USE_GUI=true
for arg in "$@"; do
    case $arg in
        --viewer)
            USE_VIEWER=true
            ;;
        --no-viewer)
            USE_VIEWER=false
            ;;
        --no-gui)
            USE_GUI=false
            ;;
    esac
done

# Source ROS2 and workspace
cd "$WS_DIR"
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || colcon build --symlink-install && source install/setup.bash

# Kill any existing ROS2 processes from this workspace
echo "[1/5] Cleaning up existing processes..."
pkill -9 -f "obedience_robot" 2>/dev/null || true
sleep 1

# Launch robot node
echo "[2/5] Starting Robot Node..."
if [ "$USE_VIEWER" = true ]; then
    ros2 run obedience_robot robot --ros-args -p use_viewer:=true &
else
    ros2 run obedience_robot robot --ros-args -p use_viewer:=false &
fi
ROBOT_PID=$!
sleep 2

# Launch health node
echo "[3/5] Starting Health Node..."
ros2 run obedience_robot health_node &
HEALTH_PID=$!
sleep 1

# Launch thinking node
echo "[4/5] Starting Thinking Node..."
ros2 run obedience_robot thinking_node &
THINKING_PID=$!
sleep 1

# Launch mission control GUI
if [ "$USE_GUI" = true ]; then
    echo "[5/5] Starting Mission Control GUI..."
    ros2 run obedience_robot mission_control &
    GUI_PID=$!
else
    echo "[5/5] Skipping Mission Control GUI (--no-gui)"
    GUI_PID=""
fi

echo "=============================================="
echo "  All nodes started!"
echo "  Robot:    PID $ROBOT_PID (MuJoCo viewer: $USE_VIEWER)"
echo "  Health:   PID $HEALTH_PID"
echo "  Thinking: PID $THINKING_PID"
echo "  GUI:      PID ${GUI_PID:-'disabled'}"
echo "=============================================="
echo ""
echo "Press Ctrl+C to stop all nodes..."

# Wait for any process to exit
wait $ROBOT_PID $HEALTH_PID $THINKING_PID $GUI_PID

# Cleanup
echo "Shutting down..."
kill $ROBOT_PID $HEALTH_PID $THINKING_PID $GUI_PID 2>/dev/null || true
