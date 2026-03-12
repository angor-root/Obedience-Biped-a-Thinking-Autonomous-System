#!/bin/bash
# =============================================================================
# OBEDIENCE Mission Control Launch Script
# 
# Launches the complete autonomous bipedal robot system:
# - Robot simulation (MuJoCo + Walking Controller)
# - Health monitoring (FMEA-based)
# - Thinking system (Autonomy)
# - Mission Control GUI (NASA-style interface)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/ros2_ws"
SCENE_XML="${SCRIPT_DIR}/models/xml/hospital_scene.xml"

echo -e "${BLUE}"
echo "============================================================"
echo "    ◆ OBEDIENCE MISSION CONTROL SYSTEM ◆"
echo "    Autonomous Bipedal Robot for Hospital Medicine Delivery"
echo "============================================================"
echo -e "${NC}"

# Check if ROS2 is sourced
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}Sourcing ROS2 Humble...${NC}"
    source /opt/ros/humble/setup.bash
fi

# Source workspace
if [ -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    source "${WORKSPACE_DIR}/install/setup.bash"
else
    echo -e "${RED}ERROR: Workspace not built. Run: cd ros2_ws && colcon build${NC}"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down all nodes...${NC}"
    pkill -f "ros2 run obedience_robot" 2>/dev/null || true
    pkill -f "obedience_robot" 2>/dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
}
trap cleanup EXIT

# Parse arguments
USE_GUI=true
USE_VIEWER=true
HEADLESS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-gui)
            USE_GUI=false
            shift
            ;;
        --no-viewer)
            USE_VIEWER=false
            shift
            ;;
        --headless)
            USE_VIEWER=false
            USE_GUI=false
            HEADLESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-gui] [--no-viewer] [--headless]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting OBEDIENCE System...${NC}"
echo ""

# Start Robot Node (background)
echo -e "${BLUE}[1/4] Starting Robot Simulation...${NC}"
ros2 run obedience_robot robot \
    --ros-args \
    -p scene_xml:="${SCENE_XML}" \
    -p use_viewer:=${USE_VIEWER} \
    -p publish_rate:=50.0 &
ROBOT_PID=$!
echo -e "  ${GREEN}✓ Robot PID: ${ROBOT_PID}${NC}"
sleep 3

# Start Health Node (background)
echo -e "${BLUE}[2/4] Starting Health Monitoring...${NC}"
ros2 run obedience_robot health_node &
HEALTH_PID=$!
echo -e "  ${GREEN}✓ Health Node PID: ${HEALTH_PID}${NC}"
sleep 1

# Start Thinking Node (background)
echo -e "${BLUE}[3/4] Starting Thinking System...${NC}"
ros2 run obedience_robot thinking_node &
THINKING_PID=$!
echo -e "  ${GREEN}✓ Thinking Node PID: ${THINKING_PID}${NC}"
sleep 1

# Start Mission Control GUI (foreground)
if [ "$USE_GUI" = true ]; then
    echo -e "${BLUE}[4/4] Starting Mission Control GUI...${NC}"
    echo ""
    echo -e "${GREEN}============================================================"
    echo -e "  System is RUNNING"
    echo -e "  Close the Mission Control GUI to stop all nodes"
    echo -e "============================================================${NC}"
    echo ""
    ros2 run obedience_robot mission_control
else
    echo -e "${YELLOW}[4/4] GUI disabled. System running in background.${NC}"
    echo ""
    echo -e "${GREEN}============================================================"
    echo -e "  System is RUNNING (headless mode)"
    echo -e "  Press Ctrl+C to stop all nodes"
    echo -e "============================================================${NC}"
    echo ""
    
    # Wait for any key or signal
    wait
fi
