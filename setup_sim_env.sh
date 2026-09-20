#!/usr/bin/env bash
# Installs everything needed to run the RoArm M3 MoveIt simulation in RViz
# on Ubuntu 24.04 (Noble) -> ROS 2 Jazzy.
#
# Run once:  ./setup_sim_env.sh
# Requires sudo. Safe to re-run.
set -euo pipefail

if [[ "$(. /etc/os-release && echo "$VERSION_CODENAME")" != "noble" ]]; then
  echo "This script targets Ubuntu 24.04 (noble). Detected: $(. /etc/os-release && echo "$VERSION_CODENAME")" >&2
  echo "Use the matching ROS 2 distro for your release instead of jazzy." >&2
  exit 1
fi

echo "==> Enabling the universe repository"
sudo apt-get update
sudo apt-get install -y software-properties-common curl
sudo add-apt-repository -y universe

echo "==> Adding the ROS 2 apt repository"
sudo curl -fsSL -o /usr/share/keyrings/ros-archive-keyring.gpg \
  https://raw.githubusercontent.com/ros/rosdistro/master/ros.key
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu noble main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt-get update

echo "==> Installing ROS 2 Jazzy (desktop: includes RViz2)"
sudo apt-get install -y ros-jazzy-desktop

echo "==> Installing MoveIt2, ros2_control and build tooling"
sudo apt-get install -y \
  ros-jazzy-moveit \
  ros-jazzy-moveit-ros-move-group \
  ros-jazzy-moveit-ros-visualization \
  ros-jazzy-moveit-planners-ompl \
  ros-jazzy-moveit-simple-controller-manager \
  ros-jazzy-moveit-kinematics \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-position-controllers \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-control-msgs \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep

echo "==> Initialising rosdep"
sudo rosdep init 2>/dev/null || true
rosdep update

echo
echo "Done. Next:"
echo "  ./build_sim.sh"
