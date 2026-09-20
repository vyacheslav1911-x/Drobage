#!/usr/bin/env bash
# Builds only the packages needed for the RViz arm simulation.
# The perception/navigation packages (my_yolo_package, navigation_package,
# robot_ctrl_package) and the serial driver (roarm_driver) are skipped —
# they need real hardware (OAK camera, /dev/ttyUSB0) and CUDA.
set -eo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS"

# ROS setup scripts reference unbound variables, so 'set -u' must stay off here.
source /opt/ros/jazzy/setup.bash

echo "==> Resolving dependencies"
# Only scan the simulation packages. Pointing rosdep at all of src/ drags in the
# hardware packages' unresolvable keys (depthai, requests.exceptions, ...).
rosdep install --ignore-src -r -y \
  --from-paths src/roarm_msgs src/roarm_description src/roarm_moveit_cmd src/roarm_moveit_config \
  || echo "(rosdep reported issues; continuing — check the output above)"

echo "==> Building"
colcon build --symlink-install \
  --packages-select roarm_msgs roarm_description roarm_moveit_cmd roarm_moveit_config

echo
echo "Done. Next:"
echo "  source $WS/install/setup.bash"
echo "  ros2 launch roarm_moveit_config sim.launch.py"
