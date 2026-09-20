#!/usr/bin/env bash
# Command the arm to an XYZ target (metres, world frame).
#
#   ./move_xyz.sh 0.2 0.0 0.15          # move only
#   ./move_xyz.sh 0.2 0.0 0.15 pick     # full pick -> drop -> home sequence
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <x> <y> <z> [pick]" >&2
  exit 1
fi

SRV="/move_to_xyz"
[[ "${4:-}" == "pick" ]] && SRV="/pick_sequential"

exec ros2 service call "$SRV" roarm_msgs/srv/MoveToXYZ "{x: $1, y: $2, z: $3}"
