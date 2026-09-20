# RViz Simulation — RoArm M3

Run and test the arm's XYZ motion in RViz with no hardware attached.
`mock_components/GenericSystem` stands in for the serial-driven servos, so
nothing needs `/dev/ttyUSB0`, the OAK camera or CUDA.

## Which node does the motion

`src/roarm_moveit_cmd/roarm_moveit_cmd/xyz_control.py` — node name `xyz_control`.

| Method | Role |
| --- | --- |
| `solve_ik(x, y, z)` | Sweeps end-effector pitch `10°…180°` calling MoveIt's `/compute_ik` until one succeeds, then overrides the base joint with `atan2(y, x)` so the arm faces the object. |
| `handle_move_request()` | Turns the IK solution into joint constraints and sends a `MoveGroup` goal. Exposed as `/move_to_xyz`. |
| `handle_pick_sequential()` | Chains four moves: rotate-to-face → grasp → drop at `(-0.34, 0, 0.16)` → home at `(0.1, 0, 0.15)`. Exposed as `/pick_sequential`. |
| `control_loop()` | Autonomous path — fires `/pick_sequential` once `/stop` has been true for >3 ticks and `/detection` is true, using coordinates from `/object_coords`. Idle in simulation unless you publish those topics. |

The orientation logic lives in `solve_ik`: the pose is solved in the arm's own
vertical plane (`x = sqrt(x²+y²)`, `y = 0`) with a pitch-only quaternion, and the
yaw is then applied directly to `base_link_to_link1`.

## One-time setup

ROS 2 Jazzy is already installed at `/opt/ros/jazzy`. If you ever need to
reinstall it on a fresh machine (Ubuntu 24.04 → Jazzy):

```bash
cd ~/Desktop/Drobage/Drobage
./setup_sim_env.sh     # needs sudo, ~2 GB
```

**Every new terminal needs ROS on its path** — this is what `ros2: command not
found` means:

```bash
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Drobage/Drobage/install/setup.bash
```

## Build

```bash
./build_sim.sh
```

Builds only `roarm_msgs`, `roarm_description`, `roarm_moveit_cmd` and
`roarm_moveit_config`. `my_yolo_package`, `navigation_package`,
`robot_ctrl_package` and `roarm_driver` are skipped — they need hardware.

## Run

```bash
source /opt/ros/jazzy/setup.bash
source ~/Desktop/Drobage/Drobage/install/setup.bash
ros2 launch roarm_moveit_config sim.launch.py
```

`[ERROR] ... No 3D sensor plugin(s) defined for octomap updates` on startup is
expected and harmless — there is no depth camera in simulation.

This starts, in order: `robot_state_publisher` → `ros2_control_node` (mock
hardware) → `joint_state_broadcaster` → `hand_controller` →
`gripper_controller` + `move_group` + RViz → `xyz_control`.

Wait for `Service /move_to_xyz ready` in the log before commanding.

Launch arguments: `use_rviz:=false`, `use_xyz_control:=false`.

## Command an XYZ target

In a second terminal (source both setup files first):

```bash
cd ~/Desktop/Drobage/Drobage

./move_xyz.sh 0.10 0.00 0.15          # home pose — try this first
./move_xyz.sh 0.20 0.00 0.15
./move_xyz.sh 0.18 0.12 0.10          # off-axis: base rotates to face it
./move_xyz.sh 0.20 0.00 0.12 pick     # full pick → drop → home sequence
```

Or directly:

```bash
ros2 service call /move_to_xyz roarm_msgs/srv/MoveToXYZ "{x: 0.2, y: 0.0, z: 0.15}"
ros2 service call /pick_sequential roarm_msgs/srv/MoveToXYZ "{x: 0.2, y: 0.0, z: 0.12}"
```

The response is `success: true` plus a message; the arm animates in RViz.

Coordinates are metres in the **`world`** frame. `base_link` sits at
`(-0.11, 0, 0.1)` in `world` (see `world_to_base_link` in
`roarm_description/urdf/roarm_m3/roarm_m3.xacro`), and reach is roughly
0.1–0.4 m in x. Outside that, every pitch fails and you get
`IK failed with error code: -31`.

## Driving the autonomous path in simulation

To exercise `control_loop()` without the camera, publish what the perception
nodes would:

```bash
ros2 topic pub -r 2 /object_coords geometry_msgs/msg/Point "{x: 0.2, y: 0.0, z: 0.12}"
ros2 topic pub -r 2 /detection std_msgs/msg/Bool "{data: true}"
ros2 topic pub -r 2 /stop std_msgs/msg/Bool "{data: true}"      # needs >3 ticks (~2 s)
```

Note `object_coords_callback` negates y (`self.y = -msg.y`).

## Checks

```bash
ros2 control list_controllers          # all three should be 'active'
ros2 service list | grep -E 'move_to_xyz|pick_sequential|compute_ik'
ros2 topic echo /joint_states --once
```

If the arm does not move but `success: true` comes back, confirm RViz's Fixed
Frame is `world` and that the MotionPlanning display's "Planned Path" is
showing the trajectory.

## Verified

Run end-to-end on ROS 2 Jazzy with `use_rviz:=false`:

| Test | Result |
| --- | --- |
| `ros2 control list_controllers` | `joint_state_broadcaster`, `hand_controller`, `gripper_controller` all `active` |
| `/move_to_xyz {0.2, 0.0, 0.15}` | `success: true`; joints `[0.0, 0.0, 2.618, -1.047, 0.0]` → `[0.002, 0.538, 2.609, -1.413, -0.008]` |
| `/move_to_xyz {0.18, 0.12, 0.10}` | `success: true`; `base_link_to_link1` = **0.5884 rad** vs `atan2(0.12, 0.18)` = **0.5880 rad** — orientation logic confirmed |
| `/move_to_xyz {0.9, 0.0, 0.5}` | `success: false`, `IK failed with error code: -31` — out of reach, fails gracefully |
| `/pick_sequential {0.2, 0.0, 0.12}` | `success: true`; all four steps ran (rotate → grasp → drop at base=180° → home) with the gripper opening and closing |

## Notes on what was fixed to get here

- `config/ompl_planning.yaml` used Humble-era adapter names
  (`default_planner_request_adapters/Fix*`) as space-separated strings. Jazzy
  (MoveIt 2.12) renamed them to `default_planning_request_adapters/Check*` /
  `Validate*` and takes them as **string arrays**. This crashed `move_group` at
  startup with `rclcpp::ParameterTypeException: expected [string_array] got
  [string]`, which in turn hung `xyz_control` on `wait_for_server()`.
- `xyz_control.py` exposed only `/pick_sequential`; `handle_move_request` was
  unreachable from outside the process. Now also bound to `/move_to_xyz`.
- `solve_ik` returned an unbound `result` when every pitch failed
  (`UnboundLocalError`); it now returns `None` and the caller reports the failure.
- `roarm_moveit_cmd/package.xml` declared `<depend>time</depend>`, which is not a
  ROS package and broke `rosdep`.
- `demo.launch.py` spawned all three controllers immediately, racing the
  controller_manager. `sim.launch.py` sequences them with event handlers.
