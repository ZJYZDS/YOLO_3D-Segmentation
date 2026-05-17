# KFS_grab

Real-time multi-face KFS detection and 3D localization pipeline for RoboCon 2025. YOLO segmentation + depth-guided projection + RANSAC plane refinement + adaptive TF transform.

## Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   RealSense      │    │  YOLOv8-seg     │    │  Mask → 3D       │
│  RGB-D Capture   │───▶│  Inference      │───▶│  Projection      │
└─────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                        │
                    ┌──────────────────┐                │
                    │  Serial Output   │◀───────────────┤
                    │  (to MCU)        │                │
                    └──────────────────┘                │
                                      ┌─────────────────▼────────┐
                                      │  Radius Search +        │
                                      │  RANSAC Plane Fitting   │
                                      └─────────────────┬────────┘
                                                        │
                                      ┌─────────────────▼────────┐
                                      │  Face Classification    │
                                      │  (normal vector →       │
                                      │   adaptive TF)          │
                                      └─────────────────┬────────┘
                                                        │
                                      ┌─────────────────▼────────┐
                                      │  Camera Frame →          │
                                      │  Base Link Transform     │
                                      │  + Serial Send           │
                                      └──────────────────────────┘
```

## Features

- **Multi-face recognition** — Classifies which face of the KFS is visible by analyzing plane normal vectors, applies per-face TF transforms automatically
- **YOLO-guided 3D projection** — Runs YOLOv8 segmentation on RGB, projects mask onto depth-aligned point cloud, filters by mask region
- **RANSAC plane refinement** — Uses the YOLO center as a seed, performs radius search and RANSAC plane fitting for sub-centimeter accuracy
- **Serial communication** — Sends computed coordinates to lower-level MCU via UART with custom protocol
- **Hybrid C++/Python** — Python node for RealSense + YOLO inference, C++ node for PCL processing and serial output

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | ROS Noetic |
| 2D Detection | YOLOv8-seg (Ultralytics) |
| 3D Processing | PCL 1.10, librealsense2 |
| Camera | Intel RealSense D455i |
| Serial | Boost.Asio `serial` library |
| Math | Eigen3 |

## ROS Nodes

### Python: `scripts/kfs_grab.py`
RealSense capture → YOLO inference → mask projection → center calculation → serial send + ROS publish.

**Published topics:**
- `/kfs_pointcloud` (`sensor_msgs/PointCloud2`) — Mask-filtered point cloud
- `/center_coord` (`std_msgs/Float32MultiArray`) — Raw center coordinate [x, y, z]

### C++: `src/kfs_grab.cpp`
Subscribes to point cloud + seed center, performs RANSAC plane fitting, classifies KFS face, applies coordinate transform, sends result via serial.

**Published topics:**
- `/plane_points` (`sensor_msgs/PointCloud2`) — RANSAC-filtered plane point cloud
- `/plane_center` (`geometry_msgs/PointStamped`) — Refined center in base link frame

### C++: `src/temp_pub.cpp`
Bridges the Python center output into a `PointStamped` with proper timestamp for synchronized subscription.

**Published topics:**
- `/center_point_with_header` (`geometry_msgs/PointStamped`)

### Utility: `scripts/utils/kfs_pointcloud.py`
Point cloud generation from RealSense depth frame with YOLO mask filtering.

## Dependencies

- ROS Noetic
- OpenCV
- PCL ≥ 1.10
- librealsense2
- Ultralytics YOLO
- Boost.Serial
- Eigen3

## Build

```bash
cd ~/catkin_ws/src
git clone https://github.com/ZJYZDS/KFS_grab.git
cd ..
catkin_make
```

## Usage

Launch RealSense camera, then:

```bash
# Terminal 1: Python perception node
rosrun pcd_view_pkg kfs_grab.py

# Terminal 2: C++ plane fitting node
rosrun pcd_view_pkg kfs_grab
```

Or with your existing robot bringup:

```bash
roslaunch your_robot_bringup kfs_perception.launch
```


MIT
