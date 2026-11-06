# OPL Human Vision

## Build

```bash
source /opt/ros/jazzy/setup.bash
source ~/orbbec_ws/install/setup.bash  # Gemini 335 driver overlay
cd ~/opl_ros
colcon build --symlink-install --packages-up-to opl_human_vision
source install/setup.bash
```

If using the Jazzy apt-installed Orbbec driver, omit the `orbbec_ws` source line.
In each new terminal, source ROS, the camera-driver overlay, and this workspace
before launching.

## Launch

Gemini 335 on the Pi, with the visualizer running on the Jetson:

```bash
ros2 launch opl_human_vision test_detector.launch.py camera_type:=gemini335 launch_camera:=false enable_visualizer:=true
```

Use ROS domain 122 on both machines and source the Jetson's LAN DDS settings.
Remote Gemini mode is the default. It receives `/head_camera/color/image_raw/compressed`
and `/head_camera/depth/image_raw/compressedDepth`, decodes each once, and supplies
`/vision/color/image_raw` and `/vision/depth/image_raw` to the pipeline. Calibration
comes from `/head_camera/color/camera_info`. No second camera driver starts.

On the Pi, run the camera-only launch below with `camera_name:=head_camera`.
It uses 640x480 color and registered depth at 15 FPS. Stop any existing camera
driver before starting it; Jetson launch arguments do not reconfigure the Pi's
running driver. Matching color/depth dimensions and optical frame IDs are required
for valid 3D positions. For a Gemini connected directly to the Jetson, use
`launch_camera:=true` instead.

This test launch accepts following positions up to 0.30 seconds old to accommodate
the measured synchronized USB 2 camera latency. It still invalidates following
after 0.15 seconds without a new tracking message, preserves capture timestamps,
and rejects invalid depth or uncertain identities. This setting accepts older
positions; it does not reduce capture latency. Use
`following_max_observation_age_sec:=0.15` to retain the stricter capture-age limit.

RealSense with the visualizer:

```bash
ros2 launch opl_human_vision test_detector.launch.py camera_type:=realsense enable_visualizer:=true
```

Launch only the Gemini camera driver:

```bash
ros2 launch opl_gemini335 gemini335.launch.py
```

For the Pi's remote camera namespace:

```bash
ros2 launch opl_gemini335 gemini335.launch.py camera_name:=head_camera
```

## Export TensorRT engines on the target device

Build the engines on the Jetson or laptop that will run them; TensorRT engine
files are tied to the target GPU and TensorRT environment. From the workspace
root, run:

```bash
cd src/opl_human_vision/laptop_models
for model in yolov8n-pose scrfd_2.5g adaface_ir50 osnet_x1_0; do
  trtexec --onnx="${model}.onnx" --saveEngine="${model}.engine"
done
```

This writes the four engine files used by `test_detector.launch.py` beside
their ONNX files. It replaces any existing engine files with the same names.
