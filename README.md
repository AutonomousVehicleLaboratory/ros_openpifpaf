# OpenPifPaf ROS Wrapper (for Noetic)

OpenPifPaf[Github](https://github.com/openpifpaf/openpifpaf)[arXiv](https://arxiv.org/abs/2103.02440) is a popular pose estimation network. It works robustly even in extreme lighting conditions [Our project based on OpenPifPaf](https://github.com/AutonomousVehicleLaboratory/anonymization). We provide a ros wrapper and a rosbag processing script for it.

The rosbag processing script will generate a json file for each rosbag that has all the detection for each frame, example usage:
```
python3 src/ros_openpifpaf/src/rosbag_openpifpaf_human_detector.py /path/to/rosbag.bag --topic_list /camera/color/image_raw/compressed

python3 src/ros_openpifpaf/src/rosbag_openpifpaf_human_detector.py /path/to/rosbag_dir --topic_list /camera/color/image_raw/compressed /oak/rgb/image_raw/compressed
```

## Requirements

install openpifpaf python package.

## Input and output

Input:
1. Image or Compressed Image. It can handle multiple cameras.

Output:
1. /.../openpifpaf_pose, JointState message of pose keypoint of all the targets. distinguished by the name field.
2. /.../openpifpaf_detections. Detection2DArray message of 2d bounding boxes of all targets.
3. /.../openpifpaf_image, Image message of visualizing the bounding box and keypoints.

## Note:
We scale down the image by 0.5 to achieve real time performance. If we don't do this, current chosen model takes 300ms to process a full image (1920x1440). 