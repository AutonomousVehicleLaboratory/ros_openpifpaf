# OpenPifPaf ROS Wrapper (for Noetic)

OpenPifPaf[Github](https://github.com/openpifpaf/openpifpaf)[arXiv](https://arxiv.org/abs/2103.02440) is a popular pose estimation network. It works robustly even in extreme lighting conditions [Our project based on OpenPifPaf](https://github.com/AutonomousVehicleLaboratory/anonymization).

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
We scale down the image by 0.5 to achieve real time performance. If we don't do this, current chosen model takes 300ms to process a full image. 