#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import openpifpaf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, JointState
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class CameraOpenPifPafNode():
    def __init__(self):
        """ Models:
        'mobilenetv2':
        'mobilenetv3large': 58.4    34ms    15.0 MB  # This model gives error
        'shufflenetv2k16':  68.1    40ms    38.9 MB
        'shufflenetv2k30':  71.8    81ms    115.0 MB # 
        """
        self.model_name = 'shufflenetv2k30' # model we use for anonymization 
        
        self.model = openpifpaf.Predictor(checkpoint=self.model_name)
        
        self.image_scale = 0.5      # resize to achieve real time performance
        # self.image_scale = 1 / 3.0      # further speed up

        self.output_image = True    # optional output image
        self.color = (0, 255, 0)    # label text color
        self.thickness = 2          # bounding box thickness
        self.show_label = True
        self.show_bbox = True
        self.show_skeleton = True

        # we have separate publisher and subscriber for each camera
        # that reuse the same callback funciton.
        self.camera_list = [
            "camera1", "camera2", "camera3", "camera4", "camera5", "camera6"
        ]

        self.sub_image, self.pub_detection, self.pub_pose = {}, {}, {}
        if self.output_image:
            self.pub_image = {}
        for cam in self.camera_list:
            self.sub_image[cam] = rospy.Subscriber("/avt_cameras/{}/image_color".format(cam), Image, self.image_callback, queue_size=1, callback_args="image_color")
            self.sub_image[cam + '/compressed'] = rospy.Subscriber("/avt_cameras/{}/image_color/compressed".format(cam), CompressedImage, self.image_callback, queue_size=1, callback_args="compressed")
            self.pub_detection[cam] = rospy.Publisher("/avt_cameras/{}/openpifpaf_detections".format(cam), Detection2DArray, queue_size=1)
            self.pub_pose[cam] = rospy.Publisher("avt_cameras/{}/openpifpaf_pose".format(cam), JointState, queue_size=1)
            if self.output_image:
                self.pub_image[cam] = rospy.Publisher("/avt_cameras/{}/openpifpaf_image/compressed".format(cam), CompressedImage, queue_size=1)
        
        self.bridge = CvBridge()

    
    def format_predictions(self, predictions):
        """ Given a list of Openpifpaf predicted object, format it to json. """
        pose_msg = JointState()
        
        for pred_id, pred in enumerate(predictions):
            pose_msg.name.extend([str(pred_id)]*pred.data.shape[0])
            pose_msg.position.extend(pred.data[:,0])
            pose_msg.velocity.extend(pred.data[:,1])
            pose_msg.effort.extend(pred.data[:,2])
            # res_dict["keypoints"] = pred.data.reshape(-1)
            # res_dict["bbox"] = pred.bbox()
            # res_dict["score"] = pred.score
            # res_dict["category_id"] = pred.category_id
            # dets.append(res_dict)
        return pose_msg


    def image_callback(self, msg, img_type="compressed"):
        cam = msg.header.frame_id
        rospy.loginfo("Received image from %s at: %d.%09ds", cam, msg.header.stamp.secs, msg.header.stamp.nsecs)

        if img_type == "compressed":
            np_arr = np.fromstring(msg.data, np.uint8)
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif img_type == "image_color":
            try:
                image_in = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)
                return
        
        ## ========== Image preprocessing
        image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)

        if self.image_scale < 1:
            width = int(image_in.shape[1] * self.image_scale)
            height = int(image_in.shape[0] * self.image_scale)
            dim = (width, height)
            image_in_resized = cv2.resize(image_in, dim, interpolation=cv2.INTER_AREA)
        else:
            image_in_resized = image_in

        predictions, gt_anns, image_meta = self.model.numpy_image(image_in_resized)
        pose_msg = self.format_predictions(predictions)

        if self.output_image:
            image_out = np.array(image_in)

        
        det2d_array = Detection2DArray()
        det2d_array.header = msg.header 
        for pred in predictions:
            output = pred.bbox()
            print(output)
            x1, y1, w, h = output
            if self.image_scale < 1:
                x1 = x1 / self.image_scale
                y1 = y1 / self.image_scale
                w = w / self.image_scale
                h = h / self.image_scale
            x2 = x1 + w
            y2 = y1 + h
            det2d = Detection2D()
            det2d.bbox.center.x = (x1 + x2) / 2.0
            det2d.bbox.center.y = (y1 + y2) / 2.0
            det2d.bbox.size_x = x2 - x1
            det2d.bbox.size_y = y2 - y1
            object_hypothesis = ObjectHypothesisWithPose()
            object_hypothesis.id = int(pred.category_id)
            object_hypothesis.score = pred.score
            det2d.results.append(object_hypothesis)
            det2d_array.detections.append(det2d)

            if self.output_image:
                if self.show_bbox:
                    cv2.rectangle(
                        image_out, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        color=self.color,
                        thickness = self.thickness,
                        lineType=cv2.LINE_AA)
                if self.show_label:
                    cls_name = pred.category_id
                    cls_conf = str(cls_name) + ": " + str(pred.score)[0:4]
                    draw_text(
                        image_out, 
                        cls_conf, 
                        font=0,
                        pos=(int(x1),int(y1)-10 if y1 > 10 else 10), 
                        font_scale=1.0, 
                        font_thickness=self.thickness,
                        text_color=self.color
                        )
                if self.show_skeleton:
                    keypoints = np.asarray(pred.data)
                    if self.image_scale < 1:
                        keypoints[:,0:2] = keypoints[:,0:2] / self.image_scale
                    draw_skeleton(image_out, keypoints.reshape(-1))
        # if self.output_image:
        #     cv2.waitKey(1)
        # print('frame:', msg.header.frame_id)
        self.pub_detection[cam].publish(det2d_array)
        self.pub_pose[cam].publish(pose_msg)
        if self.output_image:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            image_out_msg = self.bridge.cv2_to_compressed_imgmsg(image_out, 'jpg')
            image_out_msg.header = msg.header
            self.pub_image[cam].publish(image_out_msg)


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (int(x + text_w), int(y + text_h)), text_color_bg, -1)
    cv2.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness)

    return text_size

KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]

COCO_KEYPOINTS = [
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle',     # 16
]


def draw_skeleton(img, pifpaf_keypoints, show_conf=False):
    color_left = (0, 255, 255)
    color_right = (255, 0, 255)
    # openpifpaf keypoints format: (x, y, confidence)
    pp_kps = np.array(pifpaf_keypoints).reshape(-1,3)
    if len(pp_kps) == 0:
        return img
    # draw skeleton by connecting different keypoint by coco default
    for pair in KINEMATIC_TREE_SKELETON:
        partA = pair[0] -1
        partB = pair[1] -1
        # left
        color = color_left
        # right
        if partA % 2 ==0 and partB % 2 == 0:
            color = color_right
        # if confidence is not zero, the keypoint exist, otherwise the keypoint would be at (0,0)
        if  not np.isclose(pp_kps[partA, 2],  0) and not np.isclose(pp_kps[partB, 2],  0):
            cv2.line(img, pp_kps[partA,:2].astype(int), pp_kps[partB,:2].astype(int), color, 2)
    
    if show_conf:
        for kp_idx, kp in enumerate(pp_kps):
            if kp[2] > 0:
                cv2.putText(
                    img, 
                    str(kp[2])[0:4], 
                    (int(max(0, kp[0]-120*(kp_idx % 2) + 30)),
                    int(max(kp[1]-10, 0))), 
                    0, 0.9, color, 2, cv2.LINE_AA)

    return img   


def main():
    rospy.init_node('camera_openpifpaf_node')

    camera_openpifpaf_node = CameraOpenPifPafNode()
    
    rospy.spin()


if __name__ == "__main__":
    main()



