""" Process a rosbag with thermal human detector and filter interesting data. """

import os
import rosbag
import numpy as np
import cv2
import json
import argparse

import openpifpaf


def detect_bag(detector, bag_dir, bag_name, topic_list):
    bag_path = os.path.join(bag_dir, bag_name)
    bag = rosbag.Bag(bag_path)
    print('Processing rosbag', bag_path)
    # type = bag.get_type_and_topic_info()
    # print('\n All topics in the bag')
    # print(type.topics.keys())

    det_path = {}
    det_dict = {}
    det_name = {}
    for topic in topic_list:
        topic_name_new = '_'.join([item for item in topic.split('/') if item != ''])
        det_path[topic] = os.path.join(bag_dir, bag_name[0:-4] + '_' + topic_name_new + '_openpifpaf_det.json')
        det_dict[topic] = {}
        det_name[topic] = topic_name_new

    for i, (topic, msg, t) in enumerate(bag.read_messages()):
        if topic in topic_list:
            timestamp_string = '{:.9f}'.format(msg.header.stamp.to_time())
            # print('processing:', timestamp_string)
            # convert message into image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_in = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            predictions, gt_anns, image_meta =detector.numpy_image(image_in)

            dets = []
            for pred in predictions:
                res_dict = {}
                res_dict["keypoints"] = pred.data.reshape(-1).tolist()
                res_dict["bbox"] = np.array(pred.bbox()).tolist()
                res_dict["score"] = pred.score.item()
                res_dict["category_id"] = pred.category_id
                # print(type(res_dict["bbox"][0]))
                dets.append(res_dict)
            
            det_dict[topic][timestamp_string] = dets

    # write each small topic into a single file
    for topic in det_dict:
        if len(det_dict[topic].keys()) == 0:
            continue # skip empty topics

        with open(det_path[topic], 'w') as fp:
            json.dump(det_dict[topic], fp, indent=4)


def cmdline_args():
        # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    
    p.add_argument('bagfile_path', help='Path to the rosbag or rosbag dir')
    p.add_argument(
        '--topic_list', 
        nargs="*",  # 0 or more values expected => creates a list
        type=str, 
        default=['/camera/color/image_rect_color/compressed'], 
        help='list of topics for openpifpaf human detection')
    p.add_argument(
        '--model_name',
        default='shufflenetv2k30',
        type=str,
        help=" Models: \
            'mobilenetv2': \
            'mobilenetv3large': 58.4    34ms    15.0 MB  # This model gives error \
            'shufflenetv2k16':  68.1    40ms    38.9 MB \
            'shufflenetv2k30':  71.8    81ms    115.0 MB ")
    # p.add_argument('--sample_frequency',  default=1, help='sample frequency for detection')
    # p.add_argument('--output_json', action='store_true', help='output json for detection')
    return(p.parse_args())


def main():
    args = cmdline_args()
    if args.bagfile_path is None:
        print("Error, no bagile path.")
        exit(-1)
    else:
        print("bagfile path", args.bagfile_path)
    
    if args.topic_list is None:
        print("Error, no topic list for detection.")
        exit(-1)
    else:
        print("topic list:", args.topic_list)
    
    if args.model_name is None:
        print("Error, no topic list for detection.")
        exit(-1)
    else:
        print("model name:", args.model_name)
    
    bag_dir = args.bagfile_path
    topic_list = args.topic_list
    model_name = args.model_name

    if bag_dir.endswith('.bag'):
        bag_dir_list = bag_dir.split('/')
        bag_dir = ('/').join(bag_dir_list[0:-1])
        file_name = bag_dir_list[-1]
    else:
        file_name = None

    detector = openpifpaf.Predictor(checkpoint=model_name)

    if not file_name is None:
        detect_bag(detector, bag_dir, file_name, topic_list)
    else:
        for file_name in sorted(os.listdir(bag_dir)):
            if not file_name.endswith('bag'):
                continue
            
            detect_bag(detector, bag_dir, file_name, topic_list)


if __name__ == '__main__':
    main()