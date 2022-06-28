import argparse
import glob
from pathlib import Path
import time
import os
import sys
import math
import threading

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from mvmm import build_network, load_data_to_gpu
from utils import common_utils, calibration_kitti
from utils import range_image_utils, augmentor_utils
from utils import opencv_vis_utils
from utils import numpy_pc2


image_lock = threading.Lock()
lidar_lock = threading.Lock()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config/ResNet_VFE.yaml',
                        help='specify the config file')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to display results onto the RGB image')
    parser.add_argument('--print', action='store_true', default=False,
                        help='whether to print results in the txt file')
    parser.add_argument('--score_thresh', type=float, default=0.1,
                        help='specify the score threshold')
    parser.add_argument('--sub_image', type=str, default='/kitti/camera_color_left/image_raw',
                        help='the image topic to subscribe')
    parser.add_argument('--sub_lidar', type=str, default='/kitti/velo/pointcloud',
                        help='the lidar topic to subscribe')
    parser.add_argument('--pub_marker', type=str, default='/result',
                        help='the marker topic to publish')
    parser.add_argument('--frame_rate', type=int, default=10,
                        help='working frequency')
    parser.add_argument('--frame_id', type=str, default=None,
                        help='frame id for ROS msg (same as lidar topic by default)')
    parser.add_argument('--calib_file', type=str, default='data/kitti/testing/calib/000000.txt',
                        help='specify the calibration file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    if args.score_thresh:
        cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh

    return args, cfg


def publish_marker_msg(pub, boxes, labels, frame_id, frame_rate, color_map):
    markerarray = MarkerArray()
    for i in range(boxes.shape[0]):
        marker = Marker()
        marker.header = Header()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = labels[i]
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = boxes[i][0]
        marker.pose.position.y = boxes[i][1]
        marker.pose.position.z = boxes[i][2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(0.5 * boxes[i][6])
        marker.pose.orientation.w = math.cos(0.5 * boxes[i][6])
        
        marker.scale.x = boxes[i][3]
        marker.scale.y = boxes[i][4]
        marker.scale.z = boxes[i][5]
    
        marker.color.r = color_map[labels[i]][2] / 255.0
        marker.color.g = color_map[labels[i]][1] / 255.0
        marker.color.b = color_map[labels[i]][0] / 255.0
        marker.color.a = 0.75 # 0 for invisible
        
        marker.lifetime = rospy.Duration(1 / frame_rate)
        markerarray.markers.append(marker)
        
    pub.publish(markerarray)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, labels, scores, boxes, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        box = boxes[i]
        info_str = 'box:%.2f %.2f %.2f %.2f %.2f %.2f %.2f  score:%.2f  label:%s' % (
            box[0], box[1], box[2], box[3], box[4], box[5], box[6], scores[i], labels[i]
        )
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def image_callback(image):
    global image_header, image_frame
    image_lock.acquire()
    if image_header is None:
        image_header = image.header
    image_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
    image_lock.release()


def lidar_callback(lidar):
    global lidar_header, lidar_frame
    lidar_lock.acquire()
    if lidar_header is None:
        lidar_header = lidar.header
    lidar_frame = numpy_pc2.pointcloud2_to_xyzi_array(lidar, remove_nans=True)
    lidar_lock.release()


def timer_callback(event):
    global image_frame
    image_lock.acquire()
    cur_image = image_frame.copy()
    image_lock.release()
    
    global lidar_frame
    lidar_lock.acquire()
    cur_lidar = lidar_frame.copy()
    lidar_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    d = demo_dataset
    frame_id = args.frame_id
    image_shape = np.array(cur_image.shape[:2], dtype=np.int32)
    
    points = cur_lidar[d.get_fov_flag(cur_lidar, image_shape, calib)]
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect) # [N', 2], [N']
    pts_img = pts_img.astype(np.int)
    rgb = cur_image[pts_img[:, 1], pts_img[:, 0], :] # [N', 3]
    points = np.concatenate([points, rgb], axis=1)
    
    data_dict = {
        'frame_id': frame_id,
        'colored_points': points, # (N, 7)
        'calib': calib,
        'image_shape': image_shape,
    }
    mask = common_utils.mask_points_by_range(data_dict['colored_points'], d.point_cloud_range)
    data_dict['colored_points'] = data_dict['colored_points'][mask]
    points = data_dict['colored_points']
    features = d.get_point_features(points, d.used_feature_list, normalize=True)
    data_dict['range_image'] = d.range_convertor.get_range_image(points, features)
    data_dict = d.collate_batch([data_dict])
    
    with torch.no_grad():
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
    
    ref_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
    ref_labels = [cfg.CLASS_NAMES[j - 1] for j in pred_dicts[0]['pred_labels']]
    ref_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
    publish_marker_msg(pub_marker, ref_boxes, ref_labels,
        frame_id=args.frame_id, frame_rate=args.frame_rate,
        color_map=opencv_vis_utils.box_colormap
    )
    
    if args.display:
        if ref_boxes is not None:
            cur_image = opencv_vis_utils.draw_box(cur_image, calib, ref_boxes, ref_labels)
        if not display(cur_image, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")
    
    if args.print:
        cur_stamp = rospy.Time.now()
        cur_stamp = cur_stamp.secs + 0.000000001 * cur_stamp.nsecs
        delay = round(time.time() - start, 3)
        print_info(frame, cur_stamp, delay, ref_labels, ref_scores, ref_boxes, result_file)


if __name__ == '__main__':
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of MVMM in ROS-------------------------')
    
    rospy.init_node("mvmm", anonymous=True, disable_signals=True)
    frame = 0
    
    demo_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger,
        load_infos=False
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    calib_file = Path(args.calib_file)
    assert calib_file.exists(), 'Calibration file not found: %s' % calib_file
    calib = calibration_kitti.Calibration(calib_file)
    
    image_header, image_frame = None, None
    lidar_header, lidar_frame = None, None
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1,
        buff_size=52428800)
    rospy.Subscriber(args.sub_lidar, PointCloud2, lidar_callback, queue_size=1,
        buff_size=52428800)
    while image_frame is None or lidar_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s and %s...' % (args.sub_image, args.sub_lidar))
    print('  Done.\n')
    
    if args.frame_id is None:
        args.frame_id = lidar_header.frame_id
    
    if args.display:
        win_h, win_w = image_frame.shape[0], image_frame.shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)
    
    if args.print:
        result_file = 'result.txt'
        with open(result_file, 'w') as fob:
            fob.seek(0)
            fob.truncate()
    
    pub_marker = rospy.Publisher(args.pub_marker, MarkerArray, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)
    
    rospy.spin()

