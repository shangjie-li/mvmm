import os
import argparse
import yaml
from pathlib import Path
import time
import math
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data.kitti_dataset import KITTIDataset
from mvmm import build_model
from helpers.checkpoint_helper import load_checkpoint
from utils.kitti_calibration_utils import parse_calib
from utils.point_cloud_utils import get_fov_flag
from utils.point_cloud_utils import mask_points_by_range
from utils.opencv_vis_utils import box_colormap
from utils.opencv_vis_utils import normalize_img
from utils.opencv_vis_utils import draw_boxes3d
from utils.numpy_pc2_utils import pointcloud2_to_xyzi_array


image_lock = threading.Lock()
lidar_lock = threading.Lock()


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ResNet_VFE.yaml',
                        help='path to the config file')
    parser.add_argument('--display', action='store_true', default=False,
                        help='whether to show the RGB image')
    parser.add_argument('--print', action='store_true', default=False,
                        help='whether to print results in the txt file')
    parser.add_argument('--score_thresh', type=float, default=None,
                        help='score threshold for filtering detections')
    parser.add_argument('--nms_thresh', type=float, default=None,
                        help='NMS threshold for filtering detections')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint')
    parser.add_argument('--current_classes', type=str, default='0,1,2',
                        help='a filter for desired classes, e.g., 0,1,2 (split by a comma)')
    parser.add_argument('--sub_image', type=str, default='/kitti/camera_color_left/image_raw',
                        help='image topic to subscribe')
    parser.add_argument('--sub_lidar', type=str, default='/kitti/velo/pointcloud',
                        help='lidar topic to subscribe')
    parser.add_argument('--pub_marker', type=str, default='/result',
                        help='marker topic to publish')
    parser.add_argument('--frame_rate', type=int, default=10,
                        help='working frequency')
    parser.add_argument('--frame_id', type=str, default=None,
                        help='frame id for ROS message (same as lidar topic by default, which is `velo_link`)')
    parser.add_argument('--calib_file', type=str, default='data/kitti/testing/calib/000000.txt',
                        help='path to the calibration file')
    args = parser.parse_args()
    return args


def publish_marker_msg(pub, boxes, labels, scores, frame_id, frame_rate, color_map):
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
        marker.color.a = scores[i]  # 0 for invisible
        
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


def print_info(frame, stamp, delay, boxes, labels, scores, file_name='result.txt'):
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


def create_input_data(dataset, image, points, calib, frame_id):
    image = image[:, :, ::-1].astype(np.float32) / 255.0  # ndarray of float32, [H, W, 3], RGB image
    image_shape = np.array(image.shape[:2], dtype=np.int32)
    mask = get_fov_flag(points, image_shape, calib)
    points = points[mask]

    pts_img, pts_rect_depth = calib.lidar_to_img(points[:, 0:3])
    pts_img = pts_img.astype(np.int)
    rgb = image[pts_img[:, 1], pts_img[:, 0], :]
    colored_points = np.concatenate([points, rgb], axis=1)

    data_dict = {
        'frame_id': frame_id,
        'colored_points': colored_points,
        'calib': calib,
        'image_shape': image_shape,
    }
    data_dict['colored_points'] = mask_points_by_range(data_dict['colored_points'], dataset.point_cloud_range)

    points = data_dict['colored_points']

    xs = points[:, 0:1]
    ys = points[:, 1:2]
    zs = points[:, 2:3]
    intensities = points[:, 3:4]
    colors = points[:, 4:7]

    xmin, ymin, zmin, xmax, ymax, zmax = dataset.point_cloud_range
    xs = (xs - xmin) / (xmax - xmin)
    ys = (ys - ymin) / (ymax - ymin)
    zs = (zs - zmin) / (zmax - zmin)

    point_features = np.concatenate([xs, ys, zs, intensities, colors], axis=1)
    data_dict['range_image'] = dataset.range_convertor.get_range_image(points, point_features)

    return data_dict


def image_callback(image):
    global image_header, image_frame
    image_lock.acquire()
    if image_header is None:
        image_header = image.header
    image_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)  # ndarray of uint8, [H, W, 3], BGR image
    image_lock.release()


def lidar_callback(lidar):
    global lidar_header, lidar_frame
    lidar_lock.acquire()
    if lidar_header is None:
        lidar_header = lidar.header
    lidar_frame = pointcloud2_to_xyzi_array(lidar, remove_nans=True)
    lidar_lock.release()


def timer_callback(event):
    image_lock.acquire()
    cur_image = image_frame.copy()
    image_lock.release()

    lidar_lock.acquire()
    cur_lidar = lidar_frame.copy()
    lidar_lock.release()

    global frame
    frame += 1
    start = time.time()

    data_dict = create_input_data(dataset, cur_image, cur_lidar, calib, args.frame_id)

    batch_dict = dataset.collate_batch([data_dict])
    batch_dict = dataset.load_data_to_gpu(batch_dict, device)

    with torch.no_grad():
        batch_dict = model(
            batch_dict, score_thresh=cfg['tester']['score_thresh'], nms_thresh=cfg['tester']['nms_thresh']
        )

    pred_boxes = batch_dict['pred_boxes'][0].cpu().numpy()  # [M, 7]
    pred_classes = batch_dict['pred_classes'][0].cpu().numpy()  # [M]
    pred_scores = batch_dict['pred_scores'][0].cpu().numpy()  # [M]

    pred_cls_ids = pred_classes.astype(np.int) - 1
    indices = [i for i, cls_id in enumerate(pred_cls_ids) if cls_id in args.current_classes]
    pred_boxes, pred_cls_ids, pred_scores = pred_boxes[indices], pred_cls_ids[indices], pred_scores[indices]
    pred_names = [dataset.class_names[k] for k in pred_cls_ids]

    publish_marker_msg(pub_marker, pred_boxes, pred_names, pred_scores, args.frame_id, args.frame_rate, box_colormap)

    if args.display:
        image = normalize_img(cur_image)
        image = draw_boxes3d(image, calib, pred_boxes, pred_names)
        if not display(image, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")

    if args.print:
        cur_stamp = rospy.Time.now()
        cur_stamp = cur_stamp.secs + 0.000000001 * cur_stamp.nsecs
        delay = round(time.time() - start, 3)
        print_info(frame, cur_stamp, delay, pred_boxes, pred_names, pred_scores, result_file)


if __name__ == '__main__':
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.score_thresh is not None:
        cfg['tester']['score_thresh'] = args.score_thresh
    if args.nms_thresh is not None:
        cfg['tester']['nms_thresh'] = args.nms_thresh
    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    args.current_classes = list(map(int, args.current_classes.split(',')))

    rospy.init_node("mvmm", anonymous=True, disable_signals=True)
    frame = 0

    if cfg['dataset']['type'] == 'KITTI':
        dataset = KITTIDataset(cfg['dataset'], split=cfg['tester']['split'], is_training=False, augment_data=False)
    else:
        raise NotImplementedError

    model = build_model(cfg['model'], dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    assert os.path.exists(cfg['tester']['checkpoint'])
    load_checkpoint(
        file_name=cfg['tester']['checkpoint'],
        model=model,
        optimizer=None,
        map_location=device,
        logger=None,
    )

    torch.set_grad_enabled(False)
    model.eval()

    calib_file = Path(args.calib_file)
    assert os.path.exists(calib_file)
    calib = parse_calib(calib_file)

    image_header, image_frame = None, None
    lidar_header, lidar_frame = None, None
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1, buff_size=52428800)
    rospy.Subscriber(args.sub_lidar, PointCloud2, lidar_callback, queue_size=1, buff_size=52428800)
    print('==> Waiting for topic %s and %s...' % (args.sub_image, args.sub_lidar))
    while image_frame is None or lidar_frame is None:
        time.sleep(0.1)
    print('==> Done.')

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
