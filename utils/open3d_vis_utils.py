import open3d
import torch
import matplotlib
import numpy as np

from . import common_utils, box_utils


box_colormap = {
    'Car': (0, 1, 0),
    'Pedestrian': (0, 1, 1),
    'Cyclist': (1, 1, 0),
} # RGB


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None,
    point_colors=None, point_size=1.0, window_name='Open3D'):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(point_colors, torch.Tensor):
        point_colors = point_colors.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.asarray([0.4, 0.4, 0.4])

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is not None:
        pts.colors = open3d.utility.Vector3dVector(point_colors)
    else:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * 0.9)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, color=(1, 0, 0))
    if ref_boxes is not None:
        if ref_labels is not None:
            vis = draw_box(vis, ref_boxes, ref_labels)
        else:
            vis = draw_box(vis, ref_boxes, color=(0, 1, 0))

    vis.run()
    vis.destroy_window()


def draw_box(vis, boxes, labels=None, color=(1, 0, 0)):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        labels: [name]

    Returns:
    """
    for i in range(boxes.shape[0]):
        corners3d = box_utils.boxes_to_corners_3d(np.array([boxes[i]]))[0]
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [0, 5], [1, 4], # heading
        ])
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(corners3d)
        line_set.lines = open3d.Vector2iVector(edges)
        if labels is not None:
            line_set.paint_uniform_color(box_colormap[labels[i]])
        else:
            line_set.paint_uniform_color(color)
        vis.add_geometry(line_set)
    return vis
