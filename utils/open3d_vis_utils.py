import open3d
import numpy as np

from utils.box_utils import boxes3d_to_corners3d


box_colormap = {
    'Car': (0, 1, 0),
    'Pedestrian': (0, 1, 1),
    'Cyclist': (1, 1, 0),
}  # RGB


def draw_scene(points, boxes3d=None, names=None, point_colors=None, point_size=1.0, window_name='points'):
    """
    Show lidar point clouds with 3D boxes.

    Args:
        points: ndarray of float32, [N, 3], (x, y, z) in lidar coordinates
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading) in lidar coordinates
        names: list of str, name of each object
        point_colors: ndarray of float32, [N, 3], (r, g, b) values (between 0 and 1)
        point_size: float
        window_name: str

    Returns:

    """
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

    if boxes3d is not None:
        vis = draw_boxes3d(vis, boxes3d, names)

    vis.run()
    vis.destroy_window()


def draw_boxes3d(vis, boxes3d, names=None, color=(0, 1, 0)):
    """
    Draw 3D boxes as following in lidar point clouds.
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        vis: open3d.visualization.Visualizer
        boxes3d: ndarray of float32, [N, 7], (x, y, z, l, w, h, heading) in lidar coordinates
        names: list of str, name of each object
        color: tuple

    Returns:
        vis: open3d.visualization.Visualizer

    """
    corners = boxes3d_to_corners3d(boxes3d)  # [N, 8, 3]
    for i in range(boxes3d.shape[0]):
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
            [0, 5], [1, 4],  # heading
        ])

        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(corners[i])
        line_set.lines = open3d.Vector2iVector(edges)

        if names is not None:
            color = box_colormap[names[i]]
        line_set.paint_uniform_color(color)

        vis.add_geometry(line_set)

    return vis
