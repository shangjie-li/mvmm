import torch


class AnchorGenerator():
    def __init__(self, anchor_list, point_cloud_range):
        super().__init__()
        
        self.point_cloud_range = point_cloud_range
        
        self.anchor_class_names = [cfg['name'] for cfg in anchor_list]
        self.sizes_per_class = [cfg['sizes'] for cfg in anchor_list]
        self.rotations_per_class = [cfg['rotations'] for cfg in anchor_list]
        self.z_centers_per_class = [cfg['z_centers'] for cfg in anchor_list]
        
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for cfg in anchor_list:
            self.matched_thresholds[cfg['name']] = cfg['matched_threshold']
            self.unmatched_thresholds[cfg['name']] = cfg['unmatched_threshold']

    def generate_anchors(self, grid_size_per_class):
        all_anchors = []
        num_anchors_per_location = []
        
        for grid_size, sizes, rotations, z_centers in zip(grid_size_per_class, self.sizes_per_class, self.rotations_per_class, self.z_centers_per_class):
            num_anchors_per_location.append(len(sizes) * len(rotations) * len(z_centers))
            
            x_stride = (self.point_cloud_range[3] - self.point_cloud_range[0]) / (grid_size[0] - 1)
            y_stride = (self.point_cloud_range[4] - self.point_cloud_range[1]) / (grid_size[1] - 1)
            x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.point_cloud_range[0] + x_offset, self.point_cloud_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
            ).cuda()
            y_shifts = torch.arange(
                self.point_cloud_range[1] + y_offset, self.point_cloud_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
            ).cuda()
            z_shifts = x_shifts.new_tensor(z_centers)
            
            num_sizes = sizes.__len__()
            anchor_sizes = x_shifts.new_tensor(sizes)

            num_rotations = rotations.__len__()
            anchor_rotations = x_shifts.new_tensor(rotations)
            
            x_shifts, y_shifts, z_shifts = torch.meshgrid([x_shifts, y_shifts, z_shifts])  # [nx, ny, nz]
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [nx, ny, nz, 3]
            
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, num_sizes, 1)
            anchor_sizes = anchor_sizes.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_sizes), dim=-1)  # [nx, ny, nz, num_sizes, 6]
            
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_rotations, 1)
            anchor_rotations = anchor_rotations.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_sizes, 1, 1])
            anchors = torch.cat((anchors, anchor_rotations), dim=-1)  # [nx, ny, nz, num_sizes, num_rotations, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()  # [nz, ny, nx, num_sizes, num_rotations, 7]
            all_anchors.append(anchors)
            
        return all_anchors, num_anchors_per_location
