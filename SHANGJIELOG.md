## Ablation Experiments for PointPillars

| ID | Used features        | Data augmentation                 | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:--------------------:|:---------------------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 01 | xyzi, xyz_cl, xyz_ce | sampling, flip, rotation, scaling | 85.9, 76.5, 74.6 | 53.5, 48.0, 44.3 | 81.2, 65.5, 61.4 | 19ms    |
| 02 | xyzi, xyz_cl, xyz_ce | flip, rotation, scaling           | 85.0, 73.6, 68.1 | 46.3, 42.6, 39.6 | 57.0, 38.5, 37.0 | 19ms    |
| 03 | xyzi                 | flip, rotation, scaling           | 83.5, 72.4, 67.5 | 45.6, 42.1, 41.1 | 56.1, 41.0, 37.6 | 19ms    |
| 04 | xyzi, xyz_cl         | flip, rotation, scaling           | 82.9, 72.2, 67.1 | 44.9, 41.7, 38.6 | 54.6, 38.0, 36.5 | 19ms    |
| 05 | xyzi, xyz_ce         | flip, rotation, scaling           | 84.5, 72.6, 67.8 | 46.1, 44.0, 42.1 | 57.0, 38.3, 36.6 | 19ms    |
| 06 | xyzi                 | -                                 | 64.6, 53.4, 53.5 | 32.4, 30.3, 25.6 | 49.4, 34.2, 33.8 | 19ms    |
| 07 | xyzi                 | flip                              | 74.4, 64.5, 62.7 | 42.7, 36.7, 33.1 | 56.4, 36.8, 35.0 | 19ms    |
| 08 | xyzi                 | rotation                          | 76.7, 66.8, 65.1 | 44.2, 39.0, 37.8 | 60.5, 41.2, 39.3 | 19ms    |
| 09 | xyzi                 | scaling                           | 74.2, 63.5, 57.2 | 35.9, 31.0, 29.9 | 54.5, 33.8, 31.4 | 19ms    |

 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.


## Ablation Experiments for MVMM

| ID | RV features | RV backbone    | PV features | PV bridge      | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:-----------:|:--------------:|:-----------:|:--------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 10 | xyzi        | DRB(4, 64)     | -           | Linear(64, 64) | 57.4, 49.3, 44.6 | 29.6, 24.4, 20.1 | 47.1, 29.1, 27.9 | -       |
| 11 | xyzi        | DRB(4, 64)     | xyz         | Linear(67, 64) | 69.4, 55.0, 53.8 | 30.8, 28.7, 24.6 | 44.5, 30.8, 27.3 | -       |
| 12 | xyzi        | 6DDRB, 4UDRB   | -           | Linear(64, 64) | 63.1, 52.4, 51.1 | 32.2, 26.6, 24.6 | 40.5, 28.1, 27.2 | 33ms    |
| 13 | xyzi        | 6DDRB, 4UDRB   | xyz         | Linear(67, 64) | 65.1, 54.1, 53.5 | 34.9, 29.7, 25.2 | 45.0, 30.5, 29.9 | 33ms    |
| 14 | xyzri       | 6DDRB, 4UDRB   | xyz         | Linear(67, 64) | 69.5, 55.6, 55.0 | 35.9, 30.2, 26.1 | 45.8, 31.0, 27.9 | 33ms    |
| 15 | xyzri, rgb  | 6DDRB, 4UDRB   | xyz         | Linear(67, 64) | 68.2, 54.6, 54.0 | 29.4, 23.8, 23.2 | 43.9, 29.0, 28.5 | 33ms    |
| 16 | rgb         | 6DDRB, 4UDRB   | xyz         | Linear(67, 64) | 65.7, 53.5, 52.3 | 29.7, 24.9, 23.6 | 43.5, 26.8, 26.6 | 33ms    |

 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.


## Ablation Experiments for SECOND

| ID | Used features        | Data augmentation                 | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:--------------------:|:---------------------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 17 | xyzi                 | sampling, flip, rotation, scaling | 88.2, 78.5, 77.4 | 55.3, 51.4, 47.5 | 80.9, 67.6, 63.1 | 29ms    |
| 18 | occupancy            | sampling, flip, rotation, scaling | 87.7, 78.0, 76.7 | 51.6, 49.6, 45.2 | 80.3, 62.0, 58.8 | 29ms    |
| 19 | xyzi                 | -                                 | 78.1, 64.7, 64.0 | 44.1, 38.5, 33.7 | 55.1, 37.7, 36.5 | 29ms    |
| 20 | occupancy            | -                                 | 76.4, 64.3, 63.5 | 45.1, 35.3, 34.5 | 51.7, 36.2, 31.5 | 29ms    |
 
 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.
 
 
## Ablation Experiments for MVMM

| ID | RV backbone                        | Voxel backbone         | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:----------------------------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 21 | xyzi -> 6DDRB, 4UDRB -> 64         | 1-2-3-4-out            | 78.8, 65.3, 63.9 | 39.2, 32.8, 31.7 | 55.8, 37.9, 37.4 |
| 22 | xyzi -> 6DDRB, 4UDRB -> 3          | input-1-2-3-4-out      | 74.3, 65.3, 64.3 | 41.9, 36.8, 32.3 | 48.4, 32.6, 32.1 |
| 23 | xyzi -> 6DDRB, 4UDRB -> 3(softmax) | input-1-2-3-4-out      | 79.2, 65.0, 63.9 | 41.9, 35.5, 34.1 | 52.6, 36.2, 31.5 |
