## Ablation Experiments

| ID | Model | Used features          | Data augmentation                 | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:-----:|:----------------------:|:---------------------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 01 | PP    | xyz, i, xyz_cl, xyz_ce | sampling, flip, rotation, scaling | 85.9, 76.5, 74.6 | 53.5, 48.0, 44.3 | 81.2, 65.5, 61.4 | 19ms    |
| 02 | PP    | xyz, i, xyz_cl, xyz_ce | flip, rotation, scaling           | 85.0, 73.6, 68.1 | 46.3, 42.6, 39.6 | 57.0, 38.5, 37.0 | 19ms    |
| 03 | PP    | xyz, i                 | flip, rotation, scaling           | 83.5, 72.4, 67.5 | 45.6, 42.1, 41.1 | 56.1, 41.0, 37.6 | 19ms    |
| 04 | PP    | xyz, i, xyz_cl         | flip, rotation, scaling           | 82.9, 72.2, 67.1 | 44.9, 41.7, 38.6 | 54.6, 38.0, 36.5 | 19ms    |
| 05 | PP    | xyz, i, xyz_ce         | flip, rotation, scaling           | 84.5, 72.6, 67.8 | 46.1, 44.0, 42.1 | 57.0, 38.3, 36.6 | 19ms    |
| 06 | PP    | xyz, i                 | -                                 | 64.6, 53.4, 53.5 | 32.4, 30.3, 25.6 | 49.4, 34.2, 33.8 | 19ms    |
| 07 | PP    | xyz, i                 | flip                              | 74.4, 64.5, 62.7 | 42.7, 36.7, 33.1 | 56.4, 36.8, 35.0 | 19ms    |
| 08 | PP    | xyz, i                 | rotation                          | 76.7, 66.8, 65.1 | 44.2, 39.0, 37.8 | 60.5, 41.2, 39.3 | 19ms    |
| 09 | PP    | xyz, i                 | scaling                           | 74.2, 63.5, 57.2 | 35.9, 31.0, 29.9 | 54.5, 33.8, 31.4 | 19ms    |
| 10 | MVMM  | xyz, i                 | -                                 | 67.6, 54.6, 54.4 | 31.2, 25.5, 24.5 | 50.4, 34.0, 33.2 |         |
 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.
