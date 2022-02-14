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


## Ablation Experiments for MVMM (pillar)

| ID | RV backbone                        | Pillar bridge          | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:----------------------------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 10 | xyzi -> DRB(4, 64)                 | PFE(64, 64)            | 57.4, 49.3, 44.6 | 29.6, 24.4, 20.1 | 47.1, 29.1, 27.9 | -       |
| 11 | xyzi -> DRB(4, 64)                 | xyz -> PFE(67, 64)     | 69.4, 55.0, 53.8 | 30.8, 28.7, 24.6 | 44.5, 30.8, 27.3 | -       |
| 12 | xyzi -> DRNet                      | PFE(64, 64)            | 63.1, 52.4, 51.1 | 32.2, 26.6, 24.6 | 40.5, 28.1, 27.2 | 33ms    |
| 13 | xyzi -> DRNet                      | xyz -> PFE(67, 64)     | 65.1, 54.1, 53.5 | 34.9, 29.7, 25.2 | 45.0, 30.5, 29.9 | 33ms    |
| 14 | xyzri -> DRNet                     | xyz -> PFE(67, 64)     | 69.5, 55.6, 55.0 | 35.9, 30.2, 26.1 | 45.8, 31.0, 27.9 | 33ms    |
| 15 | xyzrirgb -> DRNet                  | xyz -> PFE(67, 64)     | 68.2, 54.6, 54.0 | 29.4, 23.8, 23.2 | 43.9, 29.0, 28.5 | 33ms    |
| 16 | rgb -> DRNet                       | xyz -> PFE(67, 64)     | 65.7, 53.5, 52.3 | 29.7, 24.9, 23.6 | 43.5, 26.8, 26.6 | 33ms    |
| 24 | xyzi -> DRNet -> 4                 | xyz -> PFE(7, 64)      | 64.9, 53.1, 52.6 | 30.5, 25.9, 24.2 | 45.5, 31.2, 27.9 | 30ms    |
| 25 | xyzi -> DRNet -> 4(sm)             | xyz -> PFE(7, 64)      | 70.1, 60.9, 55.6 | 35.8, 30.4, 26.2 | 51.0, 30.8, 30.2 | 30ms    |
| 26 | rgb -> DRNet -> 4                  | xyz -> PFE(7, 64)      | 68.0, 54.4, 54.2 | 36.2, 30.9, 26.6 | 48.1, 33.2, 29.7 | 30ms    |
| 27 | rgb -> DRNet -> 4(sm)              | xyz -> PFE(7, 64)      | 68.2, 59.9, 55.2 | 31.8, 29.4, 25.3 | 50.8, 35.0, 33.9 | 30ms    |
| 28 | xyzri -> DRNet -> 4                | xyz -> PFE(7, 64)      | 69.2, 55.3, 54.7 | 30.7, 25.4, 24.2 | 48.1, 32.4, 29.2 | 30ms    |
| 29 | xyzri -> DRNet -> 4(sm)            | xyz -> PFE(7, 64)      | 71.7, 61.1, 56.2 | 36.7, 31.0, 26.6 | 52.5, 35.7, 34.2 | 30ms    |
| 30 | xyzrirgb -> DRNet -> 4             | xyz -> PFE(7, 64)      | 69.4, 56.1, 55.1 | 36.5, 31.4, 26.8 | 48.2, 30.9, 29.7 | 30ms    |
| 31 | xyzrirgb -> DRNet -> 4(sm)         | xyz -> PFE(7, 64)      | 69.8, 60.8, 55.7 | 33.0, 27.7, 26.0 | 48.3, 32.7, 28.7 | 30ms    |
| 32 | xyzri(norm) -> DRNet -> 4          | xyz -> PFE(7, 64)      | 67.9, 54.7, 54.3 | 28.6, 24.3, 22.6 | 41.3, 27.7, 27.3 | 30ms    |
| 33 | xyzri(norm) -> DRNet -> 4(sm)      | xyz -> PFE(7, 64)      | 70.5, 60.2, 55.5 | 35.8, 30.3, 25.7 | 48.8, 33.3, 30.0 | 30ms    |
| 34 | rgb(norm) -> DRNet -> 4            | xyz -> PFE(7, 64)      | 65.8, 54.2, 54.2 | 31.1, 28.2, 24.6 | 48.3, 31.7, 28.7 | 30ms    |
| 35 | rgb(norm) -> DRNet -> 4(sm)        | xyz -> PFE(7, 64)      | 70.3, 60.9, 55.8 | 34.4, 29.7, 25.5 | 53.2, 35.9, 34.9 | 30ms    |
| 36 | xyzrirgb(norm) -> DRNet -> 4       | xyz -> PFE(7, 64)      | 63.5, 52.9, 52.2 | 30.0, 25.0, 24.0 | 39.0, 26.8, 25.5 | 30ms    |
| 37 | xyzrirgb(norm) -> DRNet -> 4(sm)   | xyz -> PFE(7, 64)      | 68.5, 59.6, 55.2 | 37.4, 31.9, 27.3 | 51.9, 35.0, 31.5 | 30ms    |
| 38 | xyzri(norm)+rgb(norm) -> DualDRNet | xyz -> PFE(7, 64)      | 71.3, 61.1, 55.9 | 37.0, 31.5, 26.7 | 48.9, 32.9, 29.8 | 43ms    |
| 39 | xyzri+rgb(norm) -> DualDRNet       | xyz -> PFE(7, 64)      | 69.2, 60.5, 55.5 | 36.3, 31.5, 26.3 | 51.1, 34.0, 30.0 | 43ms    |
| 40 | xyzri+rgb -> DualDRNet             | xyz -> PFE(7, 64)      | 70.9, 60.6, 55.7 | 35.0, 29.4, 25.1 | 49.2, 33.0, 29.6 | 43ms    |

 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.


## Ablation Experiments for SECOND

| ID | Used features        | Data augmentation                 | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:--------------------:|:---------------------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 17 | xyzi                 | sampling, flip, rotation, scaling | 88.2, 78.5, 77.4 | 55.3, 51.4, 47.5 | 80.9, 67.6, 63.1 | 29ms    |
| 18 | occupancy            | sampling, flip, rotation, scaling | 87.7, 78.0, 76.7 | 51.6, 49.6, 45.2 | 80.3, 62.0, 58.8 | 29ms    |
| 19 | xyzi                 | -                                 | 78.1, 64.7, 64.0 | 44.1, 38.5, 33.7 | 55.1, 37.7, 36.5 | 29ms    |
| 20 | occupancy            | -                                 | 76.4, 64.3, 63.5 | 45.1, 35.3, 34.5 | 51.7, 36.2, 31.5 | 29ms    |
 
 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.


## Ablation Experiments for MVMM (voxel)

| ID | RV backbone                        | Voxel bridge           | Car              | Pedestrian       | Cyclist          | Latency |
|:--:|:----------------------------------:|:----------------------:|:----------------:|:----------------:|:----------------:|:-------:|
| 21 | xyzi -> DRNet -> 64                | VFE(1-2-3-4-5)         | 78.8, 65.3, 63.9 | 39.2, 32.8, 31.7 | 55.8, 37.9, 37.4 | -       |
| 22 | xyzi -> DRNet -> 3                 | VFE(0-1-2-3-4-5)       | 74.3, 65.3, 64.3 | 41.9, 36.8, 32.3 | 48.4, 32.6, 32.1 | -       |
| 23 | xyzi -> DRNet -> 3(sm)             | VFE(0-1-2-3-4-5)       | 79.2, 65.0, 63.9 | 41.9, 35.5, 34.1 | 52.6, 36.2, 31.5 | -       |
| 41 | xyzi(norm) -> DRNet -> 4(sm)       | VFE(0-1-2-3-4-5)       |
| 42 | xyzri(norm) -> DRNet -> 4(sm)      | VFE(0-1-2-3-4-5)       |
| 43 | rgb(norm) -> DRNet -> 4(sm)        | VFE(0-1-2-3-4-5)       |

 * The 3D AP with 11 Recall Positions (R11) is used to evaluate the models. The latency is tested on a single 1080Ti GPU.
