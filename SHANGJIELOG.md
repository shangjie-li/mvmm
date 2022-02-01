## Ablation Experiments

| ID | Model        | Used features                           | Data augmentation                  | Car 3D AP (R11)           | Pedestrian 3D AP (R11)    | Cyclist 3D AP (R11)       | Latency |
|:--:|:------------:|----------------------------------------:|-----------------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------:|
| 01 | pointpillars | xyz, i, xyz_to_cluster, xyz_to_center   | sampling, flip, rotation, scaling  | 85.9041, 76.5959, 74.6299 | 53.5642, 48.0202, 44.3261 | 81.2495, 65.5292, 61.4776 | 19ms    |
| 02 | pointpillars | xyz, i, xyz_to_cluster, xyz_to_center   | flip, rotation, scaling            | 85.0795, 73.6429, 68.1080 | 46.3585, 42.6274, 39.6279 | 57.0723, 38.5380, 37.0831 | 19ms    |
| 03 | pointpillars | xyz, i                                  | flip, rotation, scaling            | 83.5454, 72.4637, 67.5675 | 45.6709, 42.1264, 41.1873 | 56.1153, 41.0372, 37.6368 | 19ms    |
| 04 | pointpillars | xyz, i, xyz_to_cluster                  | flip, rotation, scaling            | 82.9791, 72.2482, 67.1901 | 44.9506, 41.7879, 38.6402 | 54.6337, 38.0115, 36.5259 | 19ms    |
| 05 | pointpillars | xyz, i, xyz_to_center                   | flip, rotation, scaling            | 84.5645, 72.6793, 67.8071 | 46.1848, 44.0710, 42.1585 | 57.0083, 38.3209, 36.6195 | 19ms    |
| 06 | pointpillars | xyz, i                                  | -                                  | 64.6726, 53.4601, 53.5857 | 32.4545, 30.3049, 25.6269 | 49.4620, 34.2456, 33.8225 | 19ms    |
| 07 | pointpillars | xyz, i                                  | flip                               | 74.4864, 64.5149, 62.7211 | 42.7343, 36.7054, 33.1361 | 56.4004, 36.8465, 35.0350 | 19ms    |
| 08 | pointpillars | xyz, i                                  | rotation                           | 76.7034, 66.8348, 65.1285 | 44.2698, 39.0987, 37.8425 | 60.5570, 41.2111, 39.3014 | 19ms    |
| 09 | pointpillars | xyz, i                                  | scaling                            | 74.2255, 63.5115, 57.2467 | 35.9211, 31.0324, 29.9227 | 54.5309, 33.8602, 31.4590 | 19ms    |
 * The latency is tested on a single 1080Ti GPU.
