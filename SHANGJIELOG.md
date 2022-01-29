## Ablation Experiments

| ID | Description                                       | Car 3D AP (R11)           | Pedestrian 3D AP (R11)    | Cyclist 3D AP (R11)       | Latency |
|:--:|:-------------------------------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------:|
| 01 | original pointpillars (with gt_sampling)          | 85.9041, 76.5959, 74.6299 | 53.5642, 48.0202, 44.3261 | 81.2495, 65.5292, 61.4776 | 19ms    |
| 02 | original pointpillars                             | 85.0795, 73.6429, 68.1080 | 46.3585, 42.6274, 39.6279 | 57.0723, 38.5380, 37.0831 | 19ms    |
 * The latency is tested on a single 1080Ti GPU.
