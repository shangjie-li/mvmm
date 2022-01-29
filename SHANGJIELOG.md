## Ablation Experiments

| ID | Description                                       | Car 3D AP (R11)           | Pedestrian 3D AP (R11)    | Cyclist 3D AP (R11)       | Latency |
|:--:|:-------------------------------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------:|
| 01 | original pointpillars (with gt_sampling)          | 86.9367, 77.0562, 75.3498 | 56.0253, 49.8334, 46.3034 | 82.4460, 64.8526, 61.6534 | 18ms    |
| 02 | original pointpillars                             | 84.5152, 72.9657, 67.7584 | 46.4326, 42.8333, 39.3322 | 54.5532, 39.5813, 36.5860 | 18ms    |
 * The latency is tested on a single 1080Ti GPU.
