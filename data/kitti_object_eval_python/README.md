## Acknowledgement
 - This repository is developed based on [traveller59](https://github.com/traveller59/kitti-object-eval-python)'s work.

## Dependencies
 - Only support python 3.6+, need `numpy`, `skimage`, `numba`, `fire`. If you have Anaconda, just install `cudatoolkit` in Anaconda.

## Usage
 - Evaluate your detection results
   ```
   python evaluate.py --result_dir=your/result/folder --label_dir=your/label/folder --label_split_file=/path/to/val.txt --current_classes=0,1,2 --use_ldf_eval
   ```

