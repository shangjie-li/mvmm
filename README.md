# mvmm

Implementation of MVMM in PyTorch for KITTI 3D Object Detetcion

## Acknowledgement
 - This repository is developed based on [open-mmlab](https://github.com/open-mmlab/OpenPCDet)'s work.

## Installation
 - Clone this repository
   ```
   git clone git@github.com:shangjie-li/mvmm.git
   ```
 - Install PyTorch environment with Anaconda (Tested on Ubuntu 16.04 & CUDA 10.2)
   ```
   conda create -n pcdet.v0.5.0 python=3.6
   conda activate pcdet.v0.5.0
   cd mvmm
   pip install -r requirements.txt
   ```
 - Install spconv
   ```
   pip install spconv-cu102
   ```
 - Compile external modules
   ```
   cd mvmm
   python setup.py develop
   ```
 - Install visualization tools
   ```
   pip install mayavi
   pip install pyqt5
   pip install open3d-python
   pip install opencv-python
   ```

## KITTI3D Dataset (41.5GB)
 - Download KITTI3D dataset: [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip), [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip), [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip) and [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip).
 - Download [road plane](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) for data augmentation.
 - Organize the downloaded files as follows
   ```
   mvmm
   ├── data
   │   ├── kitti
   │   │   │── ImageSets
   │   │   │   ├──test.txt & train.txt & trainval.txt & val.txt
   │   │   │── training
   │   │   │   ├──calib & velodyne & label_2 & image_2 & planes
   │   │   │── testing
   │   │   │   ├──calib & velodyne & image_2
   ├── layers
   ├── utils
   ```
 - Generate ground truth databases and data infos by running the following command
   ```
   # This will create two database dirs and six info files in mvmm/data/kitti (Take 20 mins).
   cd mvmm
   python -m data.kitti_dataset create_kitti_infos
   ```
 - Display the dataset
   ```
   # Display the training dataset with data augmentation
   python dataset_player.py --training --data_augmentation --show_boxes
   
   # Display the testing dataset
   python dataset_player.py --show_boxes
   ```

## Demo
 - Run the demo with a trained model
   ```
   # Run on the testing dataset
   python demo.py --ckpt=path_to_your_ckpt
   
   # Run on a single sample from the testing dataset
   python demo.py --ckpt=path_to_your_ckpt --sample_idx=000008
   ```

## Training
 - Train your model using the following commands
   ```
   python train.py
   ```

## Evaluation
 - Evaluate your model using the following commands
   ```
   # Run on the testing dataset
   python test.py --ckpt=path_to_your_ckpt
   
   # Run on the testing dataset and display
   python test.py --ckpt=path_to_your_ckpt --display
   
   # Run on the testing dataset and save results of KITTI3D format
   python test.py --ckpt=path_to_your_ckpt --save_to_file
   ```

