# ros_fairmot_package
# System Requirements: 
​
This repo targets Ubuntu Focal (20.04) and ROS2 Foxy 
​
Additionally, CUDA 11.1 is installed on the machine for use with the 3d object detection model 
​
ROS2 Foxy Installation: https://index.ros.org/doc/ros2/Installation/Foxy/
CUDA 11.1 Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004
​
# Steps to use: 
​
 
## 1. Clone repository 
​
```$ git clone git@github.com:jiannaliu01/ros_fairmot_package.git```
​
## 2. Download additional data from Oracle Server
​
``` 
$ cd ros_fairmot_package
​
$ mkdir data && cd data/ 
$ scp username@150.136.182.148:home/jiannaliu/fairmot_data/real.bag_0.db3 .
​
$ cd ../ && mkdir models && cd models/
$ scp username@150.136.182.148:home/jiannaliu/fairmot_data/kitti_full_60_epochs.pth .
```
​
## 3. Setup environment
```angular2
$ python3 -m venv ros_fairmot_package_env && source ros_fairmot_package_env/bin/activate
$ cd ros_fairmot_package/ && pip3 install -r requirements.txt
​
# SET CUDA visible devices env var (NOTE: this assumes you will want to use the 0-id GPU, run '$ nvidia-smi' to view GPUs)
$ export CUDA_VISIBLE_DEVICES=0
```
​
## 4. Build the ROS workspace and run the two nodes
```angular2
# For each node, run in separate terminal
$ colcon build
$ . install/setup.bash
​
# Then run the node
$ ros2 run ros_fairmot_package fairmot_node
```
​
## 5. Play rosbag
```angular2
$ ros2 bag play <path_to_data_dir>/real.bag_0.db3
```
