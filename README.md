Sunshine
========

> "Illuminating the world, information theoretically."

Installation
--------------------------

```bash
# Install rost-cli (including dependencies)
sudo apt-get update
sudo apt-get install libboost-all-dev libflann-dev libfftw3-dev libopencv-dev libsndfile1-dev cmake
git clone git@gitlab.com:warplab/rost-cli.git
pushd rost-cli
./install-rost.sh # check rost-cli README.md for latest install instructions
popd

# Install sunshine (including dependencies)
sudo apt-get install ros-melodic-desktop ros-melodic-perception libopencv libboost1.65-dev
cd $CATKIN_WS/src
git clone git@gitlab.com:warplab/ros/sunshine.git
cd $CATKIN_WS
catkin build sunshine
```

Quick Start
------------

Common launchfiles:
 - `desktop.launch`: Uses webcam feed (requires gstreamer to be installed)
 - `sentry.launch`: Uses data from the Sentry503 mission (requires `ds_msgs` dependency, see below)
 - TODO: Finish listing launchfiles


Dependencies
------------

Required:
- [rost-cli](https://gitlab.com/warplab/rost-cli)
- ROS (roscpp, rosbag)
- Catkin Packages: 
  - sensor_msgs
  - std_msgs
  - cv_bridge
  - image_transport
  - sunshine_msgs
  - geometry_msgs
  - tf2
  - tf2_ros
  - tf
- Boost 1.65+
- PCL
- OpenCV 3.2+

Optional:
- Eigen3
- [Limbo](https://github.com/resibots/limbo) and [NLOpt](https://github.com/stevengj/nlopt)
- [CLEAR](https://arxiv.org/abs/1902.02256) (ask [Stewart](mailto:sjamieson@whoi.edu))
- [ds_msgs](https://bitbucket.org/whoidsl/ds_msgs/src/master/) (WHOI Deep Submergence Lab ROS message definitions)