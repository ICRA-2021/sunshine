#!/bin/sh
tmux new-session -d 'cd ~/warp_ws && source devel/setup.bash && roslaunch warpauv_config rgbdepth_to_pointcloud.launch camera_ns:=/warpauv_1/cameras/mapping_cam out_cloud:=/warpauv_1/cameras/mapping_cam/points color_correct_images:=True; bash'
tmux split-window -v 'cd ~/warp_ws && source devel/setup.bash && sleep 2 && roslaunch sunshine warpauv_bag.launch robot_name:=warpauv_1 bagfile:=/data/2021-usvi/2021-10-28/surveys/warpauv_1_xavier1_2021-10-28-16-08-21.bag bag_start:=1200 use_rviz:=true cell_size:=3600x1.2x1.2x15; bash'
#tmux split-window -v 'cd ~/warp_ws && source devel/setup.bash && sleep 2 && roslaunch sunshine warpauv_bag.launch robot_name:=warpauv_1 /data/2021-usvi/2021-10-25/surveys/warpauv_1_xavier1_2021-10-25-12-08-59.bag bag_start:=1500 use_rviz:=true cell_size:=3600x1.2x1.2x15; bash'
# /data/2021-usvi/2021-10-28/surveys/warpauv_1_xavier1_2021-10-28-16-08-21.bag bag_start:=1000
# /data/2021-usvi/2021-10-25/surveys/warpauv_1_xavier1_2021-10-25-12-08-59.bag bag_start:=600
tmux split-window -h 'cd ~/warp_ws && source devel/setup.bash && sleep 2 && roslaunch warpauv_config control_station.launch robot_name:=warpauv_1; bash'
tmux select-pane -t 0
tmux split-window -h 'cd ~/warp_ws && source devel/setup.bash && sleep 2 && roslaunch voxblox_ros rgbd_dataset.launch play_bag:=false voxel_size:=0.1 pointcloud_topic:=/warpauv_1/cameras/mapping_cam/points; bash'
sleep 4
rostopic pub -1 /tf_static tf2_msgs/TFMessage "transforms:
- header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: 'warpauv_1/map'
  child_frame_id: 'world'
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0"
rostopic pub -1 /tf_static tf2_msgs/TFMessage "transforms:
- header:
    seq: 0
    stamp:
      secs: 0
      nsecs: 0
    frame_id: 'warpauv_1/map'
  child_frame_id: 'map'
  transform:
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
    rotation:
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0"
tmux -2 attach-session -d
