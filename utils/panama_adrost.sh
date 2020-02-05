#!/bin/sh

tmux new-session -d -s foo 'roslaunch sunshine panama.launch image_topic:=/bag18/camera/image_raw basename:=rost_1'
tmux rename-window 'panama_adrost'
tmux select-window -t foo:0
tmux split-window -h 'roslaunch sunshine panama.launch image_topic:=/bag19/camera/image_raw basename:=rost_2'
tmux split-window -v -t 0 'rosbag play --clock /home/stewart/datasets/panama-bags/d20150418_18.bag /camera/image_raw:=/bag18/camera/image_raw --rate 3'
tmux split-window -v -t 1 'ROS_NAMESPACE="bag19" rosbag play --clock /home/stewart/datasets/panama-bags/d20150419_19.bag /camera/image_raw:=/bag19/camera/image_raw --rate 3'
tmux split-window -v -t 3 'rosrun sunshine model_translator _target_nodes:=rost_1,rost_2 _match_period:=3'
tmux -2 attach-session -t foo
