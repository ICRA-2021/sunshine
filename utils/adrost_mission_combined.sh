#!/bin/sh

echo "$@"
tmux new-session -d -s foo "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150419_19.bag image_topic:=camera/image_raw basename:=rost_1 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150403_2.bag image_topic:=camera/image_raw basename:=rost_2 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150404_3.bag image_topic:=camera/image_raw basename:=rost_3 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150418_18.bag image_topic:=camera/image_raw basename:=rost_4 || bash"
tmux split-window -v -t 0 "rosrun sunshine model_translator $match_args _target_nodes:=rost_1,rost_2,rost_3,rost_4 || bash"
tmux rename-window 'panama_adrost'

