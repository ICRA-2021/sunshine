#!/bin/sh

echo "$@"
tmux new-session -d -s foo "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150405_5.bag image_topic:=camera/image_raw basename:=rost_5 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150409_7.bag image_topic:=camera/image_raw basename:=rost_6 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150409_8.bag image_topic:=camera/image_raw basename:=rost_7 || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args num_threads:=2 rosbag:=/home/stewart/datasets/panama-bags/d20150410_9.bag image_topic:=camera/image_raw basename:=rost_8 || bash"
#tmux split-window -v -t 0 "rosrun sunshine model_translator $match_args _target_nodes:=rost_1,rost_2,rost_3,rost_4 || bash"
tmux rename-window 'panama_adrost_new'

