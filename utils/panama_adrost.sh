#!/bin/sh

tm_args="beta:=0.15 alpha:=0.073 K:=10"
tmux new-session -d -s foo "roslaunch sunshine panama.launch $tm_args rosbag:=/home/stewart/datasets/panama-bags/d20150404_3.bag image_topic:=camera/image_raw basename:=rost_1 || bash"
tmux rename-window 'panama_adrost'
tmux select-window -t foo:0
#tmux split-window -v -t 0 'rosbag play /home/stewart/datasets/panama-bags/d20150418_18.bag /camera/image_raw:=/bag18/camera/image_raw --rate 4 || bash'
#tmux split-window -v -t 1 'rosbag play /home/stewart/datasets/panama-bags/d20150404_3.bag /camera/image_raw:=/bag3/camera/image_raw --rate 4 || bash'
#tmux split-window -v -t 3 'rosrun sunshine model_translator _target_nodes:=rost_1,rost_2 _match_period:=5 _save_model_path:="/home/stewart/datasets/panama-bags/panama-topics" || bash'
tmux split-window -v -t 0 "rosrun sunshine model_translator _target_nodes:=rost_1,rost_2 $* || bash"
tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args rosbag:=/home/stewart/datasets/panama-bags/d20150418_18.bag image_topic:=camera/image_raw basename:=rost_2 || bash"
#tmux split-window -h -t 0 "roslaunch sunshine panama.launch $tm_args rosbag:=/home/stewart/datasets/panama-bags/d20150404_3.bag image_topic:=camera/image_raw basename:=rost_2 || bash"
tmux select-window -t foo:1
tmux -2 attach-session -t foo
