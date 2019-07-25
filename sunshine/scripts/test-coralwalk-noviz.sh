#!/usr/bin/env bash
if [ "$1" -eq "" ]; then
        echo "Error: you must supply a test image for test-coralwalk"
            exit 1
        fi
roslaunch sunshine test-coralwalk-noviz.launch test_image:=$1 &
rostopic echo -n 1 /image_walker/finished
rosrun sunshine save_topic_map _output_prefix:=bw_topics
rosrun sunshine save_topic_map _output_prefix:=color_topics _use_color:=true
rosservice sunshine save_topics_by_cell_csv "filename: '$PWD/topics.csv'"
killall -9 roscore
