#!/usr/bin/env bash
if [ "$1" -eq "" ]; then
        echo "Error: you must supply a test image for test-coralwalk"
            exit 1
        fi
roslaunch sunshine test-coralwalk-noviz.launch test_img:=$1 &
rostopic echo -n 1 /image_walker/finished
rosrun sunshine save_topic_map
rosservice sunshine save_topics_scsv
kill ros
