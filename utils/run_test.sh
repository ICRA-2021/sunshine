#!/bin/bash
cd $HOME/catkin_ws
source devel/setup.bash
cd $DATA_DIR

echo "Starting task $OUTFILE"
roslaunch sunshine test-imwalk-norviz.launch test_image:=$HOME/data/HAW_2016_48_RAW-subsampled-equalized.png num_threads:=6 speed:=600 size:=1800x1800 walk_scale:=0.1 fps:=1 rate:=0.5 cell_time:=3600 cell_space:=0.8 K:=${K:-8} texton:=${TEXTON:-true} alpha:=$ALPHA beta:=$BETA gamma:=$GAMMA overlap:=450 &
PROC_ID=$!

function killafter() {
	sleep 200
	rosrun sunshine save_topic_map rost $OUTFILE
	kill $PROC_ID
	pkill -f $HOME/catkin_ws
}
killafter &

wait

SCORE=$(python $HOME/bin/mutual_info $HOME/data/HAW_2016_48_Annotated-subsampled2.png $OUTFILE-topics.png 10000)
SIZE=$(du --block-size=1 $OUTFILE-topics.png | awk '{print $1;}')
echo "$OUTFILE,$SIZE,$SCORE" >> $LOGFILE
echo "Done!"
