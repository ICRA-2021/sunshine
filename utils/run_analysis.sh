#!/bin/bash
LOG_FILE="${1-results.txt}"
echo "Name,Size,Score" > $LOG_FILE
for img in *-topics.png; do 
	SCORE=$(python ~/bin/mutual_info ~/data/HAW_2016_48_Annotated-subsampled2.png $img 10000)
	SIZE=$(stat --format="%s" $img)
	echo "$img,$SIZE,$SCORE" >> $LOG_FILE
done
