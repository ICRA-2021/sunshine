#!/bin/bash

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=4
open_sem $N
for d in $1*/ ; do
	if [[ $d != *"matched"* ]]; then
		if [[ $d != *"-T1-"* ]]; then
			echo "$d"
			rosrun sunshine test.topic_match "$d" "16180" "clear-l1,clear-minpmatch,clear-pmatch,hungarian-l1,hungarian-pmatch,clear-distinct-l1,clear-distinct-minpmatch,clear-distinct-pmatch" > ${d::-1}-processed.csv
		fi
	fi
done
#python compare_matching.py $1/*-processed.csv
