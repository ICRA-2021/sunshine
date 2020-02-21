#!/usr/bin/env bash
if [ ! -d adrost ]; then
  mkdir adrost
fi

start_dir=$PWD
pushd adrost || exit 1

workdir=stats_joint
if [ ! -d $workdir ]; then mkdir $workdir; fi
pushd $workdir || exit 1
echo "" > adrost.cmd
for run in 1 2 3 4 5 6 7 8 9 10 11 12; do
  for period in 1 20; do
#    for method in "clear" "clear-js" "hungarian" "hungarian-js"; do
      runid="stats-T$period-run$run"
      mkdir "$runid"
#      mkdir "$runid-matched"
      tm_args="save_topics_period:=$(echo "scale=3; $period / 2.0" | bc -l) save_topics_path:=$PWD/$runid"
#      match_args="_match_method:=$method _match_period:=$(echo "scale=3; $period / 2.0" | bc -l) _stats_path:=$PWD/$runid.csv _save_model_path:=$PWD/$runid-matched"
      cmd=("export tm_args=\"$tm_args\"")
      echo "${cmd[@]}" >>adrost.cmd
      cmd="${cmd[*]}"
      export tm_args="$tm_args" || exit 1
#      cmd=("export match_args=\"$match_args\"")
#      echo "${cmd[@]}" >>adrost.cmd
#      cmd="${cmd[*]}"
#      export match_args="$match_args" || exit 1
      cmd=("$start_dir/adrost_mission_new.sh")
      echo "${cmd[@]}" >>adrost.cmd
      cmd="${cmd[*]}"
      $start_dir/adrost_mission_new.sh && sleep 1800 || (
        tmux kill-server
        exit 1
      )
      tmux kill-server
      sleep 30
#    done
  done
done

popd || exit 1
