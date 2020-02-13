#!/usr/bin/env bash
if [ ! -d adrost ]; then
  mkdir adrost
fi

start_dir=$PWD
pushd adrost || exit 1

workdir=stats_joint
if [ ! -d $workdir ]; then mkdir $workdir; fi
pushd $workdir || exit 1
echo "" >adrost.cmd
for run in 1 2 3; do
  for period in 1 20 40 60 80 100 250; do
    for method in "clear" "clear-js" "hungarian" "hungarian-js"; do
      runid="stats-T$period-$method-run$run"
      mkdir "$runid"
      cmd="$start_dir/adrost_mission_combined.sh _match_method:=$method _match_period:=$(echo "scale=3; $period / 2.0" | bc -l) _stats_path:=$PWD/$runid.csv _save_model_path:=$PWD/$runid/"
      echo $cmd >>adrost.cmd
      $cmd && sleep 1800 || (
        tmux kill-server
        exit 1
      )
      tmux kill-server
      sleep 30
    done
  done
done

popd || exit 1
