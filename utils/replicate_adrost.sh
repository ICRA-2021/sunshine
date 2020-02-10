#!/usr/bin/env bash
if [ ! -d adrost ]; then
  mkdir adrost
fi

start_dir=$PWD
pushd adrost || exit 1

for mission in 1 2; do
  workdir=stats_mission${mission}
  if [ ! -d $workdir ]; then mkdir $workdir; fi
  pushd $workdir || exit 1
  echo "" >adrost.cmd
  for run in 1 2 3; do
    for period in 1 20 40 60 80 100 250; do
      for method in "id" "hungarian"; do
        cmd="$start_dir/adrost_mission_$mission.sh _match_method:=$method _merge_period:=$(echo "scale=3; $period / 2.0" | bc -l) _stats_path:=$PWD/stats-T$period-$method-run$run.csv"
        echo $cmd >>adrost.cmd
        $cmd && sleep 1600 || (
          tmux kill-server
          exit 1
        )
        tmux kill-server
        sleep 30
      done
    done
  done
  popd || exit 1
done

popd || exit 1
