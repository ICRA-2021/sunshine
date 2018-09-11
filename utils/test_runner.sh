#!/bin/bash

export DATA_DIR=$PWD/test_results
mkdir -p $DATA_DIR
TAG="${1-test}"


alphas='0.1 0.3 1 3'
betas='0.001 0.01 0.05 0.1'
gammas='0.001 0.01 0.05'
Ks='80'
texton='true'

for K in $Ks; do
	for alpha in $alphas; do
		for beta in $betas; do
			for gamma in $gammas; do
				export K=$K
				export ALPHA=$alpha
				export BETA=$beta
				export GAMMA=$gamma
				export TEXTON=$texton
				export OUTFILE="$TAG-k=$K-a=$alpha-b=$beta-g=$gamma"
				$PWD/run_test.sh
				sleep 5
			done
		done
	done
done
