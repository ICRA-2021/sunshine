#!/bin/bash

export DATA_DIR="$PWD/${1:-test_results}"
mkdir -p $DATA_DIR
TAG="${1-test}"


alphas='0.1 0.5 1 5'
betas='0.01 0.05 0.1 0.5'
gammas='0.0001 0.005 0.01 0.05'
Ks='80 160'
textons='false true'

for texton in $textons; do
	for K in $Ks; do
		for alpha in $alphas; do
			for beta in $betas; do
				for gamma in $gammas; do
					export K=$K
					export ALPHA=$alpha
					export BETA=$beta
					export GAMMA=$gamma
					export TEXTON=$texton
					export OUTFILE="$TAG-texton=$texton-k=$K-a=$alpha-b=$beta-g=$gamma"
					$PWD/run_test.sh
					sleep 5
				done
			done
		done
	done
done
