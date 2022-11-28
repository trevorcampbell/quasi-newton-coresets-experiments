#!/bin/bash

for dnm in "delays_medium"
do
    for alg in "UNIF" "FULL" "LAP" "QNC-S=10" "QNC-S=50" "QNC-S=100" "QNC-S=500" "QNC-S=1000"
#    for alg in "IHT-LAP" "GIGA-LAP"
    do
        for ID in {1..10}
        do
        	for M in 50 100 500 1000 5000
        	do
				python3 main.py --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done



