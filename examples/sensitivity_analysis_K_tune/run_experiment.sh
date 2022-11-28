#!/bin/bash

for dnm in "delays_medium"
do
    for alg in "UNIF" "FULL" "LAP" "QNC-K_tune=0" "QNC-K_tune=1" "QNC-K_tune=2" "QNC-K_tune=5" "QNC-K_tune=10"
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



