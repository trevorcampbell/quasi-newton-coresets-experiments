#!/bin/bash

for dnm in "delays_medium"
do
    for alg in "UNIF" "FULL" "LAP" 'QNC-tau=0.0001' 'QNC-tau=0.001' 'QNC-tau=0.01' 'QNC-tau=0.1' 'QNC-tau=1'
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



