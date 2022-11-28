#!/bin/bash

for dnm in "delays_medium"
do
    for alg in "LAP" "UNIF" "IHT" "QNC" "GIGA" "FULL"
#    for alg in "IHT-LAP" "GIGA-LAP"
    do
        for ID in {1..10}
        do
        	for M in 100 500 1000 5000 10000
        	do
				python3 main.py --model lr --dataset $dnm --coreset_size $M --alg $alg --trial $ID run
			done
		done
    done
done