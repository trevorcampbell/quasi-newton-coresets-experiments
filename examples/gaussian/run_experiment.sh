#!/bin/bash

# Run all algorithms
for alg in "LAP" "GIGA" "IHT" "UNIF" "QNC" "FULL"
#for alg in "IHT-LAP" "GIGA-LAP"
do
    for ID in {1..10}
    do
    	for M in 100 500 1000 5000 10000
        do
       		python3 main.py --samples_inference 1000 --alg $alg --trial $ID --coreset_size $M --proj_dim 500 run
       	done
    done
done