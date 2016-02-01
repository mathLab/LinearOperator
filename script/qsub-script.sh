#!/bin/bash

for i in `seq 1 10` 
do 
    echo "Generating qsub_"${i} 
    cat qsub-template |\
    sed "s/OOOOO/$i/" > qsub_$i
    qsub qsub_$i
done
