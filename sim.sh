#!/bin/bash

for d in '2' '3' '4' '5'
    do
    for e in '4' '8' '16' '32' '64'
	    do
        for t in '0.1' '0.01' '0.001' '0.0001'
            do
				echo "Degree: $d, Element: $e, Tolerance: $t"
    	        python3 linear_1d_gllv_direct.py $d $e $t
    	    done
        done
	done