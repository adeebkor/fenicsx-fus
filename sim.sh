#!/bin/bash

for d in '2' '3' '4' '5'
    do
    for e in '8' '16' '32' '64'
	    do
        for t in '0.001' '0.000001' '0.000000001' '0.000000000001'
            do
				echo "Degree: $d, Element: $e, Tolerance: $t"
    	        python3 linear_1d_gllv_scipy.py $d $e $t
    	    done
        done
	done