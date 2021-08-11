#!/bin/bash

for d in '2' '3' 
    do
    for e in '8' '16' '32' '64'
	    do
        for t in '1e-3' '1e-6' '1e-9' '1e-12'
            do
				echo "Degree: $d, Element: $e, Tolerance: $t"
    	        python3 linear_1d_scipy.py $d $e $t
    	    done
        done
	done