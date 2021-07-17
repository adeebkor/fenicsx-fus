#!/bin/bash

for d in '1' '2' '3' '4'
    do
    for e in '8' '16' '32' '64'
	    do
        for c in '0.9' '0.45' '0.225' '0.1125'
            do
				echo "Degree: $d, Element: $e, CFL: $c"
    	        python3 linear_1d_direct.py $d $e $c
    	    done
        done
	done