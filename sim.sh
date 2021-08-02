#!/bin/bash

for d in '2' '3' '4' '5'
    do
    for e in '8' '16' '32' '64'
	    do
<<<<<<< HEAD
        for t in '0.00001' '0.0000001' '0.000000001' '0.00000000001'
=======
        for t in '0.001' '0.000001' '0.000000001' '0.000000000001'
>>>>>>> b818822ec8c7989ec4cde5ddca16afb1418f0990
            do
				echo "Degree: $d, Element: $e, Tolerance: $t"
    	        python3 linear_1d_gllv_scipy.py $d $e $t
    	    done
        done
	done