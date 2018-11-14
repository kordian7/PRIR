#!/bin/bash

compile() {
	make
}

run_test() {
	TIME=0
	TIMES=4
	for ((j=1; j<=$TIMES; j++))
	do
		TMP=$(./primes_gpu $1 $2 | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
		TIME=$(echo "$TIME + $TMP" | bc -l)
	done
	echo $( echo "$TIME/$TIMES" | bc -l )
}

compile
run_test $1 $2