#!/bin/bash

compile() {
	make
}

run_test() {
	TIME=0
	TIMES=4
	for ((j=1; j<=$TIMES; j++))
	do
		TMP=$(./RSA_CUDA $1 | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
		TIME=$(echo "$TIME + $TMP" | bc -l)
	done
	echo $( echo "$TIME/$TIMES" | bc -l )
}

run() {
	cat /dev/null > dane/times.txt
	cat /dev/null > dane/acc.txt
	for ((i=1; i<=$1; i*=2))
	do
		CZAS=$(run_test $i)
		if [ $i = 1 ]
		then
			FIRST=$CZAS
		fi
		ACC=$(echo "scale=4; $FIRST/$CZAS" | bc -l)
		echo "$i $CZAS" >> times.txt
		echo "$i $i $ACC" >> acc.txt
	done
}

create_time_plot() {
	gnuplot <<- EOF
    set terminal postscript eps
    set output 'dane/wykresCzas.eps'
    load 'gnuplot/czas.txt'
	EOF
}

create_acc_plot() {
gnuplot <<- EOF
    set terminal postscript eps
    set output 'dane/wykresPrzyspieszenie.eps'
    load 'gnuplot/przyspieszenie.txt'
	EOF
}

create_plots() {
	create_time_plot
	create_acc_plot
	
}

create_pdf() {
	pdflatex dok.tex
}

clean() {
	rm -rf zad1.log zad1.aux
}

#compile
#run $1
create_plots
#create_pdf
clean
