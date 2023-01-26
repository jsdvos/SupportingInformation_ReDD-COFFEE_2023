#!/bin/sh
#
#PBS -N _benchmark_zeopp
#PBS -l walltime=5:00:00
#PBS -l nodes=1:ppn=1
#PBS -m n

date

cd /path/to/data

rad=1.84 # angstrom

for struct in *; do
	cd $struct
	fn_cif=${struct}_optimized.cif
	for N in 250 500 750 1000 1250 1500 1750 2000 2250 2500 2750 3000 3250 3500; do
		out_res=${struct}_optimized_${N}.res
		out_sa=${struct}_optimized_${N}.sa
		out_vol=${struct}_optimized_${N}.vol
		network -ha -res $out_res -sa $rad $rad $N $out_sa -vol $rad $rad $N $out_vol $fn_cif
	done
	cd ../
done

date

