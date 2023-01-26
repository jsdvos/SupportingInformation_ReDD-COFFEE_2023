#!/bin/sh
#
#PBS -N _dreiding
#PBS -l walltime=20:00:00
#PBS -l nodes=1:ppn=1
#PBS -m n

date

cd $PBS_O_WORKDIR

module load yaff/1.6.0-intel-2020a-Python-3.8.2

python dreiding.py $database

date

