#!/bin/sh
#
#PBS -N _opt_struct
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=1
#PBS -m n

date

#################################    INPUT    ############################

#Argument: input=inputfile

src_path=/path/to/database

if [[ $only_yaff == True ]]; then
	only_yaff=only_yaff
else
	only_yaff=""
fi

if [ -z $input ]; then
	if [[ -z $name ]] || [[ -z $folder ]]; then
		echo Quitting: if no input file is given, both name and folder should be specified
	fi
fi

##############################    PREPARATION    #########################

# Set up input
ORIGDIR=$PBS_O_WORKDIR
WORKDIR=/local/$PBS_JOBID

if [ ! -d $WORKDIR ]; then mkdir -p $WORKDIR; fi
cd $WORKDIR

if [ -z $input ]; then
	input=input_single.txt
	echo $folder $name $only_yaff > $input
fi

# Copy all necessary scripts
scripts='.'
cp ${scripts}/run_dask.py $WORKDIR
cp ${scripts}/core.py $WORKDIR
cp ${scripts}/yaff_ff.py $WORKDIR
cp ${scripts}/yaff_system.py $WORKDIR
cp ${scripts}/yaff_lammpsio.py $WORKDIR
cp ${scripts}/yaff_liblammps.py $WORKDIR
cp ${scripts}/utils_log.py $WORKDIR
cp ${scripts}/utils_application.py $WORKDIR

# Copy inputfile to WORKDIR (make sure that the file is copied, not the inputfiles folder with the file)
inputfile=$(basename $input)
cp $ORIGDIR/${input} $WORKDIR/${inputfile}

count=0
while read -r line; do
	# For every structure, make a directory and copy all datafiles to it
	set ${line}
	folder="$1"
	struct="$2"
	path=${src_path}/${folder}/${struct}
	if [ ! "${1:0:1}" == \# ]; then
		if [ ! -d $path ]; then
	                echo Path ${path} not found, skipping this structure
		else
			mkdir $struct
			cp ${path}/pars_uff.txt $struct
			cp ${path}/pars_cluster.txt $struct
			cp ${path}/*.chk $struct
			cp ${path}/*.log $struct
			count=$(($count+1))
		fi
	fi
done < $inputfile

################################   OPTIMIZE   ############################

echo Starting optimization of ${count} structures in $input
if [[ $only_yaff == only_yaff ]]; then
        echo LAMMPS force field is turned off, just use normal yaff force field
fi

# Copy back results every hour
( while true; do
        sleep 3600
        while read -r line; do
		set ${line}
	        folder="$1"
        	struct="$2"
        	path=${src_path}/${folder}/${struct}
        	if [ ! "${1:0:1}" == \# ]; then
                	if [ -d $path ]; then
	                        cp ${WORKDIR}/${struct}/*.chk $path
        	                cp ${WORKDIR}/${struct}/*.log $path
                	fi
	        fi
	done < $inputfile
 done ) &

# Load modules
module load LAMMPS/3Mar2020-foss-2019b-Python-3.7.4-kokkos # LAMMPS, also loads yaff
module load dask/2.8.0-foss-2019b-Python-3.7.4 # dask for parallelization

# Run dask
python run_dask.py $inputfile $only_yaff

############################    POSTPROCESS    ############################

# Copy back results
while read -r line; do
	set ${line}
        folder="$1"
        struct="$2"
        path=${src_path}/${folder}/${struct}
        if [ ! "${1:0:1}" == \# ]; then
                if [ -d $path ]; then
                        cp ${WORKDIR}/${struct}/*.chk $path
                        cp ${WORKDIR}/${struct}/*.log $path
                fi
        fi
done < $inputfile

# Finalize
rm -rf $WORKDIR

date

echo "####################################################################################"
echo "####################################################################################"
echo
echo


