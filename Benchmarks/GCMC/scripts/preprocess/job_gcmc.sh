#!/bin/sh
#
#PBS -N _GCMC_NAME
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=1

module load RASPA2/2.0.41-foss-2020b

cd $PBS_O_WORKDIR

date
# Save date
printf -v date '%(%y-%m-%d)T\n' -1
date=${date//-/}
date_start=${date::-1}


timeout 255600 bash run_gcmc.sh # Abort after 71h

if [ $? -eq 0 ]
then

    echo "Completed before timeout"
    rm -r CrashRestart # Restartfile not needed anymore, delete it to save memory

else
    
    for fn in Output/System_0/*.data; do
            if [[ -f $fn ]]; then
                    # Test 1: cycle should be larger than zero
                    cycle=$(grep "^Current cycle:" $fn | cut -d ' ' -f3)
                    # Test 2: simulation should not be finished
                    status=$(tail -3 $fn | head -1 | cut -d ' ' -f2)
            else
                    cycle=0
                    status=None
            fi
    done

    # Test 3: some days should have passed
    printf -v date '%(%y-%m-%d)T\n' -1
    date=${date//-/}
    date_end=${date::-1}

    if [[ ! -z $cycle ]] && [[ ! $cycle == 0 ]] && [[ ! $status == finished ]] && [[ ! $date_start == $date_end ]]; then
	    echo "Rerun"
            qsub -N _${date_end}_GCMC_NAME job_gcmc.sh
    else
	    echo CHECK what went wrong
	    echo cycle: $cylce
	    echo status: $status
	    echo date_start: $date_start 
	    echo date_end: $date_end
    fi

fi

date
