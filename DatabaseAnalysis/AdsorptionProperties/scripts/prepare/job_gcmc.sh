#!/bin/sh
#
#PBS -N _GCMC_NAME
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=1

module load RASPA2/2.0.41-foss-2020b

cd $PBS_O_WORKDIR

fn_log=../../../log/log_580000Pa.log

date
# Save date
printf -v date '%(%y-%m-%d)T\n' -1
date=${date//-/}
date_start=${date::-1}

for fn in Output/System_0/*.data; do
	if [[ -f $fn ]]; then
		cycle_start=$(grep "^Current cycle:" $fn | cut -d ' ' -f3 | tail -1)
	else
		cycle_start=0
	fi
done

echo START $(date) NAME TEMPK PRESSPa ${cycle_start} >> $fn_log

timeout 255600 bash run_gcmc.sh # Abort after 71h

if [ $? -eq 0 ]
then

    echo "Completed before timeout"
    rm -r CrashRestart # Restartfile not needed anymore, delete it to save memory
    rm -r Movies
    rm -r VTK
    echo END $(date) NAME TEMPK PRESSPa >> $fn_log
else
    
    for fn in Output/System_0/*.data; do
            if [[ -f $fn ]]; then
                    # Test 1: cycle should be larger than zero
                    cycle=$(grep "^Current cycle:" $fn | cut -d ' ' -f3 | tail -1)
                    # Test 2: simulation should not be finished
                    status=$(tail -3 $fn | head -1 | cut -d ' ' -f2)
            else
                    cycle=""
                    status=None
            fi
    done

    # Test 3: some days should have passed
    printf -v date '%(%y-%m-%d)T\n' -1
    date=${date//-/}
    date_end=${date::-1}

    if [[ ! -z $cycle ]] && [[ ! $status == finished ]] && [[ ! $date_start == $date_end ]]; then
	    if [[ $cycle_start == $cycle ]]; then
		    echo SLOW $(date) NAME TEMPK PRESSPa $cycle >> $fn_log
		    echo No progress was made, job terminated
    	    elif (( $cycle_start < 10000 )) && (( $cycle >= 10000 )); then
		    echo PAUSED $(date) NAME TEMPK PRESSPa $cycle >> $fn_log
	            echo Reached 10000 steps
		    echo Waiting for manual restart if wanted
	    else
		    echo INTERRUPTED $(date) NAME TEMPK PRESSPa $cycle >> $fn_log
		    echo "Rerun"
        	    qsub -N _${date_end}_GCMC_NAME job_gcmc.sh
	    fi
    else
	    echo CHECK what went wrong
	    echo cycle: $cycle
	    echo status: $status
	    echo date_start: $date_start 
	    echo date_end: $date_end
	    echo CRASH $(date) NAME TEMPK PRESSPa $cycle >> $fn_log
    fi

fi

date
