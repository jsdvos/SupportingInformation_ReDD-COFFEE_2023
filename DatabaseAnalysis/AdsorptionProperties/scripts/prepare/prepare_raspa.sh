#!/bin/sh
#
#PBS -N _prepare_raspa
#PBS -l walltime=3:00:00
#PBS -l nodes=1:ppn=1
#PBS -m n

date

module load yaff/1.6.0-intel-2020a-Python-3.8.2

# Set up input
ORIGDIR=$PBS_O_WORKDIR
WORKDIR=/local/$PBS_JOBID

if [ ! -d $WORKDIR ]; then mkdir -p $WORKDIR; fi
cd $WORKDIR

cp ${ORIGDIR}/ch4.chk $WORKDIR
cp ${ORIGDIR}/ch4.def $WORKDIR
cp ${ORIGDIR}/pars_ch4_dreiding.txt $WORKDIR
cp ${ORIGDIR}/pars_ch4_trappe_ua.txt $WORKDIR
cp ${ORIGDIR}/prepare_raspa.py $WORKDIR

for press in 580000 6500000; do
	for struct_path in ../../data/input_files/*; do
		struct=$(basename $struct_path)
		fn_host=${struct_path}/${struct}_optimized.chk
		fn_guest=ch4.chk
		fn_host_host=${struct_path}/pars_noncov_dreiding.txt
		fn_host_guest=pars_ch4_dreiding.txt
		fn_guest_guest=pars_ch4_trappe_ua.txt
		temp=298
	
		python prepare_raspa.py $fn_host $fn_guest $fn_host_host $fn_host_guest $fn_guest_guest $temp $press

		destdir=../../data/${temp}K_${press}Pa/${struct}
		if [ -d $destdir ]; then
			echo $destdir already exists: skipping
		elif [ -d ${temp}K_${press}Pa ]; then
			mv ${temp}K_${press}Pa $destdir
			cp ${fn_guest/.chk/.def} $destdir
			cp ${ORIGDIR}/job_gcmc.sh $destdir
			cp ${ORIGDIR}/run_gcmc.sh $destdir
			sed -i s/NAME/${struct}/g ${destdir}/job_gcmc.sh
			sed -i s/TEMP/${temp}/g ${destdir}/job_gcmc.sh
			sed -i s/PRESS/${press}/g ${destdir}/job_gcmc.sh
			echo Created $destdir
			if (( ${#struct} > 200 )); then
				# Long names should be made shorter, or errors will happen
				top=${struct%%_*}
		        	sbus=${struct#*_}
		        	sbu_first=${sbus%%_*}
        			sbu_last=${sbus##*_}
		        	short_name=${top}_${sbu_first}_${sbu_last}

				olddir=$destdir
				destdir=${ORIGDIR}/../simulations/${temp}K_${press}Pa/${short_name}
				mv $olddir $destdir
				mv ${destdir}/${struct}_optimized.cif ${destdir}/${short_name}_optimized.cif
				sed -i s/${struct}_optimized/${short_name}_optimized/g ${destdir}/simulation.input
				sed -i s/GCMC_${struct}/GCMC_${short_name}/g ${destdir}/job_gcmc.sh
			fi
		else
			echo Something went wrong: check $workdir
		fi
	done
done 

date
