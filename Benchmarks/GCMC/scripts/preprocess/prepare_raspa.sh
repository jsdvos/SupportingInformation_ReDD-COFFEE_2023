for framework in COF-1 COF-5 COF-102 COF-103; do
	for host_guest in mm3 dreiding uff; do
		for guest_guest in mm3 dreiding uff trappe_ua trappe_eh; do
               		for press in 0 500000 1000000 1500000 2000000 3000000 4000000 5000000 6000000 7000000; do
				fn_host=../../data/input_files/${framework}.chk
				fn_guest=../../data/input_files/ch4_all.chk
				fn_host_host=../../data/input_files/pars_${framework}_${host_guest}.txt
				fn_host_guest=../../data/input_files/pars_ch4_${host_guest}.txt
				fn_guest_guest=../../data/input_files/pars_ch4_${guest_guest}.txt
				temp=298
				python prepare_raspa.py $fn_host $fn_guest $fn_host_host $fn_host_guest $fn_guest_guest $temp $press
	
				workdir=../../data/output_files/${framework}/${host_guest}_${guest_guest}_${temp}K_${press}Pa
				if [ -d $workdir ]; then
					echo $workdir already exists: skipping
				elif [ -d ${temp}K_${press}Pa ]; then
					mv ${temp}K_${press}Pa $workdir
					cp ${fn_guest/.chk/.def} $workdir
					cp job_gcmc.sh $workdir
					cp run_gcmc.sh $workdir
					sed -i s/NAME/${framework}_${workdir}/g ${workdir}/job_gcmc.sh
					echo Created $workdir
				else
					echo Something went wrong: check $workdir
				fi
			done
		done
	done
done
