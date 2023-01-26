%nproc=15
%mem=70GB
%chk=TetraPhenylMethane_BoronicAcid_Boroxine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TetraPhenylMethane_BoronicAcid_Boroxine.log

0 1
 C  -0.000000  -0.000001   0.000001
 C   2.911351   2.275703   2.356578
 C  -2.911390  -2.356711   2.275508
 C   2.911352  -2.275707  -2.356574
 C  -2.911381   2.356714  -2.275511
 C   2.107317   1.305279   2.968262
 C  -2.107239  -2.968351   1.305152
 C   2.107318  -1.305283  -2.968259
 C  -2.107232   2.968352  -1.305153
 C   1.153545   0.593192   2.243869
 C  -1.153449  -2.243904   0.593144
 C   1.153545  -0.593195  -2.243867
 C  -1.153445   2.243903  -0.593143
 C   0.967810   0.831201   0.879339
 C  -0.967806  -0.879366   0.831175
 C   0.967810  -0.831204  -0.879337
 C  -0.967803   0.879365  -0.831175
 C   1.790626   1.784021   0.256854
 C  -1.790738  -0.256924   1.783925
 C   1.790625  -1.784024  -0.256850
 C  -1.790734   0.256924  -1.783927
 C   2.734142   2.497601   0.979516
 C  -2.734275  -0.979639   2.497422
 C   2.734142  -2.497605  -0.979512
 C  -2.734268   0.979642  -2.497426
 H   2.232728   1.102733   4.026264
 H  -2.232574  -4.026360   1.102594
 H   2.232730  -1.102737  -4.026260
 H  -2.232565   4.026361  -1.102595
 H   0.559766  -0.154593   2.751708
 H  -0.559572  -2.751703  -0.154592
 H   0.559767   0.154590  -2.751705
 H  -0.559568   2.751701   0.154594
 H   1.692037   1.956468  -0.808261
 H  -1.692238   0.808199   1.956366
 H   1.692036  -1.956471   0.808264
 H  -1.692235  -0.808199  -1.956368
 H   3.352716   3.232540   0.476455
 H  -3.352948  -0.476613   3.232303
 H   3.352716  -3.232543  -0.476451
 H  -3.352940   0.476617  -3.232308
 B   3.965692   3.063180   3.163415
 O   4.141742   2.832797   4.520455
 B   5.078795   3.530339   5.242258
 O   4.761768   4.020006   2.549908
 B   5.701903   4.722966   3.262526
 O   5.858173   4.475681   4.609866
 H   5.217917   3.322464   6.401192
 H   6.375172   5.537133   2.724450
 B  -3.965763  -3.163604   3.062883
 O  -4.141647  -4.520675   2.832556
 B  -5.078676  -5.242544   3.530062
 O  -4.761896  -2.550160   4.019702
 B  -5.702010  -3.262844   4.722623
 O  -5.858103  -4.610217   4.475406
 H  -5.217572  -6.401529   3.322320
 H  -6.375249  -2.724846   5.536867
 B   3.965693  -3.063184  -3.163411
 O   4.141744  -2.832801  -4.520450
 B   5.078797  -3.530343  -5.242253
 O   4.761769  -4.020009  -2.549903
 B   5.701905  -4.722970  -3.262521
 O   5.858175  -4.475685  -4.609860
 H   5.217919  -3.322468  -6.401187
 H   6.375173  -5.537137  -2.724444
 B  -3.965751   3.163609  -3.062889
 O  -4.141634   4.520680  -2.832561
 B  -5.078660   5.242551  -3.530069
 O  -4.761883   2.550166  -4.019709
 B  -5.701995   3.262853  -4.722632
 O  -5.858086   4.610225  -4.475415
 H  -5.217555   6.401536  -3.322327
 H  -6.375233   2.724856  -5.536877






