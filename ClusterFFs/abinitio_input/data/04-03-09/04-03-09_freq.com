%nproc=15
%mem=70GB
%chk=Pyrene_PrimaryAmine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Pyrene_PrimaryAmine_Ketoenamine.log

0 1
 C   0.033013   3.529048  -0.057633
 C  -1.182824   2.835762  -0.011674
 C  -1.217590   1.439563   0.013742
 C   0.006160   0.709849  -0.011490
 C   1.241650   1.417689  -0.054402
 C   1.237392   2.816969  -0.069100
 H  -2.113450   3.393930  -0.003876
 H   2.187884   3.335946  -0.071618
 C  -0.005990  -0.709847   0.011496
 C  -1.241480  -1.417688   0.054409
 C  -1.237221  -2.816967   0.069108
 C  -0.032842  -3.529046   0.057642
 C   1.182994  -2.835760   0.011681
 C   1.217760  -1.439561  -0.013736
 H  -2.187713  -3.335945   0.071626
 H   2.113620  -3.393928   0.003883
 C   2.461944   0.657227  -0.071453
 C   2.451071  -0.701896  -0.054668
 H   3.404507   1.193870  -0.101234
 H   3.383785  -1.255683  -0.070884
 C  -2.450900   0.701897   0.054674
 C  -2.461774  -0.657225   0.071459
 H  -3.383615   1.255685   0.070890
 H  -3.404337  -1.193868   0.101241
 H  -2.557329   9.320559  -0.093429
 C  -1.676537   9.946935  -0.026881
 C  -1.783013   11.332690   0.004206
 H  -2.759719   11.800595  -0.048308
 C  -0.635591   12.120848   0.106132
 H  -0.717893   13.201674   0.134690
 C   0.617250   11.513618   0.176440
 H   1.511057   12.120415   0.268008
 C   0.724668   10.125573   0.133355
 H   1.705422   9.672384   0.206227
 C  -0.421159   9.325488   0.028267
 C  -0.375477   7.825765  -0.004128
 O  -1.419514   7.186902   0.227075
 C   0.867267   7.155701  -0.322639
 C   0.983697   5.790589  -0.347848
 H   1.741927   7.729313  -0.589242
 N  -0.020646   4.929993  -0.079582
 H   1.936803   5.348054  -0.613470
 H  -0.911081   5.397345   0.110001
 H   2.557437  -9.320646   0.093481
 C   1.676628  -9.946996   0.026900
 C   1.783065  -11.332753  -0.004221
 H   2.759757  -11.800687   0.048299
 C   0.635623  -12.120875  -0.106189
 H   0.717895  -13.201703  -0.134773
 C  -0.617199  -11.513607  -0.176503
 H  -1.511022  -12.120377  -0.268103
 C  -0.724578  -10.125561  -0.133385
 H  -1.705319  -9.672342  -0.206262
 C   0.421269  -9.325511  -0.028255
 C   0.375630  -7.825788   0.004182
 O   1.419692  -7.186953  -0.226987
 C  -0.867109  -7.155698   0.322665
 C  -0.983528  -5.790584   0.347863
 H  -1.741781  -7.729297   0.589261
 N   0.020816  -4.929991   0.079592
 H  -1.936633  -5.348042   0.613478
 H   0.911246  -5.397346  -0.110002





