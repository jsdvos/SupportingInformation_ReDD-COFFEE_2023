%nproc=15
%mem=70GB
%chk=PMDA_Hydrazide_Hydrazone_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from PMDA_Hydrazide_Hydrazone.log

0 1
 C   2.692284  -0.835487  -0.331419
 C   1.236114  -0.528508  -0.222599
 C   0.150362  -1.284091  -0.648782
 H   0.262610  -2.243564  -1.137638
 C  -1.091191  -0.713159  -0.396198
 C  -2.448915  -1.243819  -0.715383
 C  -1.236306   0.523545   0.233293
 C  -2.692470   0.830536   0.342108
 C  -0.150552   1.279128   0.659478
 H  -0.262804   2.238599   1.148338
 C   1.091000   0.708196   0.406896
 C   2.448725   1.238849   0.726095
 N  -3.363478  -0.268979  -0.245568
 N   3.363286   0.264028   0.256255
 O  -3.210965   1.804851   0.825130
 O  -2.731176  -2.280571  -1.258611
 O   2.730981   2.275609   1.269311
 O   3.210781  -1.809800  -0.814446
 C  -4.781258  -0.381382  -0.345969
 C   4.781070   0.376446   0.356631
 C  -5.537338   0.727212  -0.728480
 C  -6.921363   0.619884  -0.805910
 H  -5.048915   1.663776  -0.958043
 C  -7.560399  -0.591474  -0.519530
 H  -7.491111   1.481660  -1.134084
 C  -6.785864  -1.702078  -0.170291
 C  -5.405655  -1.600185  -0.067766
 H  -7.284332  -2.644124   0.020785
 H  -4.813853  -2.460948   0.210386
 C   5.405457   1.595254   0.078437
 C   6.785668   1.697148   0.180927
 H   4.813660   2.456028  -0.199691
 C   7.560232   0.586546   0.530151
 H   7.284137   2.639191  -0.010161
 C   6.921195  -0.624801   0.816546
 C   5.537170  -0.732141   0.739127
 H   7.490917  -1.486587   1.144750
 H   5.048769  -1.668720   0.968675
 C  -9.047835  -0.787076  -0.610424
 O  -9.546953  -1.854885  -0.895507
 N  -9.785610   0.352235  -0.323912
 H  -9.315263   1.158689   0.077811
 N  -11.139467   0.344832  -0.406547
 C  -11.759701   1.394195  -0.016612
 C  -13.218345   1.498985  -0.073820
 H  -11.219038   2.265439   0.378754
 C  -13.836749   2.677569   0.366170
 C  -15.222924   2.808850   0.325647
 H  -13.228878   3.494699   0.742069
 C  -16.006646   1.762444  -0.155401
 H  -15.688831   3.725448   0.669045
 C  -15.397428   0.583534  -0.596108
 H  -17.085715   1.861530  -0.187964
 C  -14.016421   0.448409  -0.557803
 H  -16.005993  -0.231962  -0.970384
 H  -13.536990  -0.461810  -0.896548
 C   9.047650   0.782191   0.620973
 O   9.546764   1.850080   0.905800
 N   9.785443  -0.357152   0.334675
 H   9.315114  -1.163723  -0.066837
 N   11.139300  -0.349721   0.417306
 C   11.759525  -1.399172   0.027603
 C   13.218169  -1.503951   0.084824
 H   11.218857  -2.270502  -0.367573
 C   13.836568  -2.682649  -0.354869
 C   15.222742  -2.813925  -0.314314
 H   13.228693  -3.499870  -0.730564
 C   16.006469  -1.767401   0.166469
 H   15.688644  -3.730612  -0.657480
 C   15.397257  -0.588376   0.606875
 H   17.085538  -1.866484   0.199058
 C   14.016250  -0.453255   0.568537
 H   16.005825   0.227214   0.980942
 H   13.536822   0.457052   0.907050





