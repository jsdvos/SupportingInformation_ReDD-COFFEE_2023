%nproc=15
%mem=70GB
%chk=DBA12_CarboxylicAnhydride_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from DBA12_CarboxylicAnhydride_Imide.log

0 1
 C   2.746967   0.715100   0.015864
 C   3.964989   1.421431   0.019923
 C   5.139359   0.696007   0.018853
 C   5.139438  -0.695423   0.011455
 C   3.965149  -1.420980   0.010427
 C   2.747048  -0.714786   0.014514
 H   3.974265   2.503886   0.024288
 H   3.974548  -2.503434   0.006072
 C   1.520582  -1.424438   0.007917
 C   0.473440  -2.029070   0.024370
 C  -0.754259  -2.736560   0.017201
 C  -0.751578  -4.144557   0.022142
 C  -1.966998  -4.798858   0.020036
 C  -3.172056  -4.103220   0.010682
 C  -3.213245  -2.723476   0.008904
 C  -1.992599  -2.021682   0.014144
 H   0.181201  -4.693835   0.027884
 H  -4.155346  -2.190346   0.003117
 C  -1.993870  -0.604710   0.007028
 C   1.520422   1.424615   0.022543
 C   0.473207   2.029123   0.006127
 C  -0.754579   2.736462   0.013270
 C  -0.752068   4.144459   0.008286
 C  -1.967568   4.798614   0.010337
 C  -3.172542   4.102830   0.019663
 C  -3.213563   2.723081   0.021475
 C  -1.992833   2.021434   0.016303
 H   0.180644   4.693850   0.002550
 H  -4.155601   2.189838   0.027221
 C  -1.993935   0.604461   0.023374
 C   6.550906   1.171656   0.019032
 C   6.551038  -1.170912   0.011184
 N   7.347292   0.000417   0.015104
 O   6.956681  -2.306304   0.007559
 O   6.956421   2.307093   0.022678
 C   8.775002   0.000502   0.015079
 C   9.467332  -0.857800  -0.840726
 H   8.921450  -1.525922  -1.492593
 C   10.859290  -0.857752  -0.831724
 H   11.396171  -1.528822  -1.492047
 C   11.558757   0.000680   0.015022
 H   12.642606   0.000750   0.015000
 C   10.859214   0.859023   0.861795
 H   11.396036   1.530164   1.522095
 C   9.467257   0.858893   0.870853
 H   8.921315   1.526948   1.522740
 C  -2.260823  -6.259119   0.020812
 C  -4.289623  -5.087992   0.009413
 N  -3.673310  -6.363205   0.015023
 O  -5.475731  -4.871678   0.003919
 O  -1.480266  -7.178023   0.026202
 C  -4.387042  -7.599717   0.014805
 C  -5.475051  -7.770861  -0.842709
 H  -5.779646  -6.964538  -1.495682
 C  -6.170882  -8.976421  -0.833987
 H  -7.019353  -9.106441  -1.495647
 C  -5.778515  -10.010685   0.014239
 H  -6.320290  -10.949414   0.014020
 C  -4.686862  -9.833302   0.862746
 H  -4.375116  -10.633247   1.524187
 C  -3.991109  -8.627700   0.872049
 H  -3.140677  -8.488307   1.525230
 C  -4.290228   5.087466   0.020813
 C  -2.261570   6.258839   0.009461
 N  -3.674069   6.362754   0.015211
 O  -1.481122   7.177836   0.004109
 O  -5.476310   4.871010   0.026294
 C  -4.387957   7.599177   0.015362
 C  -3.992123   8.627183  -0.841902
 H  -3.141646   8.487880  -1.495043
 C  -4.688038   9.832691  -0.832672
 H  -4.376370   10.632655  -1.494128
 C  -5.779752   10.009959   0.015780
 H  -6.321652   10.948617   0.015942
 C  -6.172019   8.975673   0.864025
 H  -7.020537   9.105602   1.525642
 C  -5.476027   7.770206   0.872820
 H  -5.780545   6.963865   1.525806






