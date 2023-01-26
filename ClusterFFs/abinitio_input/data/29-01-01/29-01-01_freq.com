%nproc=15
%mem=70GB
%chk=TetraPhenylSilane_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TetraPhenylSilane_BoronicAcid_BoronateEster-2.log

0 1
Si  -0.000721   0.005109  -0.004524
 C  -1.729826  -1.777054   4.010974
 C   2.953140   3.552985   0.983983
 C  -3.563246   1.599225  -2.661303
 C   2.340448  -3.363691  -2.341248
 C  -0.558274  -1.003835   4.018382
 C   1.638427   3.813106   0.566456
 C  -2.492481   0.904007  -3.244863
 C   1.418682  -3.699290  -1.337213
 C  -0.054772  -0.459745   2.840366
 C   0.769046   2.771578   0.255624
 C  -1.448202   0.416226  -2.464558
 C   0.738728  -2.709844  -0.632928
 C  -0.702638  -0.669515   1.612948
 C   1.181300   1.433310   0.351765
 C  -1.435599   0.606207  -1.073928
 C   0.954379  -1.350249  -0.906918
 C  -1.879945  -1.437645   1.609459
 C   2.500875   1.175716   0.762031
 C  -2.512852   1.296068  -0.491131
 C   1.883811  -1.016702  -1.907466
 C  -2.383712  -1.983804   2.785177
 C   3.370580   2.214944   1.075206
 C  -3.556169   1.786374  -1.269245
 C   2.562667  -2.003826  -2.613881
 H  -0.042900  -0.827680   4.956587
 H   1.299988   4.840429   0.483172
 H  -2.483949   0.743361  -4.317704
 H   1.238838  -4.744202  -1.107716
 H   0.848132   0.140728   2.878469
 H  -0.238711   3.003719  -0.072701
 H  -0.638518  -0.124935  -2.942770
 H   0.038294  -2.999238   0.143479
 H  -2.414411  -1.602802   0.679055
 H   2.855851   0.151852   0.827402
 H  -2.542145   1.442246   0.584208
 H   2.087385   0.026236  -2.129795
 H  -3.293894  -2.573564   2.759984
 H   4.384912   1.993828   1.390056
 H  -4.378037   2.315674  -0.798853
 H   3.275944  -1.724310  -3.382004
 B  -2.286963  -2.377799   5.312418
 O  -1.706700  -2.221581   6.569634
 O  -3.441508  -3.153674   5.396229
 C  -2.518562  -2.913516   7.441873
 C  -3.573419  -3.480269   6.728488
 C  -2.388469  -3.073283   8.807730
 C  -4.552181  -4.235687   7.344425
 C  -3.374550  -3.837511   9.445396
 C  -4.432449  -4.405806   8.729981
 H  -1.564420  -2.627891   9.350200
 H  -5.366419  -4.670386   6.778951
 H  -3.313758  -3.989936   10.516391
 H  -5.177162  -4.990981   9.256223
 B   3.914043   4.703631   1.326999
 O   3.586760   6.056435   1.259576
 O   5.231707   4.536460   1.748640
 C   4.723064   6.733075   1.646471
 C   5.723281   5.808940   1.943837
 C   4.929256   8.094888   1.751472
 C   6.980776   6.199268   2.361404
 C   6.200596   8.504822   2.174135
 C   7.203633   7.578051   2.472277
 H   4.144124   8.802362   1.517484
 H   7.748976   5.471463   2.589162
 H   6.408530   9.563739   2.271179
 H   8.175294   7.931284   2.796403
 B  -4.719218   2.134848  -3.522614
 O  -4.810251   1.990461  -4.905579
 O  -5.820745   2.829617  -3.026725
 C  -5.988979   2.606213  -5.266368
 C  -6.603464   3.116420  -4.124021
 C  -6.548101   2.743056  -6.522036
 C  -7.808461   3.789470  -4.178846
 C  -7.770688   3.423778  -6.592175
 C  -8.386954   3.935329  -5.446566
 H  -6.060814   2.342062  -7.401500
 H  -8.275482   4.180682  -3.284109
 H  -8.247046   3.555175  -7.556424
 H  -9.332530   4.456280  -5.538498
 B   3.093440  -4.456297  -3.118594
 O   2.937723  -5.825719  -2.913251
 O   4.025715  -4.212467  -4.125244
 C   3.789302  -6.434698  -3.809184
 C   4.450794  -5.453852  -4.546221
 C   4.015809  -7.781917  -4.013461
 C   5.372475  -5.769958  -5.525298
 C   4.948561  -8.116815  -5.003824
 C   5.611856  -7.133167  -5.743008
 H   3.495622  -8.533951  -3.434192
 H   5.879480  -4.998608  -6.090698
 H   5.158792  -9.161612  -5.199176
 H   6.327066  -7.428976  -6.501250





