%nproc=15
%mem=70GB
%chk=T-brick_BoronicAcid_Boroxine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from T-brick_BoronicAcid_Boroxine.log

0 1
 N  -1.115478   0.114159   0.050703
 C  -0.726720  -1.210317   0.070061
 C   0.687893  -1.170167   0.065119
 N   1.118613   0.141384   0.039414
 C   1.437110  -2.367006   0.037654
 C   2.915360  -2.382985   0.035155
 C   0.692277  -3.551154   0.006015
 H   1.216861  -4.498946   0.018705
 C  -0.708564  -3.567979   0.001984
 H  -1.220771  -4.523243   0.007432
 C  -1.471157  -2.396891   0.039828
 C  -2.948831  -2.405624   0.041808
 C   3.615470  -3.347508  -0.709780
 C  -3.662296  -3.259822  -0.813987
 C  -3.675185  -1.563170   0.901572
 C   3.654901  -1.450769   0.782263
 C   5.042437  -1.497371   0.793976
 H   3.133128  -0.692404   1.349733
 C  -5.064479  -1.571682   0.897994
 H  -3.142944  -0.932008   1.605156
 C   5.747074  -2.464115   0.056664
 H   5.596940  -0.775428   1.383429
 C   5.003144  -3.386359  -0.698209
 H   3.065108  -4.053897  -1.320609
 C  -5.051290  -3.266712  -0.812244
 H  -3.116960  -3.903473  -1.494728
 C  -5.781851  -2.420682   0.039050
 H  -5.608534  -0.923832   1.576604
 H  -5.586055  -3.928267  -1.484842
 H   5.525255  -4.133204  -1.286172
 C   0.031739   0.883074   0.024527
 H  -2.062328   0.440117  -0.055208
 C   0.016551   2.346064  -0.010659
 C   1.218764   3.033948  -0.242189
 C   1.231842   4.419963  -0.285044
 H   2.125307   2.460387  -0.387939
 C  -1.163301   3.081697   0.181366
 C   0.054496   5.166638  -0.100164
 H   2.164598   4.941911  -0.466826
 C  -1.141395   4.469536   0.134814
 H  -2.102547   2.579285   0.385886
 H  -2.058615   5.027200   0.286814
 B   7.287981  -2.507987   0.069496
 O   7.981506  -3.460127  -0.667244
 B   9.352776  -3.503427  -0.659358
 O   8.022977  -1.597654   0.818149
 B   9.394358  -1.633102   0.832730
 O   10.059320  -2.588001   0.092165
 H   9.929900  -4.322374  -1.294140
 H   10.007237  -0.848648   1.477394
 B  -7.325869  -2.427986   0.035806
 O  -8.041605  -1.589878   0.878651
 B  -9.415448  -1.593659   0.881595
 O  -8.031218  -3.271736  -0.809346
 B  -9.404903  -3.282995  -0.814757
 O  -10.093860  -2.441994   0.033053
 H  -10.013854  -0.872517   1.607713
 H  -9.994461  -4.009577  -1.542529
 B   0.074325   6.710945  -0.150509
 O   1.255293   7.398848  -0.382524
 B   1.278416   8.771785  -0.428393
 O  -1.090052   7.441620   0.035439
 B  -1.078260   8.814913  -0.008355
 O   0.108313   9.476468  -0.240580
 H   2.295447   9.346919  -0.627776
 H  -2.080862   9.426757   0.152281





