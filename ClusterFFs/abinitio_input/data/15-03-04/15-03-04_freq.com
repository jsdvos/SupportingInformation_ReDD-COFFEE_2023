%nproc=15
%mem=70GB
%chk=ExtendedTrisPhenyl_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from ExtendedTrisPhenyl_PrimaryAmine_Imine.log

0 1
 C   1.567388  -2.352778  -0.268226
 C   0.776975  -1.170271  -0.268205
 C  -0.623848  -1.254960  -0.267627
 H  -1.102515  -2.225485  -0.267547
 C  -1.409361  -0.091976  -0.267560
 C  -2.828662  -0.184888  -0.266883
 C  -0.782289   1.163519  -0.268030
 H  -1.383496   2.063293  -0.268190
 C   0.617611   1.262365  -0.268577
 C   1.247041   2.537873  -0.268979
 C   1.391281   0.091522  -0.268595
 H   2.471108   0.162266  -0.269264
 C   1.783948   3.622690  -0.269972
 C   2.414184   4.895204  -0.268208
 C  -4.036520  -0.263668  -0.266915
 C  -5.453498  -0.356409  -0.264183
 C   2.240594  -3.358724  -0.268643
 C   3.030884  -4.538525  -0.266406
 C   1.650385   6.079012  -0.281496
 C   2.267007   7.319212  -0.296232
 H   0.569291   6.009741  -0.290945
 C   3.666532   7.424728  -0.261590
 H   1.679652   8.229324  -0.320845
 C   4.432962   6.246450  -0.267774
 C   3.816818   5.003529  -0.271979
 H   5.514408   6.310654  -0.302888
 H   4.416593   4.101305  -0.287741
 C   2.427353  -5.809302  -0.275195
 C   3.199292  -6.961976  -0.270700
 H   1.346403  -5.880924  -0.295219
 C   4.602643  -6.882920  -0.259189
 H   2.717324  -7.932053  -0.309786
 C   5.207148  -5.616130  -0.288717
 C   4.437783  -4.464489  -0.274246
 H   6.288919  -5.559118  -0.309198
 H   4.915297  -3.492067  -0.279755
 C  -6.094600  -1.610962  -0.272347
 C  -7.476798  -1.699491  -0.286738
 H  -5.492463  -2.511548  -0.278331
 C  -8.269930  -0.541441  -0.256749
 H  -7.969639  -2.664149  -0.307623
 C  -7.634918   0.712529  -0.267926
 C  -6.250606   0.802762  -0.272543
 H  -8.232847   1.615760  -0.306978
 H  -5.770803   1.774040  -0.292424
 N   4.232148   8.706129  -0.279854
 C   5.262426   8.968364   0.428187
 C   5.939953   10.268834   0.398779
 C   7.046899   10.482870   1.231535
 C   7.712413   11.706304   1.224991
 C   7.276368   12.728637   0.384406
 C   6.172565   12.523973  -0.449448
 C   5.508288   11.305026  -0.445224
 H   5.688984   8.225947   1.117373
 H   7.385631   9.686168   1.886444
 H   8.567505   11.861318   1.872919
 H   7.792326   13.682189   0.376803
 H   5.834510   13.320165  -1.103150
 H   4.652464   11.131639  -1.085996
 N   5.433441  -8.010642  -0.276995
 C   5.145960  -9.037162   0.426709
 C   5.938339  -10.271022   0.397143
 C   5.570227  -11.340937   1.224430
 C   6.301738  -12.526101   1.217691
 C   7.409950  -12.652467   0.382370
 C   7.784636  -11.589940  -0.446019
 C   7.056411  -10.408075  -0.441601
 H   4.286625  -9.040743   1.112006
 H   4.707167  -11.241556   1.875288
 H   6.008363  -13.347463   1.861368
 H   7.981496  -13.573774   0.374634
 H   8.646981  -11.689652  -1.095593
 H   7.334184  -9.576939  -1.078102
 N  -9.662133  -0.694851  -0.275066
 C  -10.406342   0.067565   0.429532
 C  -11.871225   0.001492   0.398493
 C  -12.612786   0.854816   1.227072
 C  -14.004999   0.816947   1.218624
 C  -14.669513  -0.076043   0.380237
 C  -13.937633  -0.931294  -0.449492
 C  -12.549928  -0.894629  -0.443339
 H  -9.978703   0.811825   1.116075
 H  -12.094418   1.549840   1.880210
 H  -14.568930   1.481305   1.863317
 H  -15.753227  -0.107928   0.371137
 H  -14.455937  -1.625384  -1.101506
 H  -11.969714  -1.550279  -1.080953






