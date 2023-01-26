%nproc=15
%mem=70GB
%chk=TPG_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TPG_PrimaryAmine_Imine.log

0 1
 C   1.073225   0.940265  -0.000150
 C   1.318106  -0.459390  -0.000110
 O   2.603086  -0.861759  -0.000111
 C   0.277533  -1.399555  -0.000131
 C  -1.057031  -0.911743  -0.000113
 O  -2.048063  -1.823281  -0.000025
 C  -1.350956   0.459512  -0.000179
 C  -0.261260   1.371400  -0.000213
 O  -0.555139   2.685454  -0.000392
 H  -1.572295  -2.684076   0.000251
 H  -1.538476   2.703853  -0.000484
 H   3.110537  -0.019270  -0.000002
 N   2.241807   1.692879   0.000031
 C   2.321179   2.975472  -0.000261
 C   3.608283   3.676261  -0.000052
 C   3.609126   5.079595  -0.000672
 C   4.805508   5.791933  -0.000498
 C   6.020997   5.110522   0.000311
 C   6.032834   3.712332   0.000945
 C   4.840568   3.001436   0.000772
 H   1.433203   3.605732  -0.000811
 H   2.663181   5.611675  -0.001295
 H   4.789261   6.875917  -0.000980
 H   6.954492   5.661848   0.000448
 H   6.977628   3.180204   0.001583
 H   4.848338   1.918746   0.001292
 N   0.345137  -2.787896  -0.000088
 C   1.416273  -3.497838  -0.000237
 C   1.379749  -4.962885  -0.000067
 C   2.594685  -5.665194  -0.000242
 C   2.613504  -7.057472  -0.000125
 C   1.415696  -7.769514   0.000178
 C   0.198888  -7.080734   0.000357
 C   0.179257  -5.692744   0.000237
 H   2.406024  -3.043869  -0.000485
 H   3.528406  -5.111939  -0.000485
 H   3.560431  -7.585310  -0.000264
 H   1.426475  -8.853607   0.000267
 H  -0.734311  -7.632943   0.000584
 H  -0.762304  -5.158189   0.000382
 N  -2.587088   1.095092  -0.000114
 C  -3.737469   0.522378  -0.000002
 C  -4.987987   1.286526   0.000083
 C  -6.203662   0.585500  -0.000216
 C  -7.418791   1.265366  -0.000152
 C  -7.436537   2.658733   0.000232
 C  -6.231642   3.368149   0.000557
 C  -5.019824   2.691124   0.000491
 H  -3.839273  -0.561772  -0.000104
 H  -6.191384  -0.499754  -0.000495
 H  -8.349379   0.709231  -0.000387
 H  -8.380784   3.191437   0.000293
 H  -6.243260   4.452429   0.000879
 H  -4.086073   3.239200   0.000778






