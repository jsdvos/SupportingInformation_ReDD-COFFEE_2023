%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_Aldehyde_Azine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Triazine-TrisPhenyl_Aldehyde_Azine.log

0 1
 C  -3.389463   4.465335   0.000123
 C  -3.921852   3.163744  -0.000094
 C  -3.082575   2.063110  -0.000090
 C  -1.686441   2.225549   0.000122
 C  -1.152961   3.520754   0.000323
 C  -1.995201   4.624250   0.000326
 C  -0.792018   1.044917   0.000129
 N   0.530649   1.259110   0.000111
 C   1.300991   0.163753   0.000102
 N   0.825159  -1.088804   0.000105
 C  -0.508619  -1.208264   0.000113
 N  -1.355439  -0.169897   0.000131
 C   2.770675   0.347976   0.000112
 C   3.625548  -0.761691   0.000288
 C   5.002327  -0.584110   0.000272
 C   5.561914   0.702776   0.000076
 C   4.700973   1.814690  -0.000085
 C   3.328147   1.638243  -0.000067
 C  -1.083929  -2.573146   0.000116
 C  -0.245245  -3.701045  -0.000103
 C  -0.778824  -4.978184  -0.000109
 C  -2.172232  -5.167868   0.000112
 C  -3.006935  -4.039821   0.000344
 C  -2.472361  -2.758695   0.000341
 H  -3.491631   1.061621  -0.000264
 H  -4.997205   3.038636  -0.000270
 H  -1.571562   5.623157   0.000485
 H  -0.078739   3.647455   0.000480
 H   3.198111  -1.755324   0.000437
 H   5.655547  -1.450476   0.000415
 H   5.130373   2.808495  -0.000228
 H   2.665390   2.493261  -0.000197
 H   0.826601  -3.554567  -0.000272
 H  -0.132834  -5.846936  -0.000280
 H  -4.083838  -4.172360   0.000533
 H  -3.119187  -1.891737   0.000528
 C  -4.239536   5.653325   0.000150
 N  -5.521842   5.577588   0.000023
 H  -3.751227   6.631961   0.000315
 N  -6.133099   6.823995   0.000090
 C  -7.415527   6.744702  -0.000164
 C  -8.274166   7.926846  -0.000162
 H  -7.898706   5.763282  -0.000404
 C  -9.666441   7.754762  -0.000393
 C  -10.517616   8.856659  -0.000387
 H  -10.080165   6.751506  -0.000581
 C  -9.986165   10.144854  -0.000155
 H  -11.591717   8.710609  -0.000563
 C  -8.599756   10.327045   0.000067
 H  -10.646427   11.004825  -0.000147
 C  -7.748432   9.230750   0.000062
 H  -8.186807   11.329546   0.000237
 H  -6.673470   9.360770   0.000228
 C   7.015782   0.844874   0.000032
 N   7.591434   1.993209  -0.000080
 H   7.619089  -0.067373   0.000125
 N   8.976475   1.899230  -0.000090
 C   9.549174   3.049414  -0.000089
 C   11.002282   3.201691  -0.000103
 H   8.940972   3.958665  -0.000060
 C   11.549658   4.493363  -0.000049
 C   12.929553   4.679285  -0.000063
 H   10.887846   5.353419   0.000002
 C   13.779213   3.574763  -0.000133
 H   13.340316   5.682428  -0.000022
 C   13.243516   2.283115  -0.000191
 H   14.854132   3.716350  -0.000139
 C   11.868397   2.094265  -0.000174
 H   13.905059   1.424105  -0.000249
 H   11.443297   1.098405  -0.000222
 C  -2.776092  -6.498011   0.000098
 N  -2.069423  -7.570706  -0.000084
 H  -3.867777  -6.564370   0.000258
 N  -2.843343  -8.723189  -0.000040
 C  -2.133607  -9.794256  -0.000159
 C  -2.728300  -11.128818  -0.000136
 H  -1.042067  -9.722179  -0.000279
 C  -1.883384  -12.248706  -0.000240
 C  -2.412341  -13.536681  -0.000207
 H  -0.807644  -12.105610  -0.000347
 C  -3.793715  -13.720245  -0.000072
 H  -1.748982  -14.393988  -0.000288
 C  -4.644453  -12.610485   0.000028
 H  -4.208557  -14.721944  -0.000042
 C  -4.120420  -11.325183  -0.000002
 H  -5.719153  -12.753871   0.000131
 H  -4.770308  -10.459104   0.000079





