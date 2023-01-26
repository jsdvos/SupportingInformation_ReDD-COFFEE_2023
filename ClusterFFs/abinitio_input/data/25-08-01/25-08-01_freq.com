%nproc=15
%mem=70GB
%chk=Pc_Catechol_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Pc_Catechol_BoronateEster.log

0 1
 N   2.706531  -2.015741   0.000588
 C   6.328892   1.659503   0.000187
 C   6.537405   0.262885   0.000295
 C   5.508248  -0.651066   0.000428
 C   1.311839  -4.018630   0.000535
 C   2.230179  -5.071472   0.000557
 C   1.656695  -6.328311   0.000401
 C   0.265841  -6.536220   0.000186
 H   5.668081  -1.720536   0.000492
 H   3.300060  -4.912922   0.000666
 C   1.491960  -2.565176   0.000505
 C   2.923455  -0.719045   0.000523
 C   5.077601   2.232901   0.000227
 C   4.226341  -0.084345   0.000419
 C  -0.079723  -4.226654   0.000315
 C  -0.650083  -5.501988   0.000123
 N   0.287739  -1.926519   0.000188
 N   1.996451   0.298566   0.000692
 C   4.017176   1.316392   0.000319
 H   4.918051   3.302408   0.000165
 C  -0.677014  -2.889446   0.000114
 H  -1.719572  -5.663149  -0.000037
 C   2.585681   1.542663   0.000265
 N  -1.999175  -2.719277  -0.000049
 N   1.999178   2.719280   0.000049
 C  -2.585677  -1.542660  -0.000265
 C   0.677018   2.889450  -0.000114
 C  -4.017172  -1.316388  -0.000318
 N  -1.996447  -0.298562  -0.000693
 C   0.079727   4.226658  -0.000316
 N  -0.287735   1.926523  -0.000187
 C  -5.077598  -2.232897  -0.000227
 C  -4.226336   0.084349  -0.000419
 C  -2.923451   0.719049  -0.000523
 C  -1.311834   4.018635  -0.000536
 C   0.650086   5.501991  -0.000124
 C  -1.491956   2.565180  -0.000505
 C  -6.328890  -1.659500  -0.000187
 H  -4.918046  -3.302405  -0.000165
 C  -5.508244   0.651070  -0.000428
 N  -2.706526   2.015745  -0.000589
 C  -2.230175   5.071475  -0.000558
 C  -0.265839   6.536221  -0.000187
 H   1.719574   5.663155   0.000036
 C  -6.537400  -0.262882  -0.000294
 H  -5.668079   1.720539  -0.000492
 C  -1.656693   6.328315  -0.000402
 H  -3.300056   4.912922  -0.000666
 H   0.995402   0.149245   0.000291
 H  -0.995397  -0.149242  -0.000292
 B   8.507462   1.270880   0.000035
 C   10.024887   1.497569  -0.000111
 C   10.912595   0.408217  -0.000200
 C   10.555497   2.798793  -0.000159
 C   12.289176   0.612895  -0.000333
 C   11.931810   3.005291  -0.000292
 C   12.799109   1.911956  -0.000379
 O   7.888878   0.020178   0.000146
 O   7.550475   2.286414   0.000077
 H   12.964801  -0.235060  -0.000402
 H   12.330159   4.013664  -0.000328
 H   9.881159   3.648221  -0.000093
 H   10.515900  -0.601185  -0.000167
 H   13.871720   2.072180  -0.000481
 B   1.271375  -8.506628   0.000063
 C   1.498248  -10.025662  -0.000089
 C   0.409314  -10.913641  -0.000198
 C   2.799077  -10.556838  -0.000128
 C   0.613737  -12.290416  -0.000343
 C   3.005696  -11.933288  -0.000272
 C   1.912582  -12.800828  -0.000380
 O   0.021012  -7.889856   0.000071
 O   2.286672  -7.551221   0.000263
 H  -0.234457  -12.965853  -0.000427
 H   4.014187  -12.331527  -0.000302
 H   3.648450  -9.882386  -0.000047
 H  -0.600006  -10.516675  -0.000171
 H   2.072734  -13.873485  -0.000491
 B  -8.507459  -1.270870  -0.000034
 C  -10.024882  -1.497571   0.000111
 C  -10.912601  -0.408228   0.000200
 C  -10.555481  -2.798801   0.000160
 C  -12.289180  -0.612918   0.000333
 C  -11.931792  -3.005311   0.000292
 C  -12.799101  -1.911983   0.000379
 O  -7.888874  -0.020173  -0.000145
 O  -7.550472  -2.286409  -0.000076
 H  -12.964812   0.235031   0.000402
 H  -12.330132  -4.013687   0.000328
 H  -9.881134  -3.648222   0.000093
 H  -10.515915   0.601178   0.000167
 H  -13.871710  -2.072218   0.000481
 B  -1.271369   8.506632  -0.000063
 C  -1.498255   10.025664   0.000089
 C  -0.409330   10.913654   0.000199
 C  -2.799089   10.556826   0.000128
 C  -0.613766   12.290427   0.000344
 C  -3.005722   11.933274   0.000272
 C  -1.912617   12.800826   0.000380
 O  -0.021010   7.889860  -0.000072
 O  -2.286670   7.551223  -0.000263
 H   0.234421   12.965872   0.000428
 H  -4.014217   12.331504   0.000302
 H  -3.648455   9.882366   0.000046
 H   0.599994   10.516698   0.000171
 H  -2.072780   13.873481   0.000492






