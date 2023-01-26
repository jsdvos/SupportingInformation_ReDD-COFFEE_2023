%nproc=15
%mem=70GB
%chk=P_AmineBorane_Borazine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from P_AmineBorane_Borazine-1.log

0 1
 C  -4.269726  -0.630753   0.000770
 C  -4.254896   0.722431  -0.000598
 C  -2.849924   1.118627   0.004471
 C  -2.873813  -1.057718  -0.008021
 N  -2.037911   0.021478  -0.002706
 C  -2.483867  -2.409848  -0.009593
 C  -1.161744  -2.871833  -0.004860
 N  -0.023065  -2.101553  -0.002540
 C   1.099127  -2.895630  -0.000681
 C  -0.729878  -4.238488  -0.009340
 C   0.638608  -4.252879   0.002609
 C   2.430780  -2.461971   0.005270
 C   2.849992  -1.118741   0.004635
 N   2.037924  -0.021498  -0.002656
 C   4.269728   0.630711   0.000996
 C   4.254923  -0.722478  -0.000379
 C   1.161780   2.871778  -0.005019
 N   0.023082   2.101505  -0.002696
 C  -1.099126   2.895577  -0.000919
 C  -0.638551   4.252875   0.002248
 C   0.729868   4.238481  -0.009657
 C  -2.430721   2.461969   0.005074
 C   2.483842   2.409819  -0.009599
 C   2.873770   1.057599  -0.007902
 H  -5.127711  -1.283992   0.007928
 H  -5.098374   1.394341  -0.005076
 C  -3.555687  -3.454264  -0.018360
 H  -0.012625  -1.088527  -0.002628
 H  -1.394212  -5.086740  -0.019353
 H   1.285053  -5.114855   0.011918
 C   3.479679  -3.529531   0.015344
 H   5.127713   1.283953   0.008195
 H   5.098399  -1.394393  -0.004709
 H   0.012655   1.088482  -0.002523
 H  -1.285048   5.114816   0.011466
 H   1.394263   5.086686  -0.019835
 C  -3.479681   3.529461   0.015152
 C   3.555684   3.454211  -0.018309
 C   3.748133  -4.283908  -1.130997
 C   4.719731  -5.280594  -1.121511
 H   3.186489  -4.089219  -2.037659
 C   5.454658  -5.543305   0.037456
 H   4.908376  -5.863605  -2.015115
 C   5.193467  -4.790270   1.185265
 C   4.215588  -3.800113   1.173295
 H   5.764023  -4.978991   2.086850
 H   4.023982  -3.222206   2.070234
 C  -4.303613  -3.703995  -1.173317
 C  -5.303739  -4.671705  -1.183522
 H  -4.103828  -3.127178  -2.069160
 C  -5.575957  -5.422688  -0.036941
 H  -5.883297  -4.844088  -2.082636
 C  -4.828825  -5.181193   1.118883
 C  -3.834880  -4.206884   1.126664
 H  -5.026036  -5.763084   2.011387
 H  -3.264224  -4.028246   2.030992
 C  -3.748062   4.284032  -1.131094
 C  -4.719662   5.280694  -1.121532
 H  -3.186347   4.089513  -2.037749
 C  -5.454713   5.543212   0.037412
 H  -4.908261   5.863844  -2.015060
 C  -5.193666   4.789944   1.185092
 C  -4.215749   3.799810   1.173051
 H  -5.764356   4.978412   2.086642
 H  -4.024234   3.221753   2.069913
 C   4.303701   3.703966  -1.173218
 C   5.303812   4.671666  -1.183374
 H   4.103959   3.127183  -2.069094
 C   5.575989   5.422702  -0.036804
 H   5.883357   4.843976  -2.082506
 C   4.828808   5.181170   1.118947
 C   3.834859   4.206824   1.126691
 H   5.025832   5.762988   2.011520
 H   3.264242   4.028169   2.031042
 N   6.457363  -6.565452   0.048891
 B   6.443030  -7.573084   1.080662
 B   7.476973  -6.582254  -0.971103
 H   5.596064  -7.572489   1.915657
 N   7.460189  -8.575485   1.064536
 H   7.502061  -5.745441  -1.815804
 N   8.459290  -7.618251  -0.931509
 B   8.487909  -8.635176   0.072693
 H   7.439499  -9.282417   1.784018
 H   9.174919  -7.619212  -1.642649
 H   9.322155  -9.485398   0.082577
 N  -6.602214  -6.421089  -0.046093
 B  -7.616316  -6.417003   0.979562
 B  -6.617552  -7.425297  -1.081178
 H  -7.616750  -5.582371   1.826795
 N  -8.623175  -7.429206   0.942199
 H  -5.775514  -7.441734  -1.920982
 N  -7.658242  -8.403216  -1.062543
 B  -8.681575  -8.441750  -0.065135
 H  -9.334705  -7.415451   1.657312
 H  -7.658408  -9.107970  -1.784459
 H  -9.535900  -9.271809  -0.073085
 N  -6.457365   6.565400   0.048944
 B  -6.443153   7.572800   1.080944
 B  -7.476813   6.582495  -0.971211
 H  -5.596267   7.572028   1.916022
 N  -7.460299   8.575220   1.064925
 H  -7.501748   5.745945  -1.816174
 N  -8.459106   7.618505  -0.931509
 B  -8.487936   8.635112   0.073013
 H  -7.439651   9.282013   1.784544
 H  -9.174577   7.619694  -1.642809
 H  -9.322083   9.485429   0.082896
 N   6.602306   6.421132  -0.046017
 B   7.616242   6.417482   0.979777
 B   6.617755   7.425060  -1.081366
 H   7.616827   5.583340   1.827474
 N   8.622937   7.429861   0.942376
 H   5.775875   7.441474  -1.921324
 N   7.658268   8.403164  -1.062805
 B   8.681420   8.442116  -0.065230
 H   9.334309   7.416489   1.657649
 H   7.658463   9.107791  -1.784840
 H   9.535575   9.272359  -0.073204





