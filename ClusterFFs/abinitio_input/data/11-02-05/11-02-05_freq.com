%nproc=15
%mem=70GB
%chk=3_Phenyl_Aldehyde_Hydrazone_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from 3_Phenyl_Aldehyde_Hydrazone.log

0 1
 C   0.472806   1.319810  -0.006848
 C  -0.909841   1.058093  -0.007577
 C  -1.381361  -0.251970  -0.007374
 C  -0.463406  -1.318580  -0.007241
 C   0.906891  -1.071853  -0.006626
 C   1.371555   0.256412  -0.006633
 H  -0.843817  -2.333741  -0.009148
 H   2.440918   0.434491  -0.008062
 H  -1.598778   1.895113  -0.009696
 H   3.474510   8.850643   1.357798
 C   2.565563   8.741012   0.777452
 C   2.039349   9.832083   0.088232
 H   2.546332   10.789800   0.121162
 C   0.853022   9.692688  -0.633637
 H   0.437367   10.541876  -1.163912
 C   0.198175   8.466450  -0.670140
 H  -0.732854   8.344487  -1.209975
 C   0.732876   7.360403  -0.000152
 C  -0.036042   6.072284  -0.074674
 C   1.917303   7.508327   0.732314
 H   2.319740   6.679029   1.303686
 O  -1.240166   6.033966  -0.216909
 N   0.755531   4.933823   0.015581
 H   1.766081   5.026984  -0.035800
 N   0.199499   3.699317   0.008321
 C   0.987521   2.691334  -0.008209
 H   2.078830   2.808719  -0.022616
 H   5.931545  -7.437817   1.347958
 C   6.291341  -6.593355   0.771386
 C   7.501241  -6.679108   0.084965
 H   8.078373  -7.596320   0.116244
 C   7.973999  -5.578894  -0.631891
 H   8.918710  -5.640365  -1.159922
 C   7.237875  -4.399597  -0.666260
 H   7.597902  -3.529977  -1.202224
 C   6.010790  -4.313651   0.000831
 C   5.278161  -3.004369  -0.071368
 C   5.546425  -5.416472   0.728429
 H   4.625601  -5.353547   1.297850
 O   5.846067  -1.941405  -0.209928
 N   3.896408  -3.122256   0.016224
 H   3.472735  -4.044276  -0.037668
 N   3.104410  -2.024092   0.009960
 C   1.837562  -2.203144  -0.008040
 H   1.393967  -3.207092  -0.024142
 H  -9.406010  -1.408414   1.347668
 C  -8.856100  -2.143932   0.771810
 C  -9.537366  -3.148134   0.086377
 H  -10.620340  -3.187132   0.117737
 C  -8.822881  -4.109815  -0.629572
 H  -9.350078  -4.896662  -1.156841
 C  -7.433424  -4.064773  -0.664087
 H  -6.861845  -4.812970  -1.199450
 C  -6.743390  -3.045793   0.001981
 C  -5.243268  -3.068479  -0.070660
 C  -7.464329  -2.090100   0.728720
 H  -6.947846  -1.324469   1.297225
 O  -4.608257  -4.092828  -0.208817
 N  -4.652764  -1.813647   0.015735
 H  -5.238446  -0.985046  -0.038322
 N  -3.305584  -1.678234   0.009397
 C  -2.826475  -0.491892  -0.009075
 H  -3.473681   0.394560  -0.025438






