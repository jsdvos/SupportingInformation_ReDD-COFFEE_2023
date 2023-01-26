%nproc=15
%mem=70GB
%chk=P_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from P_BoronicAcid_BoronateEster.log

0 1
 C  -4.263552   0.673976   0.001183
 C  -4.262758  -0.678930  -0.000846
 C  -2.861800  -1.089494  -0.008559
 C  -2.863071   1.086177   0.008720
 N  -2.038262  -0.001175   0.000025
 C  -2.457788   2.433655   0.007229
 C  -1.131778   2.883916  -0.000525
 N  -0.001191   2.102206   0.000152
 C   1.128512   2.885185   0.000788
 C  -0.686521   4.246349   0.004355
 C   0.681712   4.247118  -0.004063
 C   2.455037   2.436455  -0.007040
 C   2.861903   1.089450  -0.008590
 N   2.038374   0.001127   0.000092
 C   4.263659  -0.674020   0.000953
 C   4.262867   0.678885  -0.001043
 C   1.131874  -2.883968  -0.000613
 N   0.001290  -2.102258   0.000029
 C  -1.128409  -2.885239   0.000645
 C  -0.681617  -4.247168  -0.004227
 C   0.686619  -4.246399   0.004234
 C  -2.454934  -2.436500  -0.007106
 C   2.457883  -2.433710   0.007157
 C   2.863185  -1.086226   0.008653
 H  -5.115892   1.334520  -0.003121
 H  -5.114320  -1.340479   0.003572
 C  -3.518848   3.488660   0.016345
 H  -0.000667   1.089098   0.000029
 H  -1.340858   5.102318   0.012639
 H   1.335086   5.103822  -0.012305
 C   3.514874   3.492692  -0.016178
 H   5.115992  -1.334573  -0.003492
 H   5.114435   1.340423   0.003304
 H   0.000754  -1.089150   0.000056
 H  -1.334977  -5.103883  -0.012502
 H   1.340939  -5.102380   0.012478
 C  -3.514770  -3.492742  -0.016251
 C   3.518938  -3.488722   0.016282
 C   3.766997   4.259631   1.128097
 C   4.752226   5.240984   1.118108
 H   3.188547   4.074985   2.026245
 C   5.513116   5.487750  -0.036139
 H   4.939806   5.822462   2.014213
 C   5.253423   4.715632  -1.180197
 C   4.270112   3.732571  -1.170908
 H   5.827371   4.892062  -2.083380
 H   4.077615   3.142892  -2.059947
 C  -4.274442   3.727635   1.171027
 C  -5.258909   4.709546   1.180259
 H  -4.081326   3.138163   2.060069
 C  -5.519418   5.481367   0.036192
 H  -5.833123   4.885285   2.083408
 C  -4.758165   5.235525  -1.118003
 C  -3.771788   4.255323  -1.127935
 H  -4.946357   5.816802  -2.014108
 H  -3.193058   4.071361  -2.026044
 C  -3.766827  -4.259741   1.127998
 C  -4.752078  -5.241072   1.118027
 H  -3.188311  -4.075144   2.026113
 C  -5.513045  -5.487762  -0.036181
 H  -4.939609  -5.822597   2.014111
 C  -5.253397  -4.715603  -1.180226
 C  -4.270069  -3.732559  -1.170954
 H  -5.827389  -4.891994  -2.083389
 H  -4.077619  -3.142841  -2.059975
 C   4.274591  -3.727626   1.170944
 C   5.259054  -4.709534   1.180193
 H   4.081514  -3.138094   2.059954
 C   5.519512  -5.481441   0.036165
 H   5.833317  -4.885217   2.083321
 C   4.758193  -5.235669  -1.118008
 C   3.771810  -4.255473  -1.127954
 H   4.946346  -5.817000  -2.014086
 H   3.193031  -4.071578  -2.026045
 B   6.601065   6.573048  -0.046680
 O   6.923523   7.382543   1.041110
 O   7.400589   6.883490  -1.145226
 C   7.939242   8.204819   0.603737
 C   8.229230   7.901471  -0.725458
 C   8.615761   9.191384   1.294367
 C   9.210693   8.569070  -1.431866
 C   9.614121   9.876770   0.589623
 C   9.904981   9.572516  -0.743248
 H   8.381036   9.417318   2.326633
 H   9.426387   8.323832  -2.463876
 H   10.171494   10.659165   1.090768
 H   10.683894   10.123172  -1.257080
 B  -6.608690   6.565378   0.046674
 O  -7.408713   6.874813   1.145119
 O  -6.931946   7.374531  -1.041126
 C  -8.238506   7.891833   0.725292
 C  -7.948705   8.195590  -0.603851
 C  -9.220831   8.558259   1.431612
 C  -8.626276   9.181407  -1.294514
 C  -9.916197   9.560931   0.742956
 C  -9.625522   9.865594  -0.589862
 H  -9.436371   8.312720   2.463583
 H  -8.391677   9.407667  -2.326737
 H  -10.695814   10.110655   1.256718
 H  -10.183740   10.647368  -1.091035
 B  -6.601103  -6.572951  -0.046705
 O  -6.923573  -7.382456   1.041073
 O  -7.400728  -6.883270  -1.145214
 C  -7.939438  -8.204588   0.603739
 C  -8.229470  -7.901150  -0.725425
 C  -8.616027  -9.191095   1.294380
 C  -9.211062  -8.568607  -1.431791
 C  -9.614520  -9.876331   0.589678
 C  -9.905429  -9.571990  -0.743162
 H  -8.381258  -9.417092   2.326621
 H  -9.426798  -8.323308  -2.463778
 H  -10.171957  -10.658676   1.090831
 H  -10.684444  -10.122531  -1.256964
 B   6.608735  -6.565470   0.046658
 O   7.408774  -6.874910   1.145111
 O   6.931972  -7.374680  -1.041113
 C   8.238518  -7.891974   0.725311
 C   7.948695  -8.195763  -0.603819
 C   9.220821  -8.558421   1.431643
 C   8.626220  -9.181631  -1.294456
 C   9.916140  -9.561146   0.743016
 C   9.625442  -9.865840  -0.589790
 H   9.436379  -8.312858   2.463605
 H   8.391604  -9.407916  -2.326669
 H   10.695737  -10.110886   1.256792
 H   10.183622  -10.647654  -1.090943





