%nproc=15
%mem=70GB
%chk=Adamantane_BoronicAcid_Boroxine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Adamantane_BoronicAcid_Boroxine.log

0 1
 C  -0.000000  -0.000003  -1.785822
 C   0.000000   0.000000   1.778218
 H  -0.796202   0.377528   2.429374
 H   0.796203  -0.377528   2.429374
 C  -1.160025   0.535999  -0.905766
 C   1.160025  -0.535998  -0.905766
 C  -0.535955  -1.160275   0.898196
 C   0.535955   1.160275   0.898196
 C  -0.628534   1.669552  -0.002295
 C   0.628534  -1.669552  -0.002295
 C  -1.669730  -0.628546  -0.004868
 C   1.669731   0.628547  -0.004867
 H  -0.281124   2.501580  -0.626693
 H   0.281124  -2.501580  -0.626693
 H  -2.501690  -0.281343   0.619655
 H   2.501691   0.281343   0.619656
 H  -1.438543   2.060700   0.620846
 H   1.438543  -2.060700   0.620845
 H  -2.060709  -1.438462  -0.628207
 H   2.060709   1.438463  -0.628206
 H  -0.377417  -0.795827  -2.437222
 H   0.377417   0.795827  -2.437221
 B  -2.353672   1.000713  -1.807592
 O  -2.986144   2.219322  -1.621932
 B  -4.055111   2.596318  -2.400496
 O  -2.834150   0.164268  -2.807038
 B  -3.901013   0.527030  -3.589865
 O  -4.510981   1.747062  -3.383999
 H  -4.584799   3.641728  -2.220607
 H  -4.297002  -0.198975  -4.439325
 B   2.353672  -1.000713  -1.807591
 O   2.986144  -2.219322  -1.621932
 B   4.055111  -2.596318  -2.400496
 O   2.834150  -0.164268  -2.807038
 B   3.901013  -0.527030  -3.589865
 O   4.510981  -1.747063  -3.383999
 H   4.584799  -3.641728  -2.220607
 H   4.297002   0.198974  -4.439326
 B  -1.000430  -2.351035   1.803780
 O  -0.164117  -2.831052   2.802909
 B  -0.530647  -3.892393   3.591508
 O  -2.225227  -2.974682   1.627647
 B  -2.604989  -4.039947   2.410148
 O  -1.754165  -4.497394   3.391666
 H   0.195876  -4.288467   4.440443
 H  -3.657531  -4.559228   2.241489
 B   1.000430   2.351035   1.803781
 O   0.164117   2.831052   2.802910
 B   0.530646   3.892394   3.591508
 O   2.225227   2.974682   1.627648
 B   2.604989   4.039947   2.410149
 O   1.754164   4.497395   3.391667
 H  -0.195877   4.288467   4.440443
 H   3.657531   4.559228   2.241491






