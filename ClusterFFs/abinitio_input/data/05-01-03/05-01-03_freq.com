%nproc=15
%mem=70GB
%chk=PMDA_BoronicAcid_Borosilicate_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from PMDA_BoronicAcid_Borosilicate.log

0 1
 C   2.677740  -0.758139  -0.557229
 C   1.226493  -0.477394  -0.351014
 C   0.125151  -1.141171  -0.878025
 H   0.218861  -1.993543  -1.538968
 C  -1.105905  -0.627236  -0.488541
 C  -2.475289  -1.089302  -0.861016
 C  -1.226687   0.471428   0.362787
 C  -2.677911   0.752312   0.568888
 C  -0.125332   1.135197   0.889817
 H  -0.219072   1.987542   1.550789
 C   1.105716   0.621265   0.500327
 C   2.475094   1.083277   0.872859
 N  -3.370779  -0.219490  -0.192956
 N   3.370597   0.213554   0.204737
 O  -3.176425   1.618796   1.240874
 O  -2.777132  -2.007343  -1.580030
 O   2.776873   2.001318   1.591910
 O   3.176241  -1.624585  -1.229290
 C  -4.792294  -0.310699  -0.276845
 C   4.792128   0.304901   0.288468
 C  -5.551926   0.850002  -0.440094
 C  -6.935704   0.752031  -0.515640
 H  -5.063303   1.813069  -0.493538
 C  -7.585413  -0.489632  -0.441354
 H  -7.526717   1.652183  -0.638580
 C  -6.796275  -1.639127  -0.282448
 C  -5.411983  -1.559784  -0.194964
 H  -7.278311  -2.608023  -0.222695
 H  -4.814479  -2.453282  -0.077663
 C   5.411596   1.554139   0.207066
 C   6.795871   1.633704   0.294476
 H   4.813974   2.447596   0.090101
 C   7.585249   0.484319   0.452887
 H   7.277717   2.602723   0.235080
 C   6.935768  -0.757490   0.526705
 C   5.551986  -0.855724   0.451190
 H   7.526923  -1.657600   0.649268
 H   5.063623  -1.818933   0.504207
 B  -9.134256  -0.588665  -0.532360
 O  -9.736641  -1.823123  -0.453869
 O  -9.874909   0.560067  -0.689736
Si  -11.254662  -2.467642  -0.493649
Si  -11.457241   1.005502  -0.828556
 O  -12.151279  -1.847486   0.749811
 O  -11.976734  -2.094438  -1.933615
 H  -11.170893  -3.914961  -0.354771
 O  -12.119348   0.285353  -2.161853
 O  -12.285888   0.532853   0.522073
 H  -11.543218   2.452682  -0.967535
 B  -12.582350  -0.625151   1.187084
 H  -13.236728  -0.567385   2.177536
 B  -12.338852  -0.978479  -2.636882
 H  -12.862171  -1.111392  -3.695661
 B   9.134111   0.583455   0.543663
 O   9.736343   1.818034   0.465420
 O   9.874873  -0.565229   0.700566
Si   11.254271   2.462713   0.505753
Si   11.457240  -1.010471   0.840071
 O   12.151316   1.842902  -0.737589
 O   11.976000   2.089292   1.945840
 H   11.170422   3.910054   0.367172
 O   12.118557  -0.290526   2.173837
 O   12.286423  -0.537409  -0.510052
 H   11.543278  -2.457672   0.978794
 B   12.582679   0.620723  -1.174954
 H   13.237145   0.563165  -2.165360
 B   12.338010   0.973269   2.649045
 H   12.861160   1.106042   3.707920






