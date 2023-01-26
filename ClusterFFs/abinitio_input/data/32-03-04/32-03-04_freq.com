%nproc=15
%mem=70GB
%chk=T-brick_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from T-brick_PrimaryAmine_Imine.log

0 1
 N   1.104354  -2.479629  -0.579469
 C   0.716916  -1.188997  -0.272778
 C  -0.696596  -1.235988  -0.239771
 N  -1.128665  -2.518226  -0.523099
 C  -1.445742  -0.064364   0.007240
 C  -2.922550  -0.052972   0.054869
 C  -0.697282   1.103236   0.201141
 H  -1.218656   2.024916   0.430953
 C   0.702444   1.128145   0.155337
 H   1.216149   2.063652   0.345656
 C   1.464631  -0.020767  -0.079117
 C   2.941567  -0.008778  -0.127310
 C  -3.646412   1.048981  -0.427845
 C   3.635898   1.013042  -0.795130
 C   3.698013  -1.003239   0.513245
 C  -3.650725  -1.136280   0.576754
 C  -5.036408  -1.107602   0.630459
 H  -3.119936  -2.009722   0.929640
 C   5.087663  -0.994963   0.468967
 H   3.191682  -1.764280   1.097621
 C  -5.750465   0.015394   0.188617
 H  -5.590878  -1.948933   1.029568
 C  -5.033534   1.089119  -0.362985
 H  -3.117786   1.873028  -0.893348
 C   5.022865   1.039325  -0.824843
 H   3.074930   1.789893  -1.302020
 C   5.772059   0.019620  -0.219663
 H   5.648523  -1.747881   1.011177
 H   5.551164   1.835529  -1.336034
 H  -5.567972   1.933905  -0.782662
 C  -0.043366  -3.233902  -0.728571
 H   2.045758  -2.765792  -0.792569
 C  -0.028771  -4.655929  -1.067009
 C  -1.231877  -5.289260  -1.414635
 C  -1.255364  -6.636449  -1.740995
 H  -2.140733  -4.700943  -1.433833
 C   1.152258  -5.414627  -1.070645
 C  -0.072724  -7.397624  -1.730701
 H  -2.187869  -7.100325  -2.041191
 C   1.134815  -6.758118  -1.415398
 H   2.099791  -4.964025  -0.795648
 H   2.048733  -7.339739  -1.427728
 N  -7.151454  -0.017341   0.266431
 C  -7.805421   1.035276   0.573458
 C  -9.272268   1.079592   0.607871
 C  -9.913946   2.267992   0.982268
 C  -11.304388   2.341190   1.023190
 C  -12.068294   1.224461   0.689814
 C  -11.436569   0.034399   0.315360
 C  -10.050842  -0.040108   0.272870
 H  -7.295901   1.970290   0.846365
 H  -9.318564   3.137660   1.242438
 H  -11.789735   3.265865   1.314178
 H  -13.150807   1.278020   0.720498
 H  -12.031436  -0.834223   0.055734
 H  -9.547461  -0.954547  -0.016180
 N   7.170917   0.098547  -0.283874
 C   7.870509  -0.957723  -0.443105
 C   9.337501  -0.949231  -0.451659
 C   10.028160  -2.150505  -0.662230
 C   11.420687  -2.174727  -0.674712
 C   12.136714  -0.995829  -0.476476
 C   11.455856   0.207312  -0.265764
 C   10.067924   0.233488  -0.252000
 H   7.401053  -1.939479  -0.599748
 H   9.469862  -3.068530  -0.817148
 H   11.944842  -3.109332  -0.838477
 H   13.220781  -1.011168  -0.485512
 H   12.014068   1.123882  -0.111369
 H   9.526986   1.157581  -0.089193
 N  -0.030250  -8.755596  -2.072903
 C  -0.950471  -9.549155  -1.678719
 C  -0.999408  -10.964678  -2.059180
 C  -2.027020  -11.773437  -1.554649
 C  -2.101897  -13.120858  -1.898957
 C  -1.148458  -13.672705  -2.752334
 C  -0.119608  -12.873222  -3.260200
 C  -0.043444  -11.529799  -2.918896
 H  -1.754646  -9.208739  -1.011451
 H  -2.769408  -11.342750  -0.890054
 H  -2.900897  -13.737430  -1.503464
 H  -1.204149  -14.721207  -3.022647
 H   0.621145  -13.303861  -3.924557
 H   0.746904  -10.897992  -3.305225





