%nproc=15
%mem=70GB
%chk=F_Ketoenamine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from F_Ketoenamine_Ketoenamine.log

0 1
 F   1.322884  -2.241851   0.782218
 C   0.693531  -1.133954   0.370071
 C  -0.693039  -1.100724   0.437309
 F  -1.332885  -2.191111   0.899900
 C  -1.421676   0.028969   0.070193
 C  -0.693530   1.133913  -0.370029
 F  -1.322884   2.241810  -0.782174
 C   0.693040   1.100683  -0.437266
 F   1.332886   2.191070  -0.899856
 C   1.421677  -0.029010  -0.070151
 C  -2.928693   0.095376   0.185197
 O  -3.431144   1.068631   0.762440
 C  -3.688224  -0.969480  -0.407431
 C  -5.060985  -0.988800  -0.382303
 H  -3.172080  -1.782370  -0.896686
 N  -5.828130  -0.041316   0.187652
 H  -5.585339  -1.807479  -0.861207
 C  -7.231412  -0.009589   0.266886
 H  -5.297906   0.735194   0.587772
 C  -8.031353  -1.133514   0.026018
 C  -7.839082   1.203077   0.619443
 C  -9.416763  -1.030490   0.116682
 H  -7.585365  -2.091354  -0.210022
 C  -10.022064   0.177052   0.460527
 H  -10.024971  -1.907992  -0.071209
 C  -9.222924   1.290988   0.716823
 H  -7.220922   2.074349   0.807688
 H  -11.100590   0.247313   0.534180
 H  -9.677495   2.236484   0.989455
 C   2.928693  -0.095415  -0.185154
 O   3.431145  -1.068649  -0.762433
 C   3.688220   0.969471   0.407426
 C   5.060980   0.988808   0.382280
 H   3.172072   1.782375   0.896654
 N   5.828131   0.041306  -0.187639
 H   5.585329   1.807515   0.861141
 C   7.231413   0.009596  -0.266890
 H   5.297913  -0.735234  -0.587708
 C   8.031341   1.133546  -0.026094
 C   7.839096  -1.203079  -0.619389
 C   9.416751   1.030538  -0.116771
 H   7.585342   2.091392   0.209900
 C   10.022065  -0.177014  -0.460558
 H   10.024949   1.908059   0.071065
 C   9.222938  -1.290976  -0.716782
 H   7.220946  -2.074371  -0.807578
 H   11.100591  -0.247263  -0.534221
 H   9.677519  -2.236480  -0.989369





