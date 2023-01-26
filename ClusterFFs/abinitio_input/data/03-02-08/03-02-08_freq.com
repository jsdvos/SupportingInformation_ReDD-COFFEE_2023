%nproc=15
%mem=70GB
%chk=BiPhenyl_Aldehyde_Benzobisoxazole_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from BiPhenyl_Aldehyde_Benzobisoxazole.log

0 1
 C   0.111011   3.558145  -0.112965
 C   1.227761   2.812683  -0.518929
 C   1.180354   1.427391  -0.517959
 C   0.022706   0.739713  -0.115771
 C  -1.088644   1.495887   0.286865
 C  -1.049184   2.883705   0.290733
 C  -0.022706  -0.739714  -0.115771
 C  -1.180354  -1.427393  -0.517959
 C  -1.227761  -2.812684  -0.518928
 C  -0.111011  -3.558146  -0.112965
 C   1.049184  -2.883706   0.290733
 C   1.088644  -1.495888   0.286865
 H  -2.120081  -3.335094  -0.840397
 H  -1.912845   3.450145   0.614971
 H   1.912845  -3.450146   0.614971
 H   2.120081   3.335093  -0.840398
 H  -2.043164  -0.867217  -0.858845
 H  -1.985093   0.990398   0.626302
 H   2.043164   0.867216  -0.858846
 H   1.985093  -0.990399   0.626302
 H  -1.256018   10.288057   0.452697
 C  -0.714680   9.371414   0.250257
 C   0.609365   9.444176  -0.216695
 H   1.060880   10.418283  -0.364380
 C   1.350969   8.298369  -0.492776
 H   2.370933   8.349583  -0.852645
 C   0.726898   7.066869  -0.287280
 N   1.179189   5.763446  -0.462965
 C  -0.593207   7.022668   0.178627
 O  -0.946948   5.702730   0.287206
 C  -1.350120   8.146271   0.459825
 H  -2.368805   8.077199   0.818879
 C   0.178576   5.012663  -0.118868
 H   1.256018  -10.288058   0.452698
 C   0.714680  -9.371415   0.250258
 C  -0.609365  -9.444178  -0.216695
 H  -1.060880  -10.418284  -0.364380
 C  -1.350969  -8.298370  -0.492775
 H  -2.370934  -8.349584  -0.852644
 C  -0.726898  -7.066870  -0.287280
 N  -1.179189  -5.763447  -0.462965
 C   0.593207  -7.022669   0.178628
 O   0.946948  -5.702732   0.287207
 C   1.350120  -8.146272   0.459825
 H   2.368805  -8.077201   0.818879
 C  -0.178576  -5.012664  -0.118868





