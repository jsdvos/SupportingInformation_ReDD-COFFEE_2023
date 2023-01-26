%nproc=15
%mem=70GB
%chk=BiPhenyl_BoronicAcid_Borosilicate_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from BiPhenyl_BoronicAcid_Borosilicate.log

0 1
 C  -3.575417  -0.000030  -0.000222
 C  -2.850152  -1.129077   0.413094
 C  -1.460934  -1.132244   0.413045
 C  -0.741001  -0.000041  -0.000051
 C  -1.460877   1.132162  -0.413249
 C  -2.850093   1.129006  -0.413467
 C   0.741001  -0.000041   0.000051
 C   1.460877   1.132162   0.413249
 C   2.850093   1.129005   0.413468
 C   3.575417  -0.000030   0.000222
 C   2.850152  -1.129077  -0.413094
 C   1.460934  -1.132244  -0.413045
 H   3.388757   2.009593   0.745493
 H  -3.388757   2.009593  -0.745492
 H   3.388858  -2.009670  -0.745034
 H  -3.388858  -2.009670   0.745034
 H   0.924322   2.008167   0.759658
 H  -0.924322   2.008167  -0.759657
 H  -0.924431  -2.008256   0.759514
 H   0.924431  -2.008256  -0.759514
 B  -5.127311  -0.000005  -0.000244
 O  -5.805197   1.125082  -0.415010
 O  -5.805256  -1.125043   0.414499
Si  -7.359627   1.637789  -0.606935
Si  -7.359614  -1.637785   0.606917
 O  -8.141043   0.653245  -1.682197
 O  -8.143047   1.592535   0.849221
 H  -7.362712   3.003378  -1.113678
 O  -8.140656  -0.653244   1.682457
 O  -8.143502  -1.592551  -0.848985
 H  -7.362515  -3.003366   1.113684
 B  -8.471016  -0.668632  -1.802860
 H  -9.063237  -1.026564  -2.769399
 B  -8.469853   0.668799   1.803511
 H  -9.060798   1.027006   2.770727
 B   5.127311  -0.000005   0.000244
 O   5.805197   1.125082   0.415010
 O   5.805256  -1.125044  -0.414499
Si   7.359627   1.637789   0.606935
Si   7.359614  -1.637785  -0.606917
 O   8.141043   0.653245   1.682197
 O   8.143047   1.592535  -0.849222
 H   7.362712   3.003378   1.113678
 O   8.140656  -0.653244  -1.682457
 O   8.143502  -1.592551   0.848985
 H   7.362515  -3.003366  -1.113684
 B   8.471016  -0.668631   1.802860
 H   9.063238  -1.026563   2.769400
 B   8.469853   0.668799  -1.803511
 H   9.060798   1.027005  -2.770727






