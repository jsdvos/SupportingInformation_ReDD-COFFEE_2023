%nproc=15
%mem=70GB
%chk=BiPhenyl_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from BiPhenyl_BoronicAcid_BoronateEster.log

0 1
 C  -3.572666  -0.000002  -0.000004
 C  -2.849729   1.130603  -0.414359
 C  -1.460700   1.132137  -0.413753
 C  -0.740935  -0.000002  -0.000001
 C  -1.460700  -1.132141   0.413750
 C  -2.849730  -1.130607   0.414353
 C   0.740935  -0.000002   0.000001
 C   1.460701  -1.132141  -0.413749
 C   2.849730  -1.130606  -0.414353
 C   3.572666  -0.000002   0.000004
 C   2.849729   1.130603   0.414360
 C   1.460700   1.132137   0.413753
 H   3.386350  -2.012388  -0.747270
 H  -3.386350  -2.012388   0.747271
 H   3.386348   2.012384   0.747278
 H  -3.386348   2.012384  -0.747278
 H   0.924296  -2.007885  -0.760808
 H  -0.924296  -2.007885   0.760808
 H  -0.924294   2.007881  -0.760810
 H   0.924294   2.007881   0.760810
 B  -5.108595  -0.000001  -0.000006
 O  -5.901704   1.074929  -0.398743
 O  -5.901708  -1.074930   0.398728
 C  -7.204202   0.653434  -0.242475
 C  -7.204204  -0.653432   0.242457
 C  -8.375316   1.340128  -0.497299
 C  -8.375320  -1.340124   0.497277
 C  -9.570936   0.655206  -0.243180
 C  -9.570938  -0.655200   0.243154
 H  -8.361759   2.354873  -0.873898
 H  -8.361766  -2.354869   0.873875
 H  -10.514706   1.154182  -0.428335
 H  -10.514709  -1.154174   0.428306
 B   5.108595  -0.000001   0.000006
 O   5.901704   1.074929   0.398743
 O   5.901707  -1.074930  -0.398728
 C   7.204202   0.653434   0.242475
 C   7.204204  -0.653432  -0.242457
 C   8.375316   1.340127   0.497299
 C   8.375320  -1.340124  -0.497277
 C   9.570936   0.655206   0.243179
 C   9.570938  -0.655200  -0.243154
 H   8.361759   2.354873   0.873897
 H   8.361766  -2.354869  -0.873875
 H   10.514706   1.154182   0.428334
 H   10.514709  -1.154174  -0.428306





