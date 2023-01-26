%nproc=15
%mem=70GB
%chk=Pyrene_Aldehyde_Hydrazone_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Pyrene_Aldehyde_Hydrazone.log

0 1
 C   0.044392   3.528016  -0.013947
 C  -1.169009   2.837269   0.092032
 C  -1.212566   1.438873   0.099729
 C   0.008023   0.710463  -0.002789
 C   1.246555   1.414519  -0.110875
 C   1.244565   2.810236  -0.114792
 H  -2.098834   3.392877   0.170411
 H   2.177425   3.355645  -0.195388
 C  -0.007889  -0.710473   0.002787
 C  -1.246420  -1.414529   0.110874
 C  -1.244430  -2.810246   0.114791
 C  -0.044257  -3.528026   0.013946
 C   1.169144  -2.837279  -0.092033
 C   1.212701  -1.438883  -0.099731
 H  -2.177290  -3.355655   0.195387
 H   2.098969  -3.392887  -0.170412
 C   2.461864   0.651953  -0.212719
 C   2.445207  -0.706441  -0.207367
 H   3.400127   1.189908  -0.294406
 H   3.372030  -1.265188  -0.285128
 C  -2.445072   0.706431   0.207366
 C  -2.461729  -0.651963   0.212718
 H  -3.371895   1.265179   0.285127
 H  -3.399992  -1.189917   0.294405
 H  -0.359147   11.617047  -1.409553
 C   0.493863   11.215605  -0.874461
 C   1.387679   12.074501  -0.237476
 H   1.223306   13.145751  -0.265006
 C   2.500953   11.554061   0.424137
 H   3.202368   12.220486   0.912993
 C   2.717047   10.180595   0.452806
 H   3.585553   9.760315   0.944940
 C   1.811254   9.310426  -0.163895
 C   2.117082   7.840695  -0.103008
 C   0.702095   9.838200  -0.836420
 H   0.018636   9.185474  -1.368413
 O   3.248138   7.408886  -0.030992
 N   0.990513   7.027711  -0.118086
 H   0.073749   7.451692  -0.006008
 N   1.104880   5.678128  -0.113273
 C   0.028915   4.991733  -0.016112
 H  -0.954220   5.474659   0.069112
 H   0.359283  -11.617056   1.409554
 C  -0.493726  -11.215615   0.874463
 C  -1.387543  -12.074510   0.237478
 H  -1.223169  -13.145761   0.265008
 C  -2.500817  -11.554071  -0.424135
 H  -3.202232  -12.220496  -0.912990
 C  -2.716911  -10.180605  -0.452804
 H  -3.585418  -9.760325  -0.944939
 C  -1.811118  -9.310435   0.163896
 C  -2.116946  -7.840705   0.103009
 C  -0.701959  -9.838209   0.836421
 H  -0.018500  -9.185483   1.368414
 O  -3.248003  -7.408896   0.030993
 N  -0.990378  -7.027721   0.118086
 H  -0.073614  -7.451702   0.006008
 N  -1.104745  -5.678137   0.113273
 C  -0.028780  -4.991743   0.016111
 H   0.954355  -5.474669  -0.069113






