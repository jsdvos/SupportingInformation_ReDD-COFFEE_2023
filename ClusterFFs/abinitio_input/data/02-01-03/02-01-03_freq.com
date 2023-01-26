%nproc=15
%mem=70GB
%chk=F_BoronicAcid_Borosilicate_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from F_BoronicAcid_Borosilicate.log

0 1
 F  -1.322444  -2.372799   0.004095
 C  -0.694420  -1.183316  -0.007026
 C   0.694405  -1.183351   0.005705
 F   1.322363  -2.372850  -0.005429
 C   1.433907  -0.000025  -0.000657
 C   0.694420   1.183305  -0.007028
 F   1.322444   2.372787   0.004090
 C  -0.694405   1.183339   0.005706
 F  -1.322364   2.372837  -0.005425
 C  -1.433907   0.000013  -0.000653
 B   3.010071  -0.000024  -0.000484
 O   3.664482  -0.965454   0.705206
 O   3.664785   0.965213  -0.706123
Si   5.222299  -1.436311   1.005102
Si   5.222760   1.436347  -1.004602
 O   5.982055  -1.752348  -0.427109
 O   6.009266  -0.215195   1.793397
 H   5.216426  -2.631767   1.834625
 O   5.981339   1.752324   0.428209
 O   6.010595   0.215299  -1.792149
 H   5.217446   2.631873  -1.834023
 B   6.327330  -1.098508  -1.578393
 H   6.921133  -1.690996  -2.419782
 B   6.326341   1.098550   1.579578
 H   6.919996   1.690992   2.421099
 B  -3.010071   0.000014  -0.000474
 O  -3.664478   0.965448   0.705213
 O  -3.664789  -0.965218  -0.706118
Si  -5.222293   1.436314   1.005103
Si  -5.222765  -1.436341  -1.004604
 O  -5.982043   1.752357  -0.427110
 O  -6.009270   0.215203   1.793395
 H  -5.216416   2.631770   1.834627
 O  -5.981350  -1.752316   0.428205
 O  -6.010591  -0.215288  -1.792152
 H  -5.217456  -2.631866  -1.834027
 B  -6.327320   1.098521  -1.578395
 H  -6.921115   1.691014  -2.419786
 B  -6.326352  -1.098541   1.579573
 H  -6.920015  -1.690980   2.421091






