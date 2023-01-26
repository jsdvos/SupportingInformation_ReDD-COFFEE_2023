%nproc=15
%mem=70GB
%chk=2_Phenyl_Nitrile_Triazine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    -0.000013000000000    -1.415145000000000    0.000000000000000
 C    0.000013000000000    1.415145000000000    0.000000000000000
 C    -1.206999000000000    -0.694755000000000    0.000000000000000
 C    1.206999000000000    0.694755000000000    0.000000000000000
 C    -1.206984000000000    0.694770000000000    0.000000000000000
 C    1.206984000000000    -0.694770000000000    0.000000000000000
 H    2.148535000000000    -1.233078000000000    0.000000000000000
 H    -2.148535000000000    1.233078000000000    0.000000000000000
 H    2.148565000000000    1.233036000000000    0.000000000000000
 H    -2.148565000000000    -1.233036000000000    0.000000000000000
 C    -0.000029998000000    -2.894996266000000    -0.000000000000000
 N    1.184244868000000    -3.531726923000000    -0.000000000000000
 C    1.118080598000000    -4.859423870000000    -0.000000000000000
 N    -1.184403286000000    -3.531657589000000    0.000000000000000
 C    -1.118341294000000    -4.859307932000000    0.000000000000000
 N    -0.000140915000000    -5.589904080000000    -0.000000000000000
 H    -2.061065153000000    -5.399461498000000    0.000000000000000
 H    2.060792940000000    -5.399598590000000    -0.000000000000000
 C    0.000029998000000    2.894996266000000    -0.000000000000000
 N    -1.184244868000000    3.531726923000000    -0.000000000000000
 C    -1.118080598000000    4.859423870000000    -0.000000000000000
 N    1.184403286000000    3.531657589000000    0.000000000000000
 C    1.118341294000000    4.859307932000000    0.000000000000000
 N    0.000140915000000    5.589904080000000    -0.000000000000000
 H    2.061065153000000    5.399461498000000    0.000000000000000
 H    -2.060792940000000    5.399598590000000    -0.000000000000000



