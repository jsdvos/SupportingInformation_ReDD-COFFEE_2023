%nproc=15
%mem=70GB
%chk=BiPhenyl_Aldehyde_Azine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 C    -0.000000000000000    3.577958000000000    -0.000000182000000
 C    1.133739000000000    2.852350000000000    -0.402520182000000
 C    1.135339000000000    1.463226000000000    -0.401762182000000
 C    -0.000000000000000    0.741600000000000    -0.000000182000000
 C    -1.135339000000000    1.463225000000000    0.401761818000000
 C    -1.133738000000000    2.852349000000000    0.402519818000000
 C    -0.000000000000000    -0.741600000000000    -0.000000182000000
 C    -1.135339000000000    -1.463226000000000    -0.401762182000000
 C    -1.133739000000000    -2.852350000000000    -0.402520182000000
 C    -0.000000000000000    -3.577958000000000    -0.000000182000000
 C    1.133738000000000    -2.852349000000000    0.402519818000000
 C    1.135339000000000    -1.463225000000000    0.401761818000000
 H    -2.020512000000000    -3.385811000000000    -0.727052182000000
 H    -2.020512000000000    3.385812000000000    0.727052818000000
 H    2.020512000000000    -3.385812000000000    0.727052818000000
 H    2.020512000000000    3.385811000000000    -0.727052182000000
 H    -2.015826000000000    -0.929067000000000    -0.740072182000000
 H    -2.015827000000000    0.929068000000000    0.740072818000000
 H    2.015826000000000    0.929067000000000    -0.740072182000000
 H    2.015827000000000    -0.929068000000000    0.740072818000000
 C    -0.036535883000000    5.052235064000000    0.018372241000000
 N    0.962002756000000    5.780564404000000    -0.370879375000000
 H    -0.969641235000000    5.521567936000000    0.381033841000000
 N    0.646571344000000    7.133472540000000    -0.265725800000000
 C    1.635537356000000    7.871347307000000    -0.659860585000000
 C    1.559317471000000    9.344822219999999    -0.675517974000000
 H    2.580290283000000    7.415075274000000    -1.008203118000000
 C    2.664720551000000    10.071088563000000    -1.128559899000000
 C    2.608954184000000    11.465752101000000    -1.167440407000000
 H    3.568495903000000    9.563792352000000    -1.455234768000000
 C    1.453723213000000    12.131032909000000    -0.754563911000000
 H    3.464951748000000    12.034822842000001    -1.521685360000000
 C    0.352090115000000    11.405323680000000    -0.299881172000000
 H    1.411072076000000    13.217098008000001    -0.788013220000000
 C    0.402619618000000    10.010895261000000    -0.259221946000000
 H    -0.547393327000000    11.924827860000001    0.020884362000000
 H    -0.460111667000000    9.450799508999999    0.093199122000000
 C    0.036535888000000    -5.052237063000000    0.018372488000000
 N    -0.962002857000000    -5.780566423000000    -0.370878820000000
 H    0.969641339000000    -5.521569916000001    0.381033859000000
 N    -0.646571416000000    -7.133474553000000    -0.265725262000000
 C    -1.635537535000000    -7.871349340000000    -0.659859741000000
 C    -1.559317655000000    -9.344824255000001    -0.675517076000000
 H    -2.580290557000000    -7.415077325000000    -1.008202041000000
 C    -2.664720857000000    -10.071090621000000    -1.128558664000000
 C    -2.608954500000000    -11.465754159999999    -1.167439116000000
 H    -3.568496298000000    -9.563794425999999    -1.455233314000000
 C    -1.453723418000000    -12.131034948000000    -0.754562901000000
 H    -3.464952161000000    -12.034824919000000    -1.521683809000000
 C    -0.352090196000000    -11.405325696000000    -0.299880497000000
 H    -1.411072289000000    -13.217100048000001    -0.788012166000000
 C    -0.402619688000000    -10.010897274000000    -0.259221328000000
 H    0.547393333000000    -11.924829859000001    0.020884819000000
 H    0.460111692000000    -9.450801504999999    0.093199478000000



