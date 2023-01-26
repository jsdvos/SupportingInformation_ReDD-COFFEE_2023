%nproc=15
%mem=70GB
%chk=BiPhenyl_Aldehyde_Hydrazone.chk
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
 H    -0.759731570000000    11.797813875999999    -0.276376535000000
 C    0.212536028000000    11.359408209000000    -0.489145025000000
 C    1.298833861000000    12.187586970000000    -0.764663026000000
 H    1.174908419000000    13.267721798000000    -0.764036267000000
 C    2.541883309000000    11.630236032999999    -1.052067843000000
 H    3.389925239000000    12.271689933999999    -1.278644040000000
 C    2.699372700000000    10.242694882000000    -1.055859010000000
 H    3.676365286000000    9.820558252000000    -1.287983942000000
 C    1.615685508000000    9.395792990000000    -0.773171415000000
 C    1.858499165000000    7.918008216000000    -0.816929533000000
 C    0.365197459000000    9.968714734000001    -0.498639293000000
 H    -0.517238277000000    9.368400747999999    -0.311090892000000
 O    2.918901271000000    7.502912751000000    -1.279222251000000
 N    0.849423067000000    7.103768798000000    -0.319226787000000
 H    0.023440554000000    7.481715750000000    0.120452270000000
 N    0.990082912000000    5.749328140000000    -0.385197346000000
 C    -0.019308756000000    5.053393049000000    0.027960484000000
 H    -0.940115421000000    5.531674974000000    0.407227526000000
 H    0.759731495000000    -11.797815890000001    -0.276376142000000
 C    -0.212536161000000    -11.359410234000000    -0.489144390000000
 C    -1.298834069000000    -12.187589009000000    -0.764662055000000
 H    -1.174908626000000    -13.267723836000000    -0.764035274000000
 C    -2.541883595000000    -11.630238087000000    -1.052066563000000
 H    -3.389925585000000    -12.271691999000000    -1.278642496000000
 C    -2.699372987000000    -10.242696936000000    -1.055857757000000
 H    -3.676365636000000    -9.820560318000000    -1.287982445000000
 C    -1.615685718000000    -9.395795029000000    -0.773170499000000
 C    -1.858499386000000    -7.918010257000000    -0.816928627000000
 C    -0.365197594000000    -9.968716758999999    -0.498638687000000
 H    0.517238193000000    -9.368402764000001    -0.311090556000000
 O    -2.918901618000000    -7.502914816000000    -1.279221078000000
 N    -0.849423153000000    -7.103770814000000    -0.319226196000000
 H    -0.023440521000000    -7.481717744000000    0.120452656000000
 N    -0.990083016000000    -5.749330158999999    -0.385196785000000
 C    0.019308764000000    -5.053395048000000    0.027960736000000
 H    0.940115531000000    -5.531676953000000    0.407227552000000



