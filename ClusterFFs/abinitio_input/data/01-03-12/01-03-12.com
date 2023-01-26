%nproc=15
%mem=70GB
%chk=2_Phenyl_PrimaryAmine_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

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
 C    -0.660720165000000    -5.030169543000000    -0.189077173000000
 C    -1.364271186000000    -6.203978055000000    -0.388117613000000
 H    -2.412235875000000    -6.188504273000000    -0.670892521000000
 C    -0.671786121000000    -7.405876775000000    -0.212167373000000
 H    -1.187923554000000    -8.353046832000000    -0.359605487000000
 C    0.688950437000000    -7.405713688000000    0.153941616000000
 H    1.209399404000000    -8.352757978000000    0.286222593000000
 C    1.376261290000000    -6.203630803000000    0.347953736000000
 H    2.424462505000000    -6.187897840000001    0.629863529000000
 C    0.667856754000000    -5.029988728000000    0.165877070000000
 C    1.094856350000000    -3.640600077000000    0.289157540000000
 C    -1.092219076000000    -3.640873188000000    -0.296897012000000
 N    0.000596457000000    -2.846824199000000    -0.001585681000000
 O    -2.238607083000000    -3.375109118000000    -0.612841946000000
 O    2.239579168000000    -3.374617327000000    0.610780963000000
 C    0.660720165000000    5.030169543000000    0.189077173000000
 C    1.364271186000000    6.203978055000000    0.388117613000000
 H    2.412235875000000    6.188504273000000    0.670892521000000
 C    0.671786121000000    7.405876775000000    0.212167373000000
 H    1.187923554000000    8.353046832000000    0.359605487000000
 C    -0.688950437000000    7.405713688000000    -0.153941616000000
 H    -1.209399404000000    8.352757978000000    -0.286222593000000
 C    -1.376261290000000    6.203630803000000    -0.347953736000000
 H    -2.424462505000000    6.187897840000001    -0.629863529000000
 C    -0.667856754000000    5.029988728000000    -0.165877070000000
 C    -1.094856350000000    3.640600077000000    -0.289157540000000
 C    1.092219076000000    3.640873188000000    0.296897012000000
 N    -0.000596457000000    2.846824199000000    0.001585681000000
 O    2.238607083000000    3.375109118000000    0.612841946000000
 O    -2.239579168000000    3.374617327000000    -0.610780963000000



