%nproc=15
%mem=70GB
%chk=P_PrimaryAmine_Imine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    3.834450000000000    2.028850000000000    0.170710000000000
 C    4.206330000000000    0.766460000000000    -0.028510000000000
 C    3.061750000000000    -0.118750000000000    -0.027720000000000
 C    2.370940000000000    1.922770000000000    0.075610000000000
 N    1.928970000000000    0.690610000000000    -0.011520000000000
 C    1.610010000000000    3.228620000000000    0.050410000000000
 C    0.261520000000000    3.197680000000000    0.091160000000000
 N    -0.610890000000000    2.132640000000000    0.135180000000000
 C    -1.936010000000000    2.503710000000000    0.103780000000000
 C    -0.610970000000000    4.367110000000000    0.089560000000000
 C    -1.888450000000000    3.964100000000000    0.143740000000000
 C    -3.061790000000000    1.761380000000000    0.025080000000000
 C    -2.951580000000000    0.254180000000000    -0.012510000000000
 N    -1.879610000000000    -0.503420000000000    -0.000070000000000
 C    -3.782560000000000    -1.754340000000000    0.057960000000000
 C    -4.212200000000000    -0.502070000000000    -0.047320000000000
 C    -0.176010000000000    -3.028340000000000    -0.060020000000000
 N    0.671370000000000    -1.928970000000000    -0.029400000000000
 C    1.995920000000000    -2.349370000000000    -0.010440000000000
 C    1.965000000000000    -3.733700000000000    0.015910000000000
 C    0.639340000000000    -4.146480000000000    -0.126480000000000
 C    3.147630000000000    -1.476470000000000    -0.015980000000000
 C    -1.620780000000000    -2.974640000000000    -0.017310000000000
 C    -2.336100000000000    -1.819480000000000    0.013870000000000
 H    4.463900000000001    2.885230000000000    0.286010000000000
 H    5.237850000000000    0.476960000000000    -0.118110000000000
 C    2.358930000000000    4.521340000000000    -0.050440000000000
 H    -0.306820000000000    1.166360000000000    0.052270000000000
 H    -0.301730000000000    5.398590000000000    0.063860000000000
 H    -2.734880000000000    4.629900000000000    0.185090000000000
 C    -4.397420000000000    2.434930000000000    -0.006460000000000
 H    -4.450990000000000    -2.596530000000000    0.127720000000000
 H    -5.226140000000000    -0.160350000000000    -0.095230000000000
 H    0.368460000000000    -0.960990000000000    -0.016620000000000
 H    2.820260000000000    -4.392510000000000    0.077770000000000
 H    0.310690000000000    -5.174130000000000    -0.203890000000000
 C    4.531580000000000    -2.102480000000000    -0.035610000000000
 C    -2.374400000000000    -4.291369999999999    0.032450000000000
 C    -4.741190000000000    3.311750000000000    -1.050370000000000
 C    -5.977000000000000    3.965120000000000    -1.062590000000000
 H    -4.031900000000000    3.519110000000000    -1.850140000000000
 C    -6.911260000000000    3.708380000000000    -0.062580000000000
 H    -6.208570000000000    4.674480000000000    -1.853500000000000
 C    -6.599060000000000    2.823450000000000    0.966560000000000
 C    -5.340520000000000    2.215570000000000    1.011330000000000
 H    -7.333480000000000    2.610180000000000    1.739570000000000
 H    -5.112220000000000    1.548090000000000    1.840350000000000
 C    3.256720000000000    4.734290000000000    -1.112880000000000
 C    4.070410000000000    5.869650000000000    -1.151690000000000
 H    3.370520000000000    3.985610000000000    -1.895730000000000
 C    3.946060000000000    6.847820000000000    -0.169600000000000
 H    4.807020000000000    5.983140000000000    -1.943480000000000
 C    3.020070000000000    6.683940000000000    0.858240000000000
 C    2.251750000000000    5.517520000000001    0.933110000000000
 H    2.906760000000000    7.458130000000000    1.613260000000000
 H    1.565100000000000    5.396900000000000    1.768870000000000
 C    4.872220000000000    -3.027410000000000    -1.039010000000000
 C    6.122890000000000    -3.651820000000000    -1.053390000000000
 H    4.145120000000000    -3.295470000000000    -1.803690000000000
 C    7.080580000000000    -3.314800000000000    -0.101320000000000
 H    6.344820000000000    -4.401200000000000    -1.808820000000000
 C    6.772130000000000    -2.389760000000000    0.892460000000000
 C    5.494670000000000    -1.823420000000000    0.950700000000000
 H    7.522030000000000    -2.113930000000000    1.629310000000000
 H    5.266900000000000    -1.133430000000000    1.760630000000000
 C    -3.297960000000000    -4.666250000000000    -0.957890000000000
 C    -3.944920000000000    -5.905850000000000    -0.903880000000000
 H    -3.528960000000000    -3.989670000000000    -1.777670000000000
 C    -3.627980000000000    -6.818279999999999    0.100320000000000
 H    -4.684150000000000    -6.165690000000000    -1.657210000000000
 C    -2.689640000000000    -6.480600000000001    1.070970000000000
 C    -2.091670000000000    -5.217730000000000    1.053630000000000
 H    -2.411560000000000    -7.200630000000000    1.836380000000000
 H    -1.358000000000000    -4.974810000000000    1.820860000000000
 N    -8.247361391000000    4.174297489000000    -0.048772348000000
 C    -8.386540919000000    5.426224591000000    -0.257919456000000
 C    -9.697015405000000    6.080633655000000    -0.338333836000000
 C    -9.758806371000000    7.465615877000000    -0.543806911000000
 C    -10.988064969000000    8.115097653999999    -0.627302510000000
 C    -12.168275638000001    7.384272414000000    -0.505519325000000
 C    -12.115824622000000    6.002088560000000    -0.299702430000000
 C    -10.891697011000000    5.352595534000000    -0.216935116000000
 H    -7.516209994000000    6.087333592000000    -0.374058860000000
 H    -8.838857634999998    8.034088281000001    -0.637963700000000
 H    -11.024952891000000    9.186680175999999    -0.786521432000000
 H    -13.126818110000000    7.886860781000000    -0.569985923000000
 H    -13.034926110000001    5.434808918000000    -0.205099256000000
 H    -10.835131806000000    4.282695730000000    -0.058109130000000
 N    4.523521731000000    7.981032920000000    -0.179048630000000
 C    5.788270953000000    8.054543914000000    -0.338549290000000
 C    6.506633741000000    9.329570971000001    -0.440146456000000
 C    7.900364487000000    9.319920573999999    -0.587233303000000
 C    8.610102611000000    10.513982408000000    -0.689031755000000
 C    7.931385881000000    11.730193430000000    -0.644218234000000
 C    6.540874704000000    11.749052843999999    -0.497021430000000
 C    5.831398135000000    10.560083915000000    -0.396145727000000
 H    6.411810472000000    7.151044589000000    -0.392556972000000
 H    8.428189597999999    8.371995407000000    -0.621447087000000
 H    9.687948280000001    10.495605474000000    -0.802660484000000
 H    8.480941407000000    12.661491945000000    -0.723126285000000
 H    6.014181309000000    12.696095845000000    -0.462277768000000
 H    4.754318866000000    10.558727390000000    -0.282087335000000
 N    8.307433960999999    -3.979063508000000    -0.098248976000000
 C    8.469312776000001    -5.236025743000000    -0.254056911000000
 C    9.791651549999997    -5.865013678000000    -0.342687725000000
 C    9.879342964999999    -7.256358264000000    -0.486505396000000
 C    11.120319801999999    -7.882320846000000    -0.576071728000000
 C    12.286441485999999    -7.121341882000000    -0.522172422000000
 C    12.208157627000000    -5.732574519000000    -0.378155907000000
 C    10.972287675000000    -5.106523622000000    -0.289466775000000
 H    7.611429210999999    -5.920429559000000    -0.314526711000000
 H    8.970370696000000    -7.848300766000000    -0.527803565000000
 H    11.177247711000000    -8.959072904999999    -0.687268416000000
 H    13.254051148000000    -7.605566012000000    -0.591549759000000
 H    13.116323068000000    -5.141820873000000    -0.336332938000000
 H    10.895722911000000    -4.031894993000000    -0.178012507000000
 N    -4.284172125000000    -8.199138652000000    0.140486570000000
 C    -5.545728215000000    -8.321926044000000    -0.013897397000000
 C    -6.220145275000000    -9.623714980999999    -0.064575467000000
 C    -7.613444438000000    -9.667370601000000    -0.209574410000000
 C    -8.282008196000000    -10.888016496000001    -0.263619695000000
 C    -7.562118740000000    -12.077687333000000    -0.172645140000000
 C    -6.171722593000000    -12.043333960000000    -0.027172436000000
 C    -5.503246561000000    -10.827819010000001    0.026149481000000
 H    -6.199791859000000    -7.443022422000000    -0.101980106000000
 H    -8.173360349999999    -8.740063240000000    -0.279769015000000
 H    -9.359892936000000    -10.910873842999997    -0.376058641000000
 H    -8.079562826000000    -13.029564357000000    -0.214315628000000
 H    -5.612967686000000    -12.969778314999999    0.043519217000000
 H    -4.426801834000000    -10.785264102999999    0.138250943000000



