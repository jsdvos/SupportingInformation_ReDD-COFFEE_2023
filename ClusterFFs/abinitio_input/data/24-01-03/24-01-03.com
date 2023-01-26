%nproc=15
%mem=70GB
%chk=P_BoronicAcid_Borosilicate.chk
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
 B    -8.362156619000000    4.338171953000000    -0.030550827000000
 O    -9.258378316000000    4.088287014000000    0.934773946000000
 O    -8.684724351000000    5.208661990000000    -0.998618375000000
Si    -10.781466033999999    4.353600110000000    1.340994715000000
Si    -9.907197333999999    6.066990180000000    -1.567170221000000
 O    -10.957808892999999    5.932723026000000    1.513721462000000
 O    -11.691169644000000    3.914470459000000    0.101525153000000
 H    -11.148165440000000    3.625369036000000    2.563808727000000
 O    -11.120545805000001    5.057804306000000    -1.821338131000000
 O    -10.366684576999999    7.068216604000000    -0.408580670000000
 H    -9.535849815000001    6.786538412000000    -2.793774471000000
 B    -10.801572395999999    7.091605260000000    0.859042044000000
 H    -10.779986590000000    8.104707980000001    1.464453265000000
 B    -11.881727449000000    4.142676055000000    -1.205488795000000
 H    -12.472937376999999    3.351462049000000    -1.851823140000000
 B    4.691730426000000    8.088879634000000    -0.158078903000000
 O    4.444330285000000    9.033475094000000    0.760637765000000
 O    5.616180219000000    8.332444013000000    -1.098837424000000
Si    4.764076393000000    10.557930276000002    1.119805375000000
Si    6.554299864000000    9.490802303000001    -1.675514355000000
 O    6.341166199000000    10.668913173000000    1.354392359000000
 O    4.420587231000000    11.436356322000000    -0.171342634000000
 H    4.003511247000000    11.006079079999999    2.294865531000000
 O    5.615053425000000    10.737725158000000    -2.019420574000000
 O    7.526816010000000    9.950198951000001    -0.492687364000000
 H    7.306138315000000    9.038257847000001    -2.854560191000000
 B    7.517722382000000    10.433958426000000    0.757286982000000
 H    8.502487629999999    10.390484936000000    1.406681528000000
 B    4.711961283000000    11.563851252999999    -1.473464647000000
 H    3.977257537000000    12.164152544000000    -2.175707934000000
 B    8.426418676999999    -4.139404126000000    -0.075925624000000
 O    9.343462405000000    -3.825866414000000    0.850482835000000
 O    8.741498045000000    -5.046557606000000    -1.012277682000000
Si    10.883042716000000    -4.039175660000000    1.223838134000000
Si    9.966842236000000    -5.903668328000000    -1.576468698000000
 O    11.100287191000000    -5.604448764000000    1.463181950000000
 O    11.747252501000000    -3.638290827000000    -0.060339853000000
 H    11.267195609000000    -3.247439333000000    2.400971603000000
 O    11.149137994000000    -4.881358751000000    -1.911628117000000
 O    10.481521915000000    -6.840103191000000    -0.387306304000000
 H    9.577738909000001    -6.687008701000000    -2.757647382000000
 B    10.952290251999999    -6.795463796000000    0.866857276000000
 H    10.970936192000000    -7.779720541000000    1.518207262000000
 B    11.906172951000000    -3.922517737000000    -1.360584017000000
 H    12.460707670000000    -3.149457407000000    -2.059272673000000
 B    -4.448590433000000    -8.311763968999999    0.165922097000000
 O    -4.168733880000000    -9.210856219000000    1.120291609000000
 O    -5.364514547000000    -8.623200741000000    -0.763081551000000
Si    -4.436080628000000    -10.730184156000000    1.539064678000000
Si    -6.262715785000000    -9.834485767000000    -1.292649111000000
 O    -6.008374036000000    -10.885698721000001    1.780454772000000
 O    -4.063240498000000    -11.646011637000001    0.282509639000000
 H    -3.660228808000000    -11.105996710000001    2.729393177000000
 O    -5.281538157000000    -11.061080630999999    -1.589343345000000
 O    -7.218546260000000    -10.280371852000000    -0.091197354000000
 H    -7.029997609000000    -9.454140932000000    -2.487146090000000
 B    -7.192483773000000    -10.714473495000000    1.176642659000000
 H    -8.177927602000000    -10.679364570000001    1.825513709000000
 B    -4.350553096000000    -11.834008810000000    -1.013169359000000
 H    -3.596019057000000    -12.435790796999999    -1.692752828000000




