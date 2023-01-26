%nproc=15
%mem=70GB
%chk=ExtendedTrisPhenyl_Nitrile_Triazine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    2.406280000000000    -1.304070000000000    -0.188810000000000
 C    1.203390000000000    -0.548970000000000    -0.106440000000000
 C    -0.027230000000000    -1.203660000000000    -0.047250000000000
 H    -0.068540000000000    -2.290310000000000    -0.082140000000000
 C    -1.205400000000000    -0.462500000000000    0.059980000000000
 C    -2.456090000000000    -1.145820000000000    0.109240000000000
 C    -1.153500000000000    0.928000000000000    0.109830000000000
 H    -2.069380000000000    1.508030000000000    0.195200000000000
 C    0.083700000000000    1.572160000000000    0.044410000000000
 C    0.148000000000000    2.991420000000000    0.079300000000000
 C    1.263040000000000    0.839400000000000    -0.064700000000000
 H    2.222880000000000    1.348280000000000    -0.114060000000000
 C    0.175490000000000    4.190400000000000    0.090480000000000
 C    0.127670000000000    5.611430000000000    0.085080000000000
 C    -3.456150000000000    -1.808590000000000    0.128860000000000
 C    -4.564140000000000    -2.707110000000000    0.127640000000000
 C    3.394430000000000    -1.983070000000000    -0.219750000000000
 C    4.502070000000000    -2.877030000000000    -0.213360000000000
 C    -1.117180000000000    6.242170000000000    0.077500000000000
 C    -1.189510000000000    7.632000000000000    0.064410000000000
 H    -2.030180000000000    5.651340000000000    0.079790000000000
 C    -0.016980000000000    8.387359999999999    0.058750000000000
 H    -2.159150000000000    8.121460000000001    0.057590000000000
 C    1.227970000000000    7.755110000000000    0.065810000000000
 C    1.302670000000000    6.362820000000000    0.078900000000000
 H    2.139400000000000    8.346740000000000    0.060060000000000
 H    2.273040000000000    5.872360000000000    0.082380000000000
 C    4.252440000000000    -4.250540000000001    -0.198640000000000
 C    5.314610000000000    -5.149730000000000    -0.169460000000000
 H    3.229840000000000    -4.620520000000000    -0.204530000000000
 C    6.625220000000000    -4.673790000000000    -0.155500000000000
 H    5.116950000000000    -6.217720000000000    -0.154560000000000
 C    6.874820000000000    -3.299950000000000    -0.171320000000000
 C    5.811540000000000    -2.397060000000000    -0.200240000000000
 H    7.897730000000000    -2.933230000000000    -0.157900000000000
 H    6.009820000000000    -1.328100000000000    -0.207300000000000
 C    -4.315700000000000    -4.082170000000000    0.120760000000000
 C    -5.377930000000000    -4.983700000000000    0.105750000000000
 H    -3.293760000000000    -4.454150000000001    0.124030000000000
 C    -6.689040000000000    -4.509870000000000    0.098020000000000
 H    -5.180530000000000    -6.052100000000000    0.098250000000000
 C    -6.938440000000000    -3.136120000000000    0.105990000000000
 C    -5.875190000000000    -2.231340000000000    0.120820000000000
 H    -7.961900000000000    -2.770510000000000    0.098710000000000
 H    -6.075870000000000    -1.163000000000000    0.123970000000000
 C    -0.018249941000000    9.867201649000000    0.063931758000000
 N    -1.203077274000000    10.502893418999999    0.067544588000000
 C    -1.138067693000000    11.830639533999999    0.072116307000000
 N    1.165568420000000    10.504893657000002    0.064777663000000
 C    1.098351920000000    11.832477639000000    0.069503665000000
 N    -0.020482625000000    12.562092020000000    0.073367992000000
 H    2.040605037000000    12.373451380000001    0.070293900000000
 H    -2.081248790000000    12.369987058000000    0.075108885000000
 C    7.837139141000000    -5.522913207000000    -0.169696563000000
 N    9.038103273000001    -4.918330463000000    -0.179583375000000
 C    10.087461264000000    -5.734325440000000    -0.192109234000000
 N    7.678966945000000    -6.858223605000000    -0.172025434000000
 C    8.804155366000000    -7.565897970000000    -0.184972720000000
 N    10.044078814000001    -7.069288667000000    -0.195549196000000
 H    8.705601098000001    -8.647922220000000    -0.187146759000000
 H    11.070747205000000    -5.272181551000000    -0.200299043000000
 C    -7.917701126000000    -5.334655333000000    0.107006723000000
 N    -7.786266976000000    -6.672795339000000    0.113273763000000
 C    -8.925470107000001    -7.357857567000000    0.121202421000000
 N    -9.106428248000000    -4.706162367000000    0.108472466000000
 C    -10.171893431999999    -5.500980825000000    0.116668843000000
 N    -10.155219518999999    -6.836578020000000    0.123371982000000
 H    -11.145805958000000    -5.019330885000000    0.118038305000000
 H    -8.848510311000000    -8.441622085000001    0.126393493000000




