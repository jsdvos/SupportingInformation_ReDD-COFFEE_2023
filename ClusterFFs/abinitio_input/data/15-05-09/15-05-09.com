%nproc=15
%mem=70GB
%chk=ExtendedTrisPhenyl_Ketoenamine_Ketoenamine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

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
 C    0.075642957000000    9.883146014999999    0.040556657000000
 O    1.176511598000000    10.429405619000001    0.071245169000000
 C    -1.175905986000000    10.690618899000000    -0.029335955000000
 C    -1.178799300000000    12.031185911000001    -0.002913545000000
 H    -2.127059421999999    10.188353240000000    -0.117279271000000
 N    -0.110992875000000    12.883601751000000    0.079507003000000
 H    -2.142305517000000    12.527091483000000    -0.061806267000000
 C    -0.088301446000000    14.287880084999999    0.103093927000000
 H    0.797730634000000    12.412027112000001    0.105905985000000
 C    -1.242181320000000    15.082853449000000    0.113141865000000
 C    1.152752198000000    14.943491853999999    0.106673657000000
 C    -1.161845554000000    16.480085586000001    0.127821339000000
 H    -2.230284789000000    14.636340396000000    0.115025846000000
 C    0.078925109000000    17.108823821000001    0.131116693000000
 H    -2.072652065000000    17.072899820000000    0.137045240000000
 C    1.238414452000000    16.339975250999998    0.120542053000000
 H    2.073393348000000    14.364612468000001    0.095932340000000
 H    0.141133854000000    18.193492753000001    0.141954157000000
 H    2.211687608000000    16.823178187000000    0.122093888000000
 C    7.795939588000000    -5.608832834000000    -0.192867858000000
 O    7.610985495000000    -6.823839678000000    -0.164522527000000
 C    9.174849564000001    -5.048033167000000    -0.278793428000000
 C    10.273998096000000    -5.815900648000000    -0.269935242000000
 H    9.309043429999999    -3.980617299000000    -0.364309782000000
 N    10.359283841000000    -7.179931053000000    -0.194015104000000
 H    11.232588492000000    -5.311919801000000    -0.339524587000000
 C    11.495839128000000    -8.005329661999999    -0.188714362000000
 H    9.451647950000000    -7.652921352000000    -0.157474311000000
 C    12.809467770999998    -7.517607426000000    -0.194116561000000
 C    11.319488389000000    -9.397792944000001    -0.188294139000000
 C    13.907023108000001    -8.386105329999999    -0.197378279000000
 H    13.011705279999999    -6.452330806000000    -0.190705487000000
 C    13.708837535000001    -9.762897510000000    -0.196891847000000
 H    14.915550777000000    -7.981200525000000    -0.199895805000000
 C    12.413360266000002    -10.270217283999999    -0.192333781000000
 H    10.316695315000000    -9.818746811000000    -0.187433326000000
 H    14.560949983000000    -10.436981253000001    -0.199982727000000
 H    12.249715360000000    -11.344446402000001    -0.192854473000000
 C    -7.983250621000000    -5.265579276000000    0.083591845000000
 O    -9.049797409000000    -4.655037659000000    0.114736485000000
 C    -7.957504895000000    -6.754926528000000    0.016852723000000
 C    -9.069524970000000    -7.503488833000000    0.046723417000000
 H    -7.010939077000000    -7.265703982000000    -0.071562338000000
 N    -10.371921076000000    -7.090721929000000    0.130415336000000
 H    -8.945264438000001    -8.580089661000001    -0.010064750000000
 C    -11.551115570000000    -7.853500167000000    0.157592292000000
 H    -10.485953789000000    -6.073245180000000    0.154819595000000
 C    -11.569220501000000    -9.254579527000001    0.170677198000000
 C    -12.786584722000001    -7.187428011000000    0.161788630000000
 C    -12.774663564000001    -9.965566644999999    0.188878990000000
 H    -10.648254271000001    -9.826894657000002    0.172264231000000
 C    -13.987650512000000    -9.284771257999999    0.192722118000000
 H    -12.760132005000001    -11.052184401000000    0.200410920000000
 C    -13.994372077000000    -7.893575110000000    0.179172835000000
 H    -12.818169554000001    -6.100400103000001    0.148766640000000
 H    -14.923348125000000    -9.836839955000000    0.206294088000000
 H    -14.937557471000000    -7.353996004000000    0.181128685000000



