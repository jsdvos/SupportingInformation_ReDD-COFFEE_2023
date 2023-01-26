%nproc=15
%mem=70GB
%chk=ExtendedTrisPhenyl_PrimaryAmine_Imide.chk
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
 C    0.640361263000000    12.003600169000000    -0.118439418000000
 C    1.342657648000000    13.178712506000000    -0.314190126000000
 H    2.390303523000000    13.165144332000001    -0.598241358000000
 C    0.649334155000000    14.379382435000000    -0.133223721000000
 H    1.164474677000000    15.327513406000000    -0.277947135000000
 C    -0.710972066000000    14.376748848000000    0.234471601000000
 H    -1.232089154000000    15.322869186000000    0.370675709000000
 C    -1.397009601000000    13.173394113000001    0.425076050000000
 H    -2.444865876000000    13.155758566999999    0.708153176000000
 C    -0.687798401000000    12.001016000000000    0.238063608000000
 C    -1.113444900000000    10.610831761000000    0.356976973000000
 C    1.072941636000000    10.615067299000000    -0.231627226000000
 N    -0.018836359000000    9.819034899000000    0.062178143000000
 O    2.219189542000000    10.351412573999999    -0.549839625000000
 O    -2.257558540000000    10.342724703000000    0.679003937000000
 C    9.204824516000000    -7.288594600000000    -0.377139413000000
 C    9.760508003000000    -8.537738382000001    -0.585185636000000
 H    9.143796611000001    -9.386329730000000    -0.864454071000000
 C    11.143844401999999    -8.660724559000000    -0.422983935000000
 H    11.621955410000000    -9.626484376000001    -0.577854215000000
 C    11.928019034000000    -7.547240554000000    -0.061233581000000
 H    13.003507912000000    -7.664768344000000    0.060294865000000
 C    11.339813187000001    -6.295159725000000    0.142107637000000
 H    11.931095049000000    -5.428466349000000    0.420809622000000
 C    9.970424054999999    -6.201407050000000    -0.026440458000000
 C    9.078777315000000    -5.054845397000000    0.108800135000000
 C    7.818426510000000    -6.844531608000000    -0.470249126000000
 N    7.798032559000000    -5.494755378000000    -0.170822034000000
 O    6.939944563000000    -7.630024561000000    -0.779970764000000
 O    9.520883077000001    -3.965626888000000    0.429306478000000
 C    -10.059426268999999    -5.977113825000000    -0.070439331000000
 C    -11.426903405999999    -6.048217672000000    -0.263773440000000
 H    -11.999269016999998    -5.170942907000000    -0.548760581000000
 C    -12.038123639000000    -7.292150609000000    -0.079124426000000
 H    -13.112772390000000    -7.392277396000000    -0.221853726000000
 C    -11.278119065000000    -8.419969857000000    0.289734069000000
 H    -11.773804523000001    -9.379245500000000    0.428818359000000
 C    -9.896238522000001    -8.319649122000000    0.477835342000000
 H    -9.297834954000001    -9.179731478000001    0.761768354000000
 C    -9.317375506000001    -7.078278851000000    0.287199356000000
 C    -7.925350704000000    -6.657795053000000    0.402905359000000
 C    -9.146885502000000    -4.845075431000000    -0.187568380000000
 N    -7.877363136000000    -5.308334646000000    0.105129808000000
 O    -9.566434933000000    -3.746751765000000    -0.507444304000000
 O    -7.065222544000000    -7.458336752000000    0.725227119000000




