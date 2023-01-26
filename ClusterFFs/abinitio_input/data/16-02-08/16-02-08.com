%nproc=15
%mem=70GB
%chk=N-TrisPhenyl_Aldehyde_Benzobisoxazole.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 N    0.000000000000000    0.000000000000000    0.000000000000000
 C    1.486664194000000    0.115830323000000    -0.171934194000000
 C    -0.789065806000000    1.237760323000000    0.199745806000000
 C    -0.613895806000000    -1.236089677000000    -0.038794194000000
 C    -0.462675806000000    2.500160323000000    -0.322134194000000
 C    2.176224194000000    -0.918259677000000    -0.846584194000000
 C    -1.765025806000000    -1.441179677000000    -0.822204194000000
 C    -2.007925806000000    1.140170323000000    0.910585806000000
 C    -1.297715806000000    3.610210323000000    -0.126194194000000
 H    0.438864194000000    2.642570323000000    -0.911754194000000
 C    2.273194194000000    1.157440323000000    0.348705806000000
 C    3.570374194000000    -0.927389677000000    -0.965384194000000
 H    1.632104194000000    -1.749019677000000    -1.289124194000000
 C    -0.125815806000000    -2.315259677000000    0.720105806000000
 C    -2.414125806000000    -2.677449677000000    -0.833324194000000
 H    -2.163745806000000    -0.629609677000000    -1.426834194000000
 C    -2.865415806000000    2.233870323000000    1.065575806000000
 H    -2.315445806000000    0.194440323000000    1.350765806000000
 C    -2.504725806000000    3.472930323000000    0.555025806000000
 H    -1.010125806000000    4.580640323000000    -0.521804194000000
 C    3.666684194000001    1.177220323000000    0.186235806000000
 H    1.821684194000000    1.969990323000000    0.911145806000000
 C    4.315594194000000    0.127230323000000    -0.457594194000000
 H    4.064904194000000    -1.761929677000000    -1.456734194000000
 C    -0.759375806000000    -3.559559677000000    0.685245806000000
 H    0.753504194000000    -2.187099677000000    1.347095806000000
 C    -1.906875806000000    -3.737209677000000    -0.084874194000000
 H    -3.314085806000000    -2.810259677000000    -1.427784194000000
 H    -3.812175806000000    2.108150323000000    1.585245806000000
 H    4.245704194000000    2.009030323000000    0.578395806000000
 H    -0.358405806000000    -4.386819677000000    1.264815806000000
 H    -7.340901787000000    8.262660708000000    1.828984541000000
 C    -6.417427347000000    7.800587559000000    1.484071395000000
 C    -5.464629480000000    8.583423302000000    0.818727706000000
 H    -5.658672054000000    9.641720284000000    0.654725407000000
 C    -4.261662537000000    8.026294376999999    0.357865030000000
 H    -3.522096003000001    8.628280251000000    -0.157937403000000
 C    -4.038747269000000    6.661435368999999    0.579823858000000
 N    -2.981510148000000    5.853655633000001    0.248339054000000
 C    -5.005928559000000    5.923143405000000    1.240563571000000
 O    -4.561879511000000    4.637060485000000    1.328657857000000
 C    -6.207575464000000    6.435128502000000    1.714220803000000
 H    -6.931819467000000    5.813951968000000    2.225388579000000
 C    -3.336330338000000    4.664998630000000    0.707510241000000
 H    11.099717095000001    1.465381310000000    -0.829231836000000
 C    10.146772160999999    0.948709420000000    -0.930364553000000
 C    10.110736026000000    -0.312626590000000    -1.539520251000000
 H    11.033697677999998    -0.759822744000000    -1.903799217000000
 C    8.902933504000000    -1.011597244000000    -1.689752093000000
 H    8.870536697000000    -1.987421635000000    -2.161042879000000
 C    7.729815849000000    -0.411818007000000    -1.214508066000000
 N    6.428028878000000    -0.842410612000000    -1.223524505000000
 C    7.808593479000000    0.835931020000000    -0.619522275000000
 O    6.548633909000000    1.202224581000000    -0.249008651000000
 C    8.982127974999999    1.560004073000000    -0.449315941000000
 H    8.989645334000000    2.533828446000000    0.023221612000000
 C    5.764749633000000    0.145540489000000    -0.645896351000000
 H    -4.071967012000000    -10.301916164000000    0.327551493000000
 C    -4.031768002000000    -9.247669589999999    0.058795254000000
 C    -5.024126132000000    -8.712795337999999    -0.773354176000000
 H    -5.821448280000000    -9.356181008000000    -1.140606199000000
 C    -5.008420282000000    -7.359098565000000    -1.143725459000000
 H    -5.773910695000000    -6.941391699000000    -1.787886311000000
 C    -3.969507885000000    -6.554324993000000    -0.659166301000000
 N    -3.693709021000001    -5.224655879000000    -0.849049524000000
 C    -3.008278216000000    -7.121955335000000    0.159933791000000
 O    -2.114320863000000    -6.147541710000000    0.492059936000000
 C    -2.987860914000000    -8.454786460999999    0.551826021000000
 H    -2.211915971000000    -8.849820536999999    1.194896532000000
 C    -2.586110020000000    -5.029444520000000    -0.152803881000000



