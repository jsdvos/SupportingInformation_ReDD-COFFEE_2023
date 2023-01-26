%nproc=15
%mem=70GB
%chk=N-TrisPhenyl_PrimaryAmine_Imine.chk
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
 N    -3.371655018000000    4.561844062000000    0.711568582000000
 C    -2.921481203000000    5.694341190000000    1.092657434000000
 C    -3.764290224000000    6.889363438000000    1.209515714000000
 C    -3.194325273000000    8.087113209000000    1.662007154000000
 C    -3.968038621000000    9.239109427000001    1.780949008000000
 C    -5.320772573000000    9.203431005000001    1.448275284000000
 C    -5.897774550000000    8.012393439000000    0.996565042000000
 C    -5.128282781000000    6.863196535000000    0.876527889000000
 H    -1.866495826000000    5.826588188999999    1.371128217000000
 H    -2.140610753000000    8.113704976999999    1.921450096000000
 H    -3.517539369000000    10.160501418000001    2.131787587000000
 H    -5.925850843000000    10.098448814999999    1.539807178000000
 H    -6.950405590000000    7.986160255000000    0.738140558000000
 H    -5.560978795000000    5.933520439000000    0.527834006000000
 N    5.701379969000000    0.233159452000000    -0.631323397000000
 C    6.461601864000001    -0.758119169000000    -0.367004861000000
 C    7.910985158000000    -0.734096840000000    -0.592331449000000
 C    8.675642222000000    -1.857191310000000    -0.248570705000000
 C    10.053326712000000    -1.865606337000000    -0.452995329000000
 C    10.679603603000000    -0.749124888000000    -1.003522647000000
 C    9.924362401000000    0.376006763000000    -1.348824550000000
 C    8.550973000999999    0.386257201000000    -1.146645148000000
 H    6.062008680000000    -1.686919400000000    0.063906741000000
 H    8.186902967000000    -2.726182127000000    0.180755456000000
 H    10.635672853000001    -2.739339110000000    -0.183803020000000
 H    11.751904415000000    -0.752449332000000    -1.163880361000000
 H    10.413040483999998    1.243955560000000    -1.777032577000000
 H    7.952789963000000    1.250130708000000    -1.409343694000000
 N    -2.481144764000000    -5.014009668000000    -0.127323921000000
 C    -3.742610185000000    -5.158054074000000    0.008289947000000
 C    -4.412516780000000    -6.459072772000000    -0.094843839000000
 C    -5.799253771000000    -6.531633243999999    0.094651696000000
 C    -6.464087510000000    -7.751875341000000    0.000417313000000
 C    -5.746912842000000    -8.912247925999999    -0.284240129000000
 C    -4.362900702000000    -8.849046925000000    -0.474134675000000
 C    -3.698342042000000    -7.633749857000000    -0.381426259000000
 H    -4.396069288000000    -4.302061123000000    0.228054957000000
 H    -6.356915458000000    -5.627233050000000    0.316925812000000
 H    -7.536931003000000    -7.797203897000000    0.148581576000000
 H    -6.261450447000001    -9.863738197000000    -0.358213082000000
 H    -3.806241566000000    -9.752721093000000    -0.695510156000000
 H    -2.626924780000000    -7.568999540000000    -0.526282587000000



