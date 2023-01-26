%nproc=15
%mem=70GB
%chk=Adamantane_BoronicAcid_BoronateEster.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    -0.000001000000000    -0.000001000000000    1.760854000000000
 C    -0.000001000000000    -0.000001000000000    -1.760853000000000
 H    0.583501000000000    -0.548807000000000    -2.343561000000000
 H    -0.583501000000000    0.548807000000000    -2.343561000000000
 C    0.925717000000000    -0.871783000000000    0.897399000000000
 C    -0.925717000000000    0.871783000000000    0.897399000000000
 C    -0.871783000000000    -0.925717000000000    -0.897399000000000
 C    0.871783000000000    0.925717000000000    -0.897399000000000
 C    1.773370000000000    0.051568000000000    -0.002829000000000
 C    -1.773370000000000    -0.051569000000000    -0.002829000000000
 C    0.051568000000000    -1.773370000000000    0.002829000000000
 C    -0.051569000000000    1.773370000000000    0.002829000000000
 H    2.340312000000000    0.633966000000000    0.562907000000000
 H    -2.340313000000000    -0.633966000000000    0.562907000000000
 H    0.633966000000000    -2.340313000000000    -0.562907000000000
 H    -0.633966000000000    2.340312000000000    -0.562907000000000
 H    2.368699000000000    -0.498341000000000    -0.571393000000000
 H    -2.368700000000000    0.498341000000000    -0.571393000000000
 H    -0.498341000000000    -2.368700000000000    0.571393000000000
 H    0.498341000000000    2.368699000000000    0.571393000000000
 H    -0.548807000000000    -0.583501000000000    2.343561000000000
 H    0.548807000000000    0.583501000000000    2.343561000000000
 B    1.840833611000000    -1.733575094000000    1.784513466000000
 O    3.232982403000000    -1.697043593000000    1.756460095000000
 O    1.390147774000000    -2.656697886000000    2.725209269000000
 C    3.646588979000000    -2.614788854000000    2.697427637000000
 C    2.526149395000000    -3.198258764000000    3.286427262000000
 C    4.932237080000000    -2.963944345000000    3.062949391000000
 C    2.633562737000000    -4.160981055000000    4.271330768000000
 C    5.055719059000000    -3.939279473000000    4.060854200000000
 C    3.931835736000000    -4.524542713000000    4.651664152000000
 H    5.794463753000000    -2.503493180000000    2.597987682000000
 H    1.755686511000000    -4.606693502000000    4.721120694000000
 H    6.044760896000001    -4.245341849000000    4.380342212000000
 H    4.065692948000000    -5.275942373000000    5.420710028000000
 B    -1.840833475000000    1.733575536000000    1.784513177000000
 O    -3.232982278000000    1.697045171000000    1.756458913000000
 O    -1.390147485000000    2.656697647000000    2.725209575000000
 C    -3.646588707000000    2.614790450000000    2.697426502000000
 C    -2.526149024000000    3.198259255000000    3.286427032000000
 C    -4.932236757000000    2.963946858000000    3.062947561000000
 C    -2.633562210000000    4.160981301000000    4.271330794000000
 C    -5.055718578000000    3.939281750000000    4.060852620000000
 C    -3.931835155000000    4.524543883000000    4.651663479000000
 H    -5.794463508000000    2.503496547000000    2.597985151000000
 H    -1.755685907000000    4.606692887000000    4.721121425000000
 H    -6.044760369000000    4.245344819000000    4.380340109000000
 H    -4.065692246000000    5.275943392000000    5.420709524000000
 B    -1.733581369000000    -1.840822451000000    -1.784518883000000
 O    -2.736953626000000    -1.395359538000000    -2.641844205000000
 O    -1.616816060000000    -3.227720574000000    -1.839849401000000
 C    -3.247065653000000    -2.529293852000000    -3.235751073000000
 C    -2.566024849000000    -3.643368398000000    -2.748140690000000
 C    -4.261098099000000    -2.640039280000000    -4.167352061000000
 C    -2.863886643000000    -4.925655222000000    -3.166979088000000
 C    -4.573517082000001    -3.934958879000000    -4.600847193000000
 C    -3.890383057000000    -5.052457601000000    -4.111738110000000
 H    -4.782575192000000    -1.767102667000000    -4.538408408000000
 H    -2.327671554000000    -5.782941006000000    -2.780751571000000
 H    -5.362158891000000    -4.071235607000000    -5.331205511000000
 H    -4.159214735000000    -6.039059599000000    -4.469923520000000
 B    1.733581369000000    1.840822451000000    -1.784518883000000
 O    2.736953626000000    1.395359538000000    -2.641844205000000
 O    1.616816060000000    3.227720574000000    -1.839849401000000
 C    3.247065653000000    2.529293852000000    -3.235751073000000
 C    2.566024849000000    3.643368398000000    -2.748140690000000
 C    4.261098099000000    2.640039280000000    -4.167352061000000
 C    2.863886643000000    4.925655222000000    -3.166979088000000
 C    4.573517082000001    3.934958879000000    -4.600847193000000
 C    3.890383057000000    5.052457601000000    -4.111738110000000
 H    4.782575192000000    1.767102667000000    -4.538408408000000
 H    2.327671554000000    5.782941006000000    -2.780751571000000
 H    5.362158891000000    4.071235607000000    -5.331205511000000
 H    4.159214735000000    6.039059599000000    -4.469923520000000




