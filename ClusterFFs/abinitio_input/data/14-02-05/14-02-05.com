%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_Aldehyde_Hydrazone.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    4.027462051000000    -3.893913718000000    0.310066179000000
 C    2.672148051000000    -4.262318718000000    0.319743179000000
 C    1.678672051000000    -3.300991718000001    0.229140179000000
 C    2.003629051000000    -1.937063718000000    0.160580179000000
 C    3.355470051000000    -1.566913718000000    0.187692179000000
 C    4.354395051000000    -2.527048718000000    0.261396179000000
 C    0.941359051000000    -0.909939718000000    0.077892179000000
 N    1.312939051000000    0.376982282000000    0.037767179000000
 C    0.318010051000000    1.271410282000000    -0.038992821000000
 N    -0.981642949000000    0.947974282000000    -0.075637821000000
 C    -1.260558949000000    -0.362144718000000    -0.031325821000000
 N    -0.332144949000000    -1.325265718000000    0.045418179000000
 C    0.676958051000000    2.706621282000000    -0.083768821000000
 C    -0.317350949000000    3.691753282000000    -0.190296821000000
 C    0.018704051000000    5.034651282000000    -0.245812821000000
 C    1.360665051000000    5.440586282000000    -0.161808821000000
 C    2.359252051000000    4.454275282000000    -0.075031821000000
 C    2.018771051000000    3.109724282000000    -0.036659821000000
 C    -2.682392949000000    -0.770638718000000    -0.066681821000000
 C    -3.700649949000000    0.190456282000000    -0.135975821000000
 C    -5.035099948999999    -0.188277718000000    -0.165449821000000
 C    -5.391426949000000    -1.548090718000000    -0.125807821000000
 C    -4.369830949000000    -2.511469718000000    -0.093439821000000
 C    -3.039347949000000    -2.128166718000000    -0.047251821000000
 H    0.636350051000000    -3.590416718000000    0.220005179000000
 H    2.423007051000000    -5.314639718000000    0.386235179000000
 H    5.392662051000000    -2.220384718000000    0.316972179000000
 H    3.609547051000000    -0.515584718000000    0.158735179000000
 H    -1.353435949000000    3.384405282000000    -0.238761821000000
 H    -0.746394949000000    5.795773282000000    -0.341783821000000
 H    3.403496051000000    4.745260282000000    -0.073396821000000
 H    2.789534051000000    2.352613282000000    0.021499179000000
 H    -3.429036949000000    1.237058282000000    -0.175087821000000
 H    -5.807122949000000    0.567416282000000    -0.254919821000000
 H    -4.647001949000000    -3.558870718000000    -0.092107821000000
 H    -2.256291949000000    -2.873402718000000    -0.005591821000000
 H    10.379554396999998    -9.172785209000001    0.246464561000000
 C    9.373874688000001    -9.572770307000001    0.353209518000000
 C    9.188805930999999    -10.945013830000001    0.509831611000000
 H    10.046944998000001    -11.612335224000001    0.527960409000000
 C    7.902235152000000    -11.462907254999999    0.632042730000000
 H    7.751818435000000    -12.533880182000003    0.743248816000000
 C    6.800411688000000    -10.605375089000001    0.605371323000000
 H    5.799398187000000    -11.025222878999999    0.696451706000000
 C    6.971728929000000    -9.219852836999999    0.455332949000000
 C    5.740240143000000    -8.367477324999999    0.415177764000000
 C    8.271871465000000    -8.711452921999999    0.319367929000000
 H    8.466978218000000    -7.657202452000000    0.162099326000000
 O    4.642739348000000    -8.912182192000000    0.318948647000000
 N    5.928116147000000    -6.993653621000000    0.494168032000000
 H    6.838072268000000    -6.584903983000000    0.646851611000000
 N    4.848878387000000    -6.165412836000000    0.405230631000000
 C    5.105025671000000    -4.897245459000000    0.411285112000000
 H    6.138208649000000    -4.512255125000000    0.480548220000000
 H    2.790336266000000    13.535700319000000    -0.964787971000000
 C    3.629502907000000    12.880319041000000    -0.743317947000000
 C    4.901674914000000    13.421402345000001    -0.568273804000000
 H    5.053592030000000    14.494839388000001    -0.650147233000000
 C    5.981398538000000    12.584614726000000    -0.299207078000000
 H    6.977855881000000    13.000648664000000    -0.172860256000000
 C    5.785915537000000    11.205658356000001    -0.197455155000000
 H    6.640865971000000    10.562627136000000    0.007889522000000
 C    4.508681320000000    10.646775468000000    -0.364355402000000
 C    4.382704963000000    9.157420220000001    -0.261050099000000
 C    3.431232675000000    11.498597923000000    -0.648662445000000
 H    2.429710327000000    11.124740599000001    -0.825941323000000
 O    5.404250798000000    8.474777603000000    -0.230807810000000
 N    3.094492293000000    8.641674279000000    -0.202375869000000
 H    2.280559996000000    9.236390685000002    -0.155096959000000
 N    2.916707979000000    7.290507566000000    -0.165119250000000
 C    1.690329949000000    6.878902688000000    -0.186912233000000
 H    0.840604467000000    7.583150993999999    -0.236741249000000
 H    -13.105034804000001    -4.419356375000000    -0.815513520000000
 C    -12.959129690999999    -3.348361119000000    -0.695284149000000
 C    -14.065181545000000    -2.503539793000000    -0.626295512000000
 H    -15.069811039999999    -2.914334256000000    -0.689672491000000
 C    -13.882718823000001    -1.130303683000000    -0.487214715000000
 H    -14.742081661000000    -0.465763704000000    -0.444120358000000
 C    -12.591917745999998    -0.603207751000000    -0.409175080000000
 H    -12.464232515999999    0.473673238000000    -0.306123396000000
 C    -11.468086420000001    -1.442744632000000    -0.470117649000000
 C    -10.116185129000000    -0.800684243000000    -0.400530021000000
 C    -11.664392677000000    -2.822988767000000    -0.624122696000000
 H    -10.837992967000000    -3.517857705000000    -0.716126989000000
 O    -10.033328409999999    0.422348884000000    -0.488825908000000
 N    -9.029108434999999    -1.649026808000000    -0.234675930000000
 H    -9.140706618999999    -2.641766159000000    -0.091766486000000
 N    -7.769835400000000    -1.126811662000000    -0.221654089000000
 C    -6.802115577000000    -1.981515376000000    -0.138608991000000
 H    -6.988801492000000    -3.069089354000000    -0.085569254000000



