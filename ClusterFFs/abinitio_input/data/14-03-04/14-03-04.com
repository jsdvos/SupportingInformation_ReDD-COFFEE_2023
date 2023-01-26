%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_PrimaryAmine_Imine.chk
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
 N    5.064601967000000    -4.834987128000000    0.333015306000000
 C    5.030849286000000    -5.811227355000000    1.155359940000000
 C    6.049478078000001    -6.866547204000000    1.182864057000000
 C    5.950841647000000    -7.888101886000000    2.137280082000000
 C    6.900921145000000    -8.905343369000001    2.186711920000000
 C    7.959915376000000    -8.909866592000000    1.280983120000000
 C    8.066375552000000    -7.894087804000000    0.325784924000000
 C    7.120112107000000    -6.879638304000000    0.274231736000000
 H    4.232069965000000    -5.900745070000000    1.905004901000000
 H    5.125843261000000    -7.883344564000000    2.842749488000000
 H    6.815469190000000    -9.690918609000001    2.928715958000000
 H    8.701217060999999    -9.700271745000000    1.316846554000000
 H    8.890899836999999    -7.898918418000000    -0.378232027000000
 H    7.189086792000000    -6.086404100000000    -0.460024961000000
 N    1.660823923000000    6.805168763000000    -0.259912514000000
 C    2.477267141000000    7.345923504000000    0.559501731000000
 C    2.885357495000000    8.752029009999999    0.467901271000000
 C    3.765792907000000    9.272830434999999    1.425926545000000
 C    4.173627918000000    10.603138137000000    1.364065290000000
 C    3.703632430000000    11.425825549000001    0.342241645000000
 C    2.824368408000000    10.914561684999999    -0.617470786000000
 C    2.417180945000000    9.588718291999999    -0.558160646000000
 H    2.908566933000000    6.778438403000000    1.396144152000000
 H    4.130876198000000    8.631295976000001    2.221840975000000
 H    4.854987484000000    10.996181094000001    2.109898289000000
 H    4.019139334000000    12.461882546000002    0.291290962000000
 H    2.459501793000000    11.555870767000000    -1.411887878000000
 H    1.736915174000000    9.178450306000000    -1.294387219000000
 N    -6.721938446000000    -1.977444008000000    -0.210660290000000
 C    -7.614647827000000    -1.462179796000000    0.543002549000000
 C    -9.035018773999999    -1.818497120000000    0.455564503000000
 C    -9.945233569000001    -1.224561502000000    1.340416227000000
 C    -11.300388470000000    -1.540496243000000    1.280289478000000
 C    -11.758012934000000    -2.454864746000000    0.333535027000000
 C    -10.856615066000000    -3.052698153000000    -0.552817618000000
 C    -9.505598961000000    -2.738674939000000    -0.495155416000000
 H    -7.355096439000000    -0.726209779000000    1.316960301000000
 H    -9.587649967000001    -0.512847548000000    2.077931832000000
 H    -11.996217811999999    -1.075449850000000    1.969168180000000
 H    -12.812307349999999    -2.703016426000000    0.284151743000000
 H    -11.214143042000000    -3.763966510000000    -1.288870993000000
 H    -8.795583334000000    -3.193272315000000    -1.175103901000000



