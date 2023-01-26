%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_Aldehyde_Benzobisoxazole.chk
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
 H    9.744034899000001    -7.799691374000000    0.447113317000000
 C    8.693251737000001    -7.514879695000000    0.451445822000000
 C    7.711611595000000    -8.512511247999999    0.518166970000000
 H    8.010501971000000    -9.558006230000000    0.564497973000000
 C    6.346219510000000    -8.187601344000001    0.525550929000000
 H    5.583493111000000    -8.956404555000001    0.576421962000000
 C    5.989438894000000    -6.834638679000000    0.462331719000000
 N    4.761636163000000    -6.224337531000000    0.449157157000000
 C    6.990151280000000    -5.879987080000000    0.396706359000000
 O    6.395195666000000    -4.654530164000000    0.339850369000000
 C    8.351404042000000    -6.158139720000000    0.389141774000000
 H    9.095031130000001    -5.373342179000000    0.336286017000000
 C    5.050423434000000    -4.935798960000000    0.372272302000000
 H    1.904072552000000    12.323789366000000    -0.690306448000000
 C    2.177889669000000    11.277624793999999    -0.564427179000000
 C    3.525346626000000    10.937081774999999    -0.386303626000000
 H    4.280649597000000    11.720635704999999    -0.376182409000000
 C    3.920198384000000    9.600406265000000    -0.221054775000000
 H    4.961714344999999    9.332542144000000    -0.083609819000000
 C    2.928077174000000    8.611922313999999    -0.240854576000000
 N    3.008547070000000    7.249228712000000    -0.111504862000000
 C    1.608352627000000    8.990436012000000    -0.420148020000000
 O    0.844443887000000    7.861187583000000    -0.406332856000000
 C    1.175040725000000    10.300202514000000    -0.585034965000000
 H    0.129243016000000    10.543446642999999    -0.722226814000000
 C    1.753008653000000    6.847210567000000    -0.219420166000000
 H    -11.618970350000000    -4.553893236000000    -0.494508990000000
 C    -10.850811205999999    -3.785332958000000    -0.427216018000000
 C    -11.229945452000001    -2.436946817000000    -0.389139310000000
 H    -12.285481548000000    -2.174427927000000    -0.427468923000000
 C    -10.270964311000000    -1.415739863000000    -0.303069326000000
 H    -10.559986132000001    -0.371220866000000    -0.274013288000000
 C    -8.919666416999998    -1.781466380000000    -0.257956531000000
 N    -7.780894489000000    -1.021860712000000    -0.178342891000000
 C    -8.587233422000001    -3.124939616000000    -0.298694843000000
 O    -7.228176023000000    -3.220645330000000    -0.246227336000000
 C    -9.503748484000001    -4.165886820000000    -0.381545662000000
 H    -9.191253269000001    -5.201830640000000    -0.410688401000000
 C    -6.805136216000000    -1.914909771000000    -0.177907150000000



