%nproc=15
%mem=70GB
%chk=DBA18_Cyanohydrine_Benzobisoxazole.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    4.268980742000000    0.714070222000000    0.000000000000000
 C    5.485820742000000    1.427250222000000    0.000000000000000
 C    6.653358742000000    0.697185222000000    0.000000000000000
 C    6.653340742000000    -0.697276778000000    0.000000000000000
 C    5.485773742000000    -1.427277778000000    0.000000000000000
 C    4.268958742000000    -0.714036778000000    0.000000000000000
 H    5.491464742000000    2.508811222000000    0.000000000000000
 H    5.491292742000000    -2.508840778000000    0.000000000000000
 C    -1.516059070000000    4.054085222000000    0.000000000000000
 C    -1.506797070000000    5.464505222000000    0.000000000000000
 C    -2.722787070000000    6.110634222000000    0.000000000000000
 C    -3.930446070000000    5.413394222000000    0.000000000000000
 C    -3.978927070000000    4.037261222000000    0.000000000000000
 C    -2.752808070000000    3.340052222000000    0.000000000000000
 H    -0.572908070000000    6.010087222000000    0.000000000000000
 H    -4.918315070000000    3.501217222000000    0.000000000000000
 C    -1.516101070000000    -4.054000778000000    0.000000000000000
 C    -1.506849070000000    -5.464403778000000    0.000000000000000
 C    -2.722852070000000    -6.110536777999999    0.000000000000000
 C    -3.930513069999999    -5.413353778000000    0.000000000000000
 C    -3.978982070000000    -4.037205778000000    0.000000000000000
 C    -2.752938070000000    -3.339971778000000    0.000000000000000
 H    -0.573000070000000    -6.010048778000000    0.000000000000000
 H    -4.918425070000000    -3.501242778000000    0.000000000000000
 C    -2.752973465999999    -1.920029778000000    0.000000000000000
 C    -2.752973465999999    -0.710029778000000    0.000000000000000
 C    -2.752973465999999    0.709970222000000    0.000000000000000
 C    -2.752973465999999    1.919970222000000    0.000000000000000
 C    -0.286317393000000    -3.344029778000000    0.000000000000000
 C    0.761573346000000    -2.739029778000000    0.000000000000000
 C    1.991329419000000    -2.029029778000000    0.000000000000000
 C    3.039220158000000    -1.424029778000000    0.000000000000000
 C    -0.286317393000000    3.343970222000000    0.000000000000000
 C    0.761573346000000    2.738970222000000    0.000000000000000
 C    1.991329419000000    2.028970222000000    0.000000000000000
 C    3.039220158000000    1.423970222000000    0.000000000000000
 N    7.982106622000000    1.111166648000000    0.064989441000000
 O    7.911808115000000    -1.154280371000000    -0.022003196000000
 C    8.676868989999999    -0.013904704000000    0.030920069000000
 C    10.856202636000001    -1.250476979000000    -0.032420345000000
 C    12.254242640999999    -1.248995756000000    -0.038546109000000
 H    10.333570225000001    -2.203431521000000    -0.081511553000000
 C    12.951652554000001    -0.044281237000000    0.021256462000000
 H    12.797228125000000    -2.189314909000000    -0.091450314000000
 C    12.254139165000000    1.159469319000000    0.088568927000000
 H    14.038669235000000    -0.043426993000000    0.014871726000000
 C    10.856919062999999    1.157258280000000    0.094243010000000
 H    12.794642825000000    2.101403722000000    0.135334292000000
 H    10.324629549999999    2.106081789000000    0.145564429000000
 C    10.137954308999999    -0.046635267000000    0.032608119000000
 N    -4.953330796000000    6.357151793000000    0.064989441000000
 O    -2.956242544000000    7.428986943000000    -0.022003196000000
 C    -4.326366906000000    7.521366797000000    0.030920069000000
 C    -4.345120603000000    10.027011311000001    -0.032420345000000
 C    -5.045418495000000    11.237011688000001    -0.038546109000000
 H    -3.258521459000000    10.050871248000000    -0.081511553000000
 C    -6.437436823000000    11.238634752999999    0.021256462000000
 H    -4.502567164000000    12.177408294999999    -0.091450314000000
 C    -7.131163561000000    10.032697963000000    0.088568927000000
 H    -6.981681159000000    12.179593889000000    0.014871726000000
 C    -6.430643578000000    8.823772549999999    0.094243010000000
 H    -8.217154525000000    10.029825047999999    0.135334292000000
 H    -6.986207862000000    7.888386799000000    0.145564429000000
 C    -5.028558889000000    8.803071919000001    0.032608119000000
 N    -3.028715946000000    -7.468287504000000    0.064989441000000
 O    -4.955506170000000    -6.274691464000000    -0.022003196000000
 C    -4.350437345000000    -7.507439013000000    0.030920069000000
 C    -6.511002060000000    -8.776519879000000    -0.032420345000000
 C    -7.208734405000000    -9.988001461000000    -0.038546109000000
 H    -7.074972439000000    -7.847431935000000    -0.081511553000000
 C    -6.514121125000000    -11.194330623999999    0.021256462000000
 H    -8.294567421000000    -9.988085481000001    -0.091450314000000
 C    -5.122885878000000    -11.192135983000002    0.088568927000000
 H    -7.056885874000000    -12.136143991999999    0.014871726000000
 C    -4.426195521000000    -9.980999555000000    0.094243010000000
 H    -4.577394803000000    -12.131190889000001    0.135334292000000
 H    -3.338345448000000    -9.994430686999999    0.145564429000000
 C    -5.109320472000000    -8.756413792000000    0.032608119000000




