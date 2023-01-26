%nproc=15
%mem=70GB
%chk=DBA12_Catechol_BoronateEster.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    2.750545300000000    0.714052633000000    0.000000000000000
 C    3.967385300000000    1.427232633000000    0.000000000000000
 C    5.134923300000000    0.697167633000000    0.000000000000000
 C    5.134905300000000    -0.697294367000000    0.000000000000000
 C    3.967338300000000    -1.427295367000000    0.000000000000000
 C    2.750523300000000    -0.714054367000000    0.000000000000000
 H    3.973029300000000    2.508793633000000    0.000000000000000
 H    3.972857300000000    -2.508858367000000    0.000000000000000
 C    1.523744300000000    -1.428965367000000    0.000000000000000
 C    0.475695300000000    -2.034128367000000    0.000000000000000
 C    -0.756889700000000    -2.739018366999999    0.000000000000000
 C    -0.747637700000000    -4.149421367000000    0.000000000000000
 C    -1.963640700000000    -4.795554367000000    0.000000000000000
 C    -3.171301700000000    -4.098371367000000    0.000000000000000
 C    -3.219770700000000    -2.722223367000000    0.000000000000000
 C    -1.993726700000000    -2.024989367000000    0.000000000000000
 H    0.186211300000000    -4.695066367000000    0.000000000000000
 H    -4.159213700000000    -2.186260367000000    0.000000000000000
 C    -1.999634700000000    -0.605105367000000    0.000000000000000
 C    1.523820300000000    1.429081633000000    0.000000000000000
 C    0.475673300000000    2.034067633000000    0.000000000000000
 C    -0.756847700000000    2.739067633000000    0.000000000000000
 C    -0.747585700000000    4.149487633000000    0.000000000000000
 C    -1.963575700000000    4.795616633000000    0.000000000000000
 C    -3.171234700000000    4.098376633000000    0.000000000000000
 C    -3.219715700000000    2.722243633000000    0.000000000000000
 C    -1.993596700000000    2.025034633000000    0.000000000000000
 H    0.186303300000000    4.695069633000000    0.000000000000000
 H    -4.159103700000000    2.186199633000000    0.000000000000000
 C    -1.999479700000000    0.605114633000000    0.000000000000000
 B    7.229632068000000    -0.000151273000000    -0.000009000000000
 C    8.765326904000000    -0.000192057000000    0.000000000000000
 C    9.482778138000000    1.207929772000000    0.000003000000000
 C    9.482819943999999    -1.208307103000000    0.000004000000000
 C    10.874638091000000    1.209091136000000    0.000010000000000
 C    10.874683467000002    -1.209415698000000    0.000010000000000
 C    11.571294401999999    -0.000153013000000    0.000013000000000
 O    6.436131713000000    1.148083493000000    -0.000003000000000
 O    6.436050436000001    -1.148318937000000    -0.000002000000000
 H    11.417306255000000    2.147787527000001    0.000012000000000
 H    11.417383316000000    -2.148095783000000    0.000012000000000
 H    8.941242318000002    -2.147992246000000    0.000001000000000
 H    8.941165489999999    2.147590119000000    0.000000000000000
 H    12.655828880000000    -0.000134653000000    0.000018000000000
 B    -3.614888719000001    -6.261007815000000    -0.000009000000000
 C    -4.382758632000000    -7.590945569000000    0.000000000000000
 C    -3.695208238000000    -8.816330848000000    0.000003000000000
 C    -5.787763306000000    -7.608268795000000    0.000004000000000
 C    -4.390120814000000    -10.022304309000001    0.000010000000000
 C    -6.484643520000000    -8.813110387000000    0.000010000000000
 C    -5.785685134000000    -10.021017756000001    0.000013000000000
 O    -2.223739155000000    -6.147920318000000    -0.000003000000000
 O    -4.212452432000000    -4.999667893000000    -0.000002000000000
 H    -3.848510917000000    -10.961611698000000    0.000012000000000
 H    -7.568914238000000    -8.813772655999999    0.000012000000000
 H    -6.330774752000000    -6.669411477000000    0.000001000000000
 H    -2.610632174000000    -8.817100250000001    -0.000000000000000
 H    -6.327927416000000    -10.960266574000000    0.000018000000000
 B    -3.614689077000000    6.261118739000000    -0.000009000000000
 C    -4.382501998000000    7.591089397000000    0.000000000000000
 C    -5.787491820000000    7.608358608000000    0.000003000000000
 C    -3.694990955000000    8.816514545000000    0.000004000000000
 C    -6.484428313000000    8.813163573000001    0.000010000000000
 C    -4.389963391000001    10.022457580999999    0.000010000000000
 C    -5.785521063000000    10.021108141999999    0.000013000000000
 O    -4.212338596000000    4.999809521000000    -0.000003000000000
 O    -2.223555826000000    6.147941578000000    -0.000002000000000
 H    -7.568697316000000    8.813779123000000    0.000012000000000
 H    -3.848393096000000    10.961789816000000    0.000012000000000
 H    -2.610410937000000    8.817337804999999    0.000001000000000
 H    -6.330454647000000    6.669477786000000    0.000000000000000
 H    -6.327804782000000    10.960333037000000    0.000018000000000



