%nproc=15
%mem=70GB
%chk=TrisPhenyl_Hydrazide_Hydrazone.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    5.413628598000000    -1.872546979000000    -0.010594081000000
 C    5.035722949999999    -0.755661278000000    0.753536436000000
 C    3.723673695000000    -0.300430655000000    0.759525264000000
 C    2.735803795999999    -0.944625681000000    0.000449131000000
 C    3.108315992000000    -2.059076085000000    -0.765208093000000
 C    4.420719166000000    -2.513357956000000    -0.770224134000000
 C    1.334053259000000    -0.458848890000000    0.000402830000000
 C    0.268355259000000    -1.366873890000000    0.008860830000000
 C    -1.059173741000000    -0.922488890000000    0.008337830000000
 C    -1.309241741000000    0.453656110000000    0.006809830000000
 C    -0.264218741000000    1.383363110000000    0.003920830000000
 C    1.053270259000000    0.911980110000000    -0.001354170000000
 C    -0.547166741000000    2.832910110000000    0.023508830000000
 C    0.132648259000000    3.710843110000000    -0.833951170000000
 C    -0.134981741000000    5.073690110000000    -0.817152170000000
 C    -1.095013741000000    5.616633110000000    0.054395830000000
 C    -1.773249741000000    4.731796110000000    0.910653830000000
 C    -1.504821741000000    3.368908110000000    0.897304830000000
 C    -2.182774741000000    -1.891955890000000    -0.017346170000000
 C    -3.315488741000000    -1.662366890000000    -0.812060170000000
 C    -4.366923741000000    -2.569668890000000    -0.831200170000000
 C    -4.330026741000000    -3.741083890000000    -0.056511170000000
 C    -3.194045741000000    -3.964973890000000    0.739344830000000
 C    -2.141176741000000    -3.059528890000000    0.758579830000000
 H    0.474380259000000    -2.429560890000000    -0.040814170000000
 H    1.873134259000000    1.618471110000000    0.074149830000000
 H    -2.335361741000000    0.804210110000000    -0.020704170000000
 H    0.839454259000000    3.316212110000000    -1.556644170000000
 H    -2.004881741000000    2.721095110000000    1.609926830000000
 H    0.396903259000000    5.726432110000000    -1.500808170000000
 H    -2.509284741000000    5.119474110000000    1.606548830000000
 H    4.680359519000000    -3.384593294000000    -1.361290287000000
 H    2.355934167000000    -2.588063678000000    -1.339179593000000
 H    5.783140304000000    -0.232983007000000    1.339887964000000
 H    3.464345167000000    0.579450163000000    1.337108744000000
 H    -1.287700741000000    -3.244167890000000    1.400987830000000
 H    -3.142543741000000    -4.854378890000000    1.357384830000000
 H    -5.225637741000000    -2.375433890000000    -1.464217170000000
 H    -3.358715741000000    -0.779456890000000    -1.439657170000000
 C    6.850196550000000    -2.295056391000000    0.039199456000000
 O    7.653912013000001    -1.598861911000000    0.655499814000000
 N    7.179769382000000    -3.469252998000000    -0.625309763000000
 H    6.480924267000000    -4.054304669000000    -1.058598041000000
 N    8.481323482000001    -3.873255978000000    -0.662529651000000
 C    8.726179629000001    -4.934308889000000    -1.361177466000000
 C    10.094776777000000    -5.470894233000000    -1.491872635000000
 H    7.926332257000000    -5.475739210000000    -1.897441759000000
 C    10.314889389999999    -6.577121254000000    -2.317523409000000
 C    11.606930544000001    -7.087783072000000    -2.457009754000000
 H    9.496823253000001    -7.044292627000000    -2.858265836000000
 C    12.670642064999999    -6.496860789000000    -1.772885691000000
 H    11.787638039999999    -7.945442527000000    -3.100298907000000
 C    12.447812174999999    -5.393929585000000    -0.948126657000000
 H    13.676518965000000    -6.894915739000000    -1.882986799000000
 C    11.159656269999999    -4.878014155000000    -0.806918101000000
 H    13.275778996000000    -4.932641391000000    -0.415569606000000
 H    10.991180685000000    -4.015291709000000    -0.164731256000000
 C    -1.443660559000000    7.071832917000000    0.128961241000000
 O    -2.371305976000000    7.424818606000000    0.853769156000000
 N    -0.676954441000000    7.938424788000000    -0.639224077000000
 H    0.124105905000000    7.622453686000000    -1.165411560000000
 N    -0.984168844000000    9.266579601000000    -0.655683327000000
 C    -0.276276696000000    10.002764657000000    -1.449923122000000
 C    -0.513949729000000    11.454370277000001    -1.569989214000000
 H    0.525820426000000    9.576399794000000    -2.078680000000000
 C    0.230368585000000    12.190396092000000    -2.496349420000000
 C    0.007638009000000    13.562675861000001    -2.626124616000000
 H    0.977278555000000    11.710980452999999    -3.122584050000000
 C    -0.951558185000000    14.194127100999999    -1.832456504000000
 H    0.579323389000000    14.141980231000000    -3.346983927000000
 C    -1.691793194000000    13.457393529000001    -0.907462644000000
 H    -1.124707051000000    15.262703039000000    -1.935039489000000
 C    -1.475313907000000    12.085859311000000    -0.775232426000000
 H    -2.439542918000000    13.948233767000000    -0.289498991000000
 H    -2.057581309000000    11.514585394999999    -0.054338001000000
 C    -5.417122132000000    -4.771312302000000    -0.016836090000000
 O    -5.232840099000000    -5.806228092000000    0.619922004000000
 N    -6.582043219000000    -4.477928754000000    -0.713839232000000
 H    -6.727165040000000    -3.586660858000000    -1.164383214000000
 N    -7.583345309000000    -5.401961375000000    -0.760792833000000
 C    -8.607117067000003    -5.092531594000000    -1.488832105000000
 C    -9.754476649000001    -6.009525675000000    -1.632884363000000
 H    -8.661405221000001    -4.137326866000000    -2.041207635000000
 C    -10.801881008000001    -5.657983528000000    -2.489184085000000
 C    -11.888215302000003    -6.521877953000000    -2.641389544000000
 H    -10.782705986000000    -4.724313992000000    -3.043960783000000
 C    -11.926990504000001    -7.727909353000000    -1.939476570000000
 H    -12.705236834000001    -6.258072918000000    -3.308426719000000
 C    -10.881052243999999    -8.075462064000000    -1.084164852000000
 H    -12.773113732000001    -8.400240998999999    -2.059493137000000
 C    -9.792164339999999    -7.217536322000000    -0.930084427000000
 H    -10.910085948000001    -9.014872575000000    -0.537723820000000
 H    -8.976873164000001    -7.494478518000000    -0.263990361000000



