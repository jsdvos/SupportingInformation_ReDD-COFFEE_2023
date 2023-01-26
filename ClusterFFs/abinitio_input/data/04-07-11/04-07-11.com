%nproc=15
%mem=70GB
%chk=Pyrene_AmineBorane_Borazine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 C    0.000149000000000    3.538221000000000    0.000000000000000
 C    -1.208314000000000    2.825814000000000    0.000000000000000
 C    -1.234523000000000    1.426499000000000    0.000000000000000
 C    0.000173000000000    0.711855000000000    0.000000000000000
 C    1.234878000000000    1.426489000000000    0.000000000000000
 C    1.208633000000000    2.825798000000000    0.000000000000000
 H    -2.147734000000000    3.368891000000000    0.000000000000000
 H    2.148055000000000    3.368886000000000    0.000000000000000
 C    0.000167000000000    -0.711882000000000    0.000000000000000
 C    -1.234536000000000    -1.426518000000000    0.000000000000000
 C    -1.208310000000000    -2.825825000000000    0.000000000000000
 C    0.000175000000000    -3.538236000000000    0.000000000000000
 C    1.208656000000000    -2.825835000000000    0.000000000000000
 C    1.234871000000000    -1.426526000000000    0.000000000000000
 H    -2.147732000000000    -3.368910000000000    0.000000000000000
 H    2.148078000000000    -3.368915000000000    0.000000000000000
 C    2.463326000000000    0.679235000000000    0.000000000000000
 C    2.463326000000000    -0.679271000000000    0.000000000000000
 H    3.399127000000000    1.228117000000000    0.000000000000000
 H    3.399123000000000    -1.228155000000000    0.000000000000000
 C    -2.462978000000000    0.679245000000000    0.000000000000000
 C    -2.462987000000000    -0.679261000000000    0.000000000000000
 H    -3.398774000000000    1.228133000000000    0.000000000000000
 H    -3.398787000000000    -1.228142000000000    0.000000000000000
 N    -0.005632916000000    4.996151365000000    0.016748070000000
 B    -1.190139092000000    5.800480890000000    0.576363721000000
 B    1.172182302000000    5.822512271000000    -0.524598686000000
 H    -2.141759916000000    5.284366578000000    1.030418688000000
 N    -1.212454567000000    7.326049319000000    0.598682250000000
 H    2.128278446000000    5.324519787000000    -0.989358086000000
 N    1.181227376000000    7.348352831000000    -0.514320838000000
 B    -0.019034041000000    8.085739533000000    0.049832927000000
 H    -2.033802528000000    7.840043928000000    0.988927811000000
 H    1.997904460000000    7.877688569000000    -0.893698488000000
 H    -0.024368627000000    9.266985843000001    0.062016797000000
 N    0.005632916000000    -4.996151365000000    -0.016748070000000
 B    1.190139092000000    -5.800480890000000    -0.576363721000000
 B    -1.172182302000000    -5.822512271000000    0.524598686000000
 H    2.141759916000000    -5.284366578000000    -1.030418688000000
 N    1.212454567000000    -7.326049319000000    -0.598682250000000
 H    -2.128278446000000    -5.324519787000000    0.989358086000000
 N    -1.181227376000000    -7.348352831000000    0.514320838000000
 B    0.019034041000000    -8.085739533000000    -0.049832927000000
 H    2.033802528000000    -7.840043928000000    -0.988927811000000
 H    -1.997904460000000    -7.877688569000000    0.893698488000000
 H    0.024368627000000    -9.266985843000001    -0.062016797000000




