%nproc=15
%mem=70GB
%chk=3_Phenyl_BoronicAcid_Boroxine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    1.212954000000000    0.700272000000000    0.000000000000000
 C    1.208877000000000    -0.698223000000000    0.000000000000000
 C    -0.000137000000000    -1.400767000000000    0.000000000000000
 C    -1.209188000000000    -0.698053000000000    0.000000000000000
 C    -1.213163000000000    0.700261000000000    0.000000000000000
 C    0.000000000000000    1.395989000000000    0.000000000000000
 H    -2.145136000000000    -1.238581000000000    0.000000000000000
 H    -0.000098000000000    2.476797000000000    0.000000000000000
 H    2.144938000000000    -1.238545000000000    0.000000000000000
 B    2.554058672000000    1.474508572000000    -0.000019085000000
 O    3.763945922000000    0.797953473000000    0.000142203000000
 B    4.957017971000000    1.479619705000000    0.000229744000000
 O    2.573170455000000    2.860576874000000    0.000233870000000
 B    3.760087312000000    3.552904924000000    0.000315559000000
 O    4.950689898000000    2.858109502000000    0.000269840000000
 H    5.984109234000000    0.887850059000000    0.000669829000000
 H    3.761213983000000    4.738276798000000    0.000825343000000
 B    -2.554313056000000    1.474418955000000    -0.000019085000000
 O    -2.573502432000000    2.860487397000000    0.000142203000000
 B    -3.760458476000000    3.552748270000000    0.000229744000000
 O    -3.764161380000000    0.797796732000000    0.000233870000000
 B    -4.957272005000001    1.479395435000000    0.000315559000000
 O    -4.951021592999999    2.857885959000000    0.000269840000000
 H    -3.761653098000000    4.738120088000000    0.000669829000000
 H    -5.984329334000000    0.887566918000000    0.000825343000000
 B    -0.000306092000000    -2.949315992000000    -0.000019085000000
 O    -1.191221024000000    -3.658735181000000    0.000142203000000
 B    -1.197527953000000    -5.032798494000000    0.000229744000000
 O    1.190450960000000    -3.658997781000000    0.000233870000000
 B    1.196455032000000    -5.033062447000000    0.000315559000000
 O    -0.000612061000000    -5.716659967000000    0.000269840000000
 H    -2.223609156000000    -5.626317777000000    0.000669829000000
 H    2.222405815000000    -5.626807123999999    0.000825343000000




