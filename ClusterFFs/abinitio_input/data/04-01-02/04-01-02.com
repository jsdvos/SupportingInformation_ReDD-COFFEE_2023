%nproc=15
%mem=70GB
%chk=Pyrene_BoronicAcid_Boroxine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

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
 B    0.000231851000000    5.086769999000000    -0.000019085000000
 O    1.191107272000000    5.796255511000000    0.000142203000000
 B    1.197337676000000    7.170319173000000    0.000229744000000
 O    -1.190564724000000    5.796385472000000    0.000233870000000
 B    -1.196645320000000    7.170449801000000    0.000315559000000
 O    0.000383700000000    7.854113987000000    0.000269840000000
 H    2.223385823000000    7.763895600000000    0.000669829000000
 H    -2.222629168000000    7.764137339000000    0.000825343000000
 B    0.000233952000000    -5.086785001000000    -0.000019085000000
 O    -1.190576496000000    -5.796379559000000    0.000142203000000
 B    -1.196681075000000    -7.170443786000000    0.000229744000000
 O    1.191095502000000    -5.796291427000000    0.000233870000000
 B    1.197301923000000    -7.170355194000000    0.000315559000000
 O    0.000335511000000    -7.854128990000000    0.000269840000000
 H    -2.222674864000000    -7.764114167000000    0.000669829000000
 H    2.223340131000000    -7.763948779000000    0.000825343000000



