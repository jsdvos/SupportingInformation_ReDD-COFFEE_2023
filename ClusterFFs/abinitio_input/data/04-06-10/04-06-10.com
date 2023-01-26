%nproc=15
%mem=70GB
%chk=Pyrene_Nitrile_Triazine.chk
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
 C    0.000214722000000    5.018072264000000    0.000000000000000
 N    -1.184039179000000    5.654841914000000    -0.000000000000000
 C    -1.117831194000000    6.982536681000000    0.000000000000000
 N    1.184608972000000    5.654694592000000    0.000000000000000
 C    1.118590693000000    6.982347109000000    0.000000000000000
 N    0.000414370000000    7.712980073000000    0.000000000000000
 H    2.061332336000000    7.522469635000000    0.000000000000000
 H    -2.060525750000000    7.522742440000000    -0.000000000000000
 C    0.000244790000000    -5.018087264000000    -0.000000000000000
 N    1.184556995000000    -5.654748468000000    -0.000000000000000
 C    1.118470590000000    -6.982449292000000    -0.000000000000000
 N    -1.184091159000000    -5.654818045000000    -0.000000000000000
 C    -1.117951305000000    -6.982464512000000    -0.000000000000000
 N    0.000291918000000    -7.712995080000000    -0.000000000000000
 H    -2.060643485000000    -7.522673364000000    -0.000000000000000
 H    2.061214609000000    -7.522568725000000    -0.000000000000000



