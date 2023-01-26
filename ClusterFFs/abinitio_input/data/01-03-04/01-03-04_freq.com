%nproc=15
%mem=70GB
%chk=2_Phenyl_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from 2_Phenyl_PrimaryAmine_Imine.log

0 1
 C  -0.010967  -1.410958  -0.008598
 C   0.010967   1.410958   0.008598
 C  -1.207267  -0.684224   0.123206
 C   1.207267   0.684224  -0.123206
 C  -1.191525   0.703104   0.148719
 C   1.191525  -0.703104  -0.148719
 H   2.112681  -1.264789  -0.252125
 H  -2.112681   1.264789   0.252125
 H   2.152320   1.211811  -0.182723
 H  -2.152320  -1.211811   0.182723
 N   0.043402  -2.812681  -0.044567
 C  -0.717521  -3.511012   0.706639
 C  -0.751330  -4.977371   0.662601
 C  -1.594623  -5.670112   1.542093
 C  -1.649154  -7.061810   1.524070
 C  -0.859349  -7.775677   0.624965
 C  -0.014870  -7.092875  -0.256051
 C   0.040029  -5.705739  -0.240483
 H  -1.383962  -3.043766   1.445275
 H  -2.209298  -5.113780   2.242892
 H  -2.305167  -7.587023   2.208879
 H  -0.899473  -8.859049   0.608237
 H   0.599158  -7.649001  -0.955610
 H   0.688438  -5.163368  -0.917591
 N  -0.043402   2.812681   0.044567
 C   0.717521   3.511012  -0.706639
 C   0.751330   4.977371  -0.662601
 C   1.594623   5.670112  -1.542093
 C   1.649154   7.061810  -1.524070
 C   0.859349   7.775677  -0.624965
 C   0.014870   7.092875   0.256051
 C  -0.040029   5.705739   0.240483
 H   1.383962   3.043766  -1.445275
 H   2.209298   5.113780  -2.242892
 H   2.305167   7.587023  -2.208879
 H   0.899473   8.859049  -0.608237
 H  -0.599157   7.649001   0.955610
 H  -0.688438   5.163368   0.917591






