%nproc=15
%mem=70GB
%chk=F_Aldehyde_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from F_Aldehyde_Imine.log

0 1
 F   1.212326  -2.424630  -0.222475
 C   0.653893  -1.219124  -0.114565
 C  -0.729423  -1.150936  -0.078116
 F  -1.414943  -2.306942  -0.157723
 C  -1.435881   0.055912   0.037751
 C  -0.653893   1.219124   0.114565
 F  -1.212326   2.424630   0.222475
 C   0.729423   1.150936   0.078115
 F   1.414943   2.306942   0.157723
 C   1.435881  -0.055912  -0.037751
 C  -7.818900   0.891834   0.164688
 C  -7.138874   1.769189   1.011209
 C  -7.097898   0.067048  -0.698556
 C  -5.706468   0.098800  -0.703985
 C  -5.018347   0.967623   0.159116
 C  -5.749861   1.823229   0.995874
 C  -2.901291   0.022716   0.065090
 N  -3.618085   1.070897   0.196356
 H  -7.694018   2.422627   1.674655
 H  -7.620849  -0.596782  -1.377899
 H  -5.153082  -0.520104  -1.400962
 H  -5.207441   2.511666   1.632728
 H  -3.338420  -0.976738  -0.018339
 H  -8.902371   0.863838   0.165339
 C   7.818899  -0.891834  -0.164688
 C   7.138874  -1.769189  -1.011209
 C   7.097898  -0.067048   0.698556
 C   5.706468  -0.098800   0.703984
 C   5.018347  -0.967623  -0.159116
 C   5.749861  -1.823230  -0.995874
 C   2.901291  -0.022716  -0.065091
 N   3.618085  -1.070897  -0.196355
 H   7.694018  -2.422627  -1.674655
 H   7.620849   0.596783   1.377899
 H   5.153082   0.520104   1.400962
 H   5.207441  -2.511666  -1.632728
 H   3.338420   0.976738   0.018339
 H   8.902371  -0.863838  -0.165339






