%nproc=15
%mem=70GB
%chk=BiPhenyl_Aldehyde_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from BiPhenyl_Aldehyde_Imine.log

0 1
 C  -0.021900   3.565905  -0.036578
 C   1.121028   2.858071  -0.441495
 C   1.126259   1.472999  -0.439583
 C  -0.006087   0.740098  -0.037804
 C  -1.145701   1.451988   0.363176
 C  -1.152081   2.841702   0.364462
 C   0.006087  -0.740099  -0.037804
 C  -1.126259  -1.473000  -0.439583
 C  -1.121028  -2.858072  -0.441495
 C   0.021900  -3.565906  -0.036577
 C   1.152081  -2.841704   0.364462
 C   1.145701  -1.451990   0.363176
 H  -1.992000  -3.416128  -0.762676
 H  -2.041004   3.373231   0.689476
 H   2.041004  -3.373233   0.689476
 H   1.992000   3.416127  -0.762676
 H  -2.008853  -0.943583  -0.779771
 H  -2.022810   0.913034   0.701799
 H   2.008853   0.943582  -0.779772
 H   2.022810  -0.913035   0.701798
 C   0.724228   9.955605  -0.425846
 C   1.850512   9.300317   0.075609
 C  -0.342825   9.207115  -0.920791
 C  -0.297588   7.815359  -0.899496
 C   0.826053   7.152015  -0.380281
 C   1.910700   7.911236   0.082922
 C  -0.060037   5.031314  -0.025408
 N   0.941834   5.752201  -0.352010
 H   2.689505   9.875158   0.451719
 H  -1.211056   9.707924  -1.334938
 H  -1.115537   7.238326  -1.315768
 H   2.785398   7.391141   0.455282
 H  -1.009659   5.479650   0.299394
 H   0.686079   11.038624  -0.445055
 C  -0.724228  -9.955606  -0.425846
 C  -1.850512  -9.300318   0.075610
 C   0.342825  -9.207116  -0.920791
 C   0.297588  -7.815360  -0.899496
 C  -0.826053  -7.152016  -0.380281
 C  -1.910700  -7.911237   0.082923
 C   0.060037  -5.031315  -0.025408
 N  -0.941834  -5.752203  -0.352009
 H  -2.689505  -9.875159   0.451720
 H   1.211056  -9.707925  -1.334938
 H   1.115536  -7.238327  -1.315767
 H  -2.785398  -7.391142   0.455283
 H   1.009659  -5.479651   0.299394
 H  -0.686079  -11.038625  -0.445054






