%nproc=15
%mem=70GB
%chk=DBA12_Catechol_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from DBA12_Catechol_BoronateEster.log

0 1
 C  -2.473691  -1.394315  -0.000003
 C  -3.466080  -2.396390  -0.000002
 C  -4.781361  -1.989531  -0.000001
 C  -5.138929  -0.640867  -0.000001
 C  -4.197981   0.364183  -0.000002
 C  -2.839483  -0.014601  -0.000003
 H  -3.196058  -3.443592  -0.000002
 H  -4.482247   1.407610  -0.000002
 C  -1.836232   0.988305  -0.000003
 C  -0.978764   1.841279  -0.000004
 C   0.029378   2.839263  -0.000003
 C  -0.342303   4.199723  -0.000002
 C   0.667663   5.135394  -0.000002
 C   2.014440   4.770776  -0.000001
 C   2.414406   3.453376  -0.000002
 C   1.407142   2.466258  -0.000003
 H  -1.384230   4.489433  -0.000002
 H   3.460179   3.177868  -0.000002
 C   1.774092   1.095978  -0.000004
 C  -1.105375  -1.768522  -0.000004
 C   0.062039  -2.084679  -0.000005
 C   1.432234  -2.451935  -0.000004
 C   1.783562  -3.817780  -0.000003
 C   3.124476  -4.130002  -0.000002
 C   4.113578  -3.145899  -0.000001
 C   3.808142  -1.803425  -0.000002
 C   2.444086  -1.445202  -0.000003
 H   1.022122  -4.585738  -0.000003
 H   4.579938  -1.045876  -0.000002
 C   2.083773  -0.073169  -0.000004
 B  -6.985386  -1.852159  -0.000000
 C  -8.468964  -2.245529   0.000002
 C  -8.852614  -3.597366   0.000003
 C  -9.471995  -1.261369   0.000003
 C  -10.197651  -3.954998   0.000004
 C  -10.817533  -1.617104   0.000004
 C  -11.180790  -2.964546   0.000005
 O  -5.921696  -2.756545  -0.000000
 O  -6.509486  -0.539577  -0.000000
 H  -10.481831  -5.001328   0.000005
 H  -11.582791  -0.849024   0.000005
 H  -9.189493  -0.214248   0.000002
 H  -8.088420  -4.366962   0.000002
 H  -12.229089  -3.242498   0.000006
 B   1.888591   6.975488  -0.000001
 C   2.289716   8.456990   0.000002
 C   1.310843   9.465182   0.000002
 C   3.643549   8.833540   0.000003
 C   1.673636   10.808832   0.000004
 C   4.008236   10.176680   0.000004
 C   3.022967   11.165012   0.000005
 O   0.573541   6.506466  -0.000000
 O   2.787408   5.907089  -0.000000
 H   0.909580   11.578108   0.000005
 H   5.056045   10.455361   0.000005
 H   4.409131   8.065324   0.000002
 H   0.262255   9.188170   0.000002
 H   3.306418   12.211837   0.000006
 B   5.096805  -5.123227  -0.000000
 C   6.179343  -6.211276   0.000002
 C   7.541875  -5.867540   0.000003
 C   5.828647  -7.572032   0.000002
 C   8.524173  -6.853496   0.000004
 C   6.809555  -8.559370   0.000004
 C   8.158085  -8.200171   0.000005
 O   5.348063  -3.749832  -0.000000
 O   3.722155  -5.367531  -0.000000
 H   9.572394  -6.576369   0.000005
 H   6.527067  -9.606158   0.000005
 H   4.780578  -7.850999   0.000002
 H   7.826200  -4.820913   0.000002
 H   8.922998  -8.968999   0.000007





