%nproc=15
%mem=70GB
%chk=Pyrene_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Pyrene_BoronicAcid_BoronateEster.log

0 1
 C   0.000055   3.535039   0.000000
 C   1.209039   2.824886   0.000000
 C   1.235065   1.425956   0.000000
 C   0.000046   0.711640   0.000000
 C  -1.234976   1.425963   0.000000
 C  -1.208938   2.824889   0.000000
 H   2.147377   3.370011   0.000000
 H  -2.147270   3.370026   0.000000
 C   0.000046  -0.711639  -0.000000
 C   1.235065  -1.425962  -0.000000
 C   1.209032  -2.824890  -0.000000
 C   0.000044  -3.535038  -0.000000
 C  -1.208945  -2.824885  -0.000000
 C  -1.234975  -1.425956  -0.000000
 H   2.147364  -3.370026  -0.000000
 H  -2.147280  -3.370015  -0.000000
 C  -2.463536   0.679217   0.000000
 C  -2.463535  -0.679205  -0.000000
 H  -3.399138   1.228413   0.000000
 H  -3.399137  -1.228402  -0.000000
 C   2.463625   0.679207   0.000000
 C   2.463626  -0.679215  -0.000000
 H   3.399227   1.228404   0.000000
 H   3.399228  -1.228412  -0.000000
 B   0.000022   5.072663   0.000000
 O  -1.146175   5.864852   0.000000
 O   1.146177   5.864908   0.000000
 C  -0.696917   7.167762   0.000000
 C   0.696855   7.167798   0.000000
 C  -1.429555   8.338827   0.000000
 C   1.429434   8.338899   0.000000
 C  -0.699023   9.534327   0.000000
 C   0.698841   9.534362   0.000000
 H  -2.511908   8.325299   0.000000
 H   2.511788   8.325427   0.000000
 H  -1.231064   10.478193   0.000000
 H   1.230836   10.478255   0.000000
 B   0.000013  -5.072663  -0.000000
 O   1.146173  -5.864907  -0.000000
 O  -1.146179  -5.864855  -0.000000
 C   0.696854  -7.167796  -0.000000
 C  -0.696917  -7.167765  -0.000000
 C   1.429437  -8.338895  -0.000000
 C  -1.429552  -8.338832  -0.000000
 C   0.698848  -9.534361  -0.000000
 C  -0.699016  -9.534330  -0.000000
 H   2.511791  -8.325419  -0.000000
 H  -2.511906  -8.325309  -0.000000
 H   1.230844  -10.478252  -0.000000
 H  -1.231055  -10.478197  -0.000000





