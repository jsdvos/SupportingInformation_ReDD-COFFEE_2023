%nproc=15
%mem=70GB
%chk=Pyrene_PrimaryAmine_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Pyrene_PrimaryAmine_Imide.log

0 1
 C  -0.002472   3.510907  -0.007726
 C  -1.188243   2.823440  -0.261519
 C  -1.207768   1.424423  -0.253125
 C  -0.000436   0.711122  -0.001567
 C   1.205861   1.427290   0.246793
 C   1.184307   2.826299   0.248955
 H  -2.101205   3.372503  -0.448229
 H   2.096492   3.377548   0.433079
 C   0.000606  -0.711133   0.001564
 C  -1.205692  -1.427301  -0.246794
 C  -1.184138  -2.826310  -0.248956
 C   0.002641  -3.510918   0.007722
 C   1.188412  -2.823450   0.261518
 C   1.207937  -1.424434   0.253124
 H  -2.096323  -3.377559  -0.433076
 H   2.101373  -3.372513   0.448231
 C   2.409808   0.682316   0.494069
 C   2.410793  -0.676618   0.497100
 H   3.327187   1.231480   0.676850
 H   3.328969  -1.223631   0.682315
 C  -2.410624   0.676607  -0.497099
 C  -2.409639  -0.682326  -0.494068
 H  -3.328800   1.223620  -0.682313
 H  -3.327019  -1.231491  -0.676848
 C   0.586769   7.147269  -0.382450
 C   1.202608   8.326076  -0.769457
 H   2.122486   8.310703  -1.341141
 C   0.588552   9.524820  -0.391201
 H   1.038900   10.469346  -0.673792
 C  -0.599508   9.525949   0.348334
 H  -1.050716   10.471333   0.626657
 C  -1.212455   8.328368   0.732032
 H  -2.132309   8.314745   1.303800
 C  -0.595548   7.148383   0.350346
 C  -0.998307   5.737627   0.604809
 C   0.990855   5.735751  -0.630483
 N  -0.003336   4.937596  -0.011023
 O   1.955625   5.326362  -1.228031
 O  -1.962549   5.330129   1.204507
 C  -0.586627  -7.147273   0.382452
 C  -1.202478  -8.326074   0.769459
 H  -2.122357  -8.310692   1.341143
 C  -0.588435  -9.524824   0.391202
 H  -1.038793  -10.469346   0.673793
 C   0.599624  -9.525966  -0.348333
 H   1.050822  -10.471354  -0.626656
 C   1.212584  -8.328392  -0.732031
 H   2.132437  -8.314778  -1.303800
 C   0.595689  -7.148400  -0.350345
 C   0.998463  -5.737648  -0.604809
 C  -0.990698  -5.735751   0.630484
 N   0.003500  -4.937607   0.011023
 O  -1.955468  -5.326351   1.228026
 O   1.962706  -5.330162  -1.204512






