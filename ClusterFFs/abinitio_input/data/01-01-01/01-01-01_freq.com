%nproc=15
%mem=70GB
%chk=2_Phenyl_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from 2_Phenyl_BoronicAcid_BoronateEster.log

0 1
 C   0.056562  -1.411089   0.000000
 C  -0.056562   1.411089   0.000000
 C  -1.179526  -0.742631   0.000000
 C   1.179526   0.742631   0.000000
 C  -1.235182   0.645839   0.000000
 C   1.235182  -0.645839   0.000000
 H   2.195719  -1.149470   0.000000
 H  -2.195719   1.149470   0.000000
 H   2.096671   1.321527   0.000000
 H  -2.096671  -1.321527   0.000000
 B   0.118200  -2.948819   0.000000
 O  -0.995263  -3.785835   0.000000
 O   1.295079  -3.694034   0.000000
 C  -0.494277  -5.069670   0.000000
 C   0.898452  -5.013848   0.000000
 C  -1.179526  -6.269062   0.000000
 C   1.677497  -6.154548   0.000000
 C  -0.401548  -7.434155   0.000000
 C   0.995262  -7.378168   0.000000
 H  -2.261557  -6.298812   0.000000
 H   2.758437  -6.097602   0.000000
 H  -0.895389  -8.398543   0.000000
 H   1.564704  -8.299938   0.000000
 B  -0.118200   2.948819   0.000000
 O   0.995263   3.785835   0.000000
 O  -1.295079   3.694034   0.000000
 C   0.494277   5.069670   0.000000
 C  -0.898452   5.013848   0.000000
 C   1.179526   6.269062   0.000000
 C  -1.677497   6.154548   0.000000
 C   0.401548   7.434155   0.000000
 C  -0.995262   7.378168   0.000000
 H   2.261557   6.298812   0.000000
 H  -2.758437   6.097602   0.000000
 H   0.895389   8.398543   0.000000
 H  -1.564704   8.299938   0.000000





