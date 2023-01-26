%nproc=15
%mem=70GB
%chk=TPG_Aldehyde_Benzobisoxazole_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TPG_Aldehyde_Benzobisoxazole.log

0 1
 C   0.059215  -1.425428   0.000003
 C   1.254639  -0.651584   0.000026
 O   2.447290  -1.227937   0.000048
 C   1.204893   0.764079   0.000027
 C  -0.062989   1.412431   0.000029
 O  -0.160160   2.733472   0.000048
 C  -1.264121   0.661526   0.000005
 C  -1.191653  -0.760667  -0.000015
 O  -2.287110  -1.505367  -0.000054
 H   0.761402   3.130602   0.000060
 H  -3.091819  -0.905844  -0.000060
 H   2.330452  -2.224603   0.000045
 H  -0.759027  -8.286828  -0.000016
 C  -0.280277  -7.314858  -0.000007
 C   1.122902  -7.239738   0.000017
 H   1.698127  -8.158194   0.000026
 C   1.791123  -6.017208   0.000030
 H   2.872209  -5.957278   0.000049
 C   1.006187  -4.864471   0.000018
 N   1.330163  -3.514346   0.000027
 C  -0.387612  -4.964837  -0.000006
 O  -0.902734  -3.691291  -0.000012
 C  -1.073260  -6.165725  -0.000019
 H  -2.154592  -6.206148  -0.000037
 C   0.185039  -2.868429   0.000002
 H   7.556275   3.485860  -0.000104
 C   6.475145   3.414547  -0.000069
 C   5.708564   4.592216  -0.000012
 H   6.216409   5.549576  -0.000003
 C   4.315713   4.559726   0.000034
 H   3.723323   5.466045   0.000079
 C   3.709802   3.303618   0.000021
 N   2.378548   2.909204   0.000056
 C   4.493559   2.146696  -0.000037
 O   3.648140   1.063862  -0.000038
 C   5.876381   2.153271  -0.000083
 H   6.452002   1.236988  -0.000127
 C   2.391666   1.594532   0.000024
 H  -6.797305   4.800644   0.000021
 C  -6.194896   3.900073   0.000009
 C  -6.831381   2.647297  -0.000032
 H  -7.914398   2.608319  -0.000051
 C  -6.106702   1.457369  -0.000050
 H  -6.595300   0.491133  -0.000084
 C  -4.715941   1.560833  -0.000026
 N  -3.708654   0.605241  -0.000038
 C  -4.106009   2.818103   0.000016
 O  -2.745518   2.627490   0.000032
 C  -4.803229   4.012305   0.000034
 H  -4.297609   4.968998   0.000066
 C  -2.576728   1.274028   0.000004





