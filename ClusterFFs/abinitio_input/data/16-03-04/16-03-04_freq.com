%nproc=15
%mem=70GB
%chk=N-TrisPhenyl_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from N-TrisPhenyl_PrimaryAmine_Imine.log

0 1
 N   0.000959  -0.000039  -0.101112
 C  -0.551901   1.304876  -0.087707
 C   1.407316  -0.174255  -0.087535
 C  -0.852959  -1.130859  -0.088863
 C   2.220625   0.638470   0.714526
 C  -1.664955   1.601236   0.711384
 C  -0.557298  -2.241306   0.714172
 C   2.011787  -1.176224  -0.861138
 C   3.599010   0.475531   0.719068
 H   1.762774   1.395348   1.339834
 C   0.015740   2.330661  -0.857885
 C  -2.213391   2.876249   0.716393
 H  -2.093089   0.825035   1.334056
 C  -2.021873  -1.153619  -0.864117
 C  -1.387879  -3.353291   0.718295
 H   0.326273  -2.222793   1.340720
 C   3.385675  -1.360503  -0.827292
 H   1.393273  -1.817031  -1.477551
 C   4.209490  -0.519805  -0.063248
 H   4.204535   1.092852   1.372943
 C  -0.511976   3.612490  -0.823408
 H   0.882069   2.116920  -1.471832
 C  -1.654357   3.904050  -0.062455
 H  -3.052891   3.090619   1.367970
 C  -2.868751  -2.251018  -0.830937
 H  -2.266569  -0.297620  -1.481072
 C  -2.554027  -3.384361  -0.065683
 H  -1.157267  -4.186046   1.372987
 H   3.849900  -2.141946  -1.417209
 H  -0.065731   4.406396  -1.410570
 H  -3.776793  -2.262397  -1.422117
 N   5.591212  -0.755268  -0.084474
 C   6.418235   0.215810  -0.009901
 C   7.871576   0.021467   0.037613
 C   8.712008   1.142275   0.087746
 C   10.095791   0.990310   0.132785
 C   10.654538  -0.286280   0.128175
 C   9.823996  -1.410263   0.078407
 C   8.444440  -1.260964   0.034702
 H   6.080126   1.261612   0.005061
 H   8.276873   2.136840   0.090796
 H   10.735567   1.864623   0.171320
 H   11.731337  -0.408102   0.163339
 H   10.259139  -2.403438   0.075217
 H   7.788950  -2.122351  -0.002990
 N  -2.141430   5.218414  -0.082812
 C  -3.396222   5.448755  -0.012274
 C  -3.954900   6.804357   0.036715
 C  -5.345948   6.971525   0.082530
 C  -5.906532   8.245716   0.129205
 C  -5.080466   9.367992   0.130571
 C  -3.691623   9.210977   0.085050
 C  -3.130846   7.941782   0.039694
 H  -4.132807   4.632938  -0.001932
 H  -5.989612   6.097337   0.081020
 H  -6.983734   8.362403   0.164414
 H  -5.513586   10.361293   0.167037
 H  -3.049171   10.084481   0.086456
 H  -2.056979   7.805039   0.005265
 N  -3.449295  -4.462902  -0.086900
 C  -3.022307  -5.664747  -0.011039
 C  -3.917751  -6.825795   0.037682
 C  -3.367835  -8.114141   0.090728
 C  -4.191774  -9.236161   0.137444
 C  -5.576634  -9.081231   0.131547
 C  -6.134248  -7.799860   0.078800
 C  -5.314739  -6.680156   0.033474
 H  -1.947656  -5.895168   0.005115
 H  -2.288998  -8.234974   0.094916
 H  -3.754889  -10.227462   0.178309
 H  -6.220890  -9.952538   0.168009
 H  -7.211888  -7.679734   0.074619
 H  -5.732592  -5.681715  -0.006393






