%nproc=15
%mem=70GB
%chk=2_Phenyl_H_CarboxylicAnhydride_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from 2_Phenyl_H_CarboxylicAnhydride_Imide.log

0 1
 C   0.673120  -1.171631   0.177697
 C   1.394814  -0.001806   0.382594
 C   0.672750   1.169896   0.190107
 C  -0.673177   1.171631  -0.177587
 C  -1.394870   0.001806  -0.382487
 C  -0.672807  -1.169896  -0.189996
 H   2.439088  -0.003154   0.668357
 H  -2.439144   0.003154  -0.668250
 C  -1.128050   2.588487  -0.296021
 C   1.127186   2.585560   0.323517
 N  -0.000475   3.382687   0.017666
 O   2.222466   2.986904   0.625569
 O  -2.223297   2.992665  -0.594388
 C  -0.000559   4.811190   0.024697
 C   1.052211   5.505728  -0.572818
 H   1.867760   4.962013  -1.029635
 C   1.049700   6.897610  -0.557387
 H   1.870306   7.436858  -1.015973
 C  -0.000767   7.593837   0.038352
 H  -0.000851   8.677657   0.043665
 C  -1.051122   6.891638   0.627236
 H  -1.871809   7.426235   1.091092
 C  -1.053428   5.499670   0.629005
 H  -1.868897   4.951371   1.080454
 C   1.127992  -2.588487   0.296138
 C  -1.127244  -2.585560  -0.323400
 N   0.000429  -3.382687  -0.017596
 O  -2.222513  -2.986908  -0.625488
 O   2.223251  -2.992661   0.594469
 C   0.000517  -4.811190  -0.024692
 C  -1.052264  -5.505745   0.572781
 H  -1.867811  -4.962042   1.029616
 C  -1.049766  -6.897627   0.557284
 H  -1.870379  -7.436888   1.015841
 C   0.000695  -7.593836  -0.038485
 H   0.000768  -8.677655  -0.043852
 C   1.051055  -6.891620  -0.627340
 H   1.871734  -7.426204  -1.091226
 C   1.053374  -5.499652  -0.629043
 H   1.868845  -4.951341  -1.080474





