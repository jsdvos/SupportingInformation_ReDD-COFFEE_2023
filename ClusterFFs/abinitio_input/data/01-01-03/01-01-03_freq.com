%nproc=15
%mem=70GB
%chk=2_Phenyl_BoronicAcid_Borosilicate_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from 2_Phenyl_BoronicAcid_Borosilicate.log

0 1
 C   0.442181   1.344401   0.000684
 C  -0.442181  -1.344401   0.000684
 C  -0.928521   1.037099   0.000645
 C   0.928521  -1.037099   0.000645
 C  -1.362835  -0.283436   0.000741
 C   1.362835   0.283436   0.000741
 H   2.424044   0.506400   0.000832
 H  -2.424044  -0.506400   0.000832
 H   1.650189  -1.846462   0.000570
 H  -1.650189   1.846462   0.000570
 B   0.928521   2.822694   0.000618
 O   2.278423   3.090027   0.001461
 O   0.001202   3.839526  -0.000284
Si   3.285584   4.395575   0.002601
Si  -0.033470   5.488091  -0.002883
 O   3.010253   5.308891  -1.348462
 O   3.005697   5.310057   1.351912
 H   4.669547   3.941844   0.005117
 O   0.731053   6.060486   1.347398
 O   0.734605   6.056285  -1.352894
 H  -1.416330   5.945192  -0.005366
 B   1.975817   5.993985  -1.924254
 H   2.161960   6.554659  -2.955671
 B   1.970117   5.997111   1.923289
 H   2.153338   6.558690   2.954736
 B  -0.928521  -2.822694   0.000618
 O  -2.278423  -3.090027   0.001461
 O  -0.001202  -3.839526  -0.000284
Si  -3.285584  -4.395575   0.002601
Si   0.033470  -5.488091  -0.002883
 O  -3.010253  -5.308891  -1.348462
 O  -3.005697  -5.310057   1.351912
 H  -4.669547  -3.941844   0.005117
 O  -0.731053  -6.060486   1.347398
 O  -0.734605  -6.056285  -1.352894
 H   1.416330  -5.945192  -0.005366
 B  -1.975817  -5.993985  -1.924254
 H  -2.161960  -6.554659  -2.955671
 B  -1.970117  -5.997111   1.923289
 H  -2.153338  -6.558690   2.954736






