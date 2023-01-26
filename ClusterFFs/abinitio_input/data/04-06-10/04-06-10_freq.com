%nproc=15
%mem=70GB
%chk=Pyrene_Nitrile_Triazine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Pyrene_Nitrile_Triazine.log

0 1
 C  -0.000003   3.523399   0.000000
 C  -1.212415   2.823085   0.000000
 C  -1.234187   1.425744   0.000000
 C  -0.000003   0.710645   0.000000
 C   1.234182   1.425745   0.000000
 C   1.212410   2.823085   0.000000
 H  -2.141347   3.378692   0.000000
 H   2.141341   3.378695   0.000000
 C  -0.000003  -0.710645   0.000000
 C  -1.234187  -1.425745   0.000000
 C  -1.212415  -2.823086   0.000000
 C  -0.000003  -3.523399   0.000000
 C   1.212410  -2.823085   0.000000
 C   1.234182  -1.425745   0.000000
 H  -2.141346  -3.378694   0.000000
 H   2.141341  -3.378694   0.000000
 C   2.463338   0.679241   0.000000
 C   2.463338  -0.679241   0.000000
 H   3.398542   1.228642   0.000000
 H   3.398542  -1.228643   0.000000
 C  -2.463343   0.679241   0.000000
 C  -2.463343  -0.679242   0.000000
 H  -3.398547   1.228643   0.000000
 H  -3.398547  -1.228643   0.000000
 C  -0.000004   5.002802   0.000000
 N  -1.184161   5.641382   0.000000
 C  -1.117675   6.968852   0.000000
 N   1.184179   5.641369   0.000000
 C   1.117717   6.968822   0.000000
 N   0.000019   7.700294   0.000000
 H   2.060550   7.508908   0.000000
 H  -2.060509   7.508936   0.000000
 C   0.000003  -5.002802   0.000000
 N   1.184162  -5.641373   0.000000
 C   1.117686  -6.968843   0.000000
 N  -1.184178  -5.641377   0.000000
 C  -1.117706  -6.968830   0.000000
 N  -0.000003  -7.700293   0.000000
 H  -2.060536  -7.508922   0.000000
 H   2.060524  -7.508921   0.000000





