%nproc=15
%mem=70GB
%chk=TrisPhenyl_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TrisPhenyl_PrimaryAmine_Imine.log

0 1
 C  -5.639130  -0.978109   0.020426
 C  -4.772880  -1.775435   0.781736
 C  -3.407625  -1.526182   0.791626
 C  -2.849955  -0.496326   0.017328
 C  -3.717795   0.274306  -0.769485
 C  -5.088247   0.042227  -0.771130
 C  -1.388351  -0.242621   0.016572
 C  -0.892152   1.066271   0.016299
 C   0.483665   1.323370   0.015980
 C   1.369091   0.239192   0.015798
 C   0.903813  -1.080862   0.016152
 C  -0.477823  -1.305594   0.016508
 C   1.854388  -2.219760   0.016588
 C   1.620040  -3.357271  -0.769054
 C   2.506315  -4.428035  -0.770926
 C   3.666409  -4.394202   0.019106
 C   3.924665  -3.244723   0.779178
 C   3.026054  -2.187124   0.789424
 C   0.994692   2.716033   0.016392
 C   2.096239   3.082139  -0.770147
 C   2.580446   4.385040  -0.771949
 C   1.971939   5.372474   0.019126
 C   0.847996   5.021131   0.780054
 C   0.381321   3.714129   0.790167
 H  -1.587644   1.897143   0.015822
 H  -0.849719  -2.323318   0.016151
 H   2.436404   0.426009   0.014852
 H   0.748923  -3.389871  -1.413784
 H   3.227037  -1.326358   1.417114
 H   2.323646  -5.276740  -1.420260
 H   4.828689  -3.209987   1.375769
 H  -5.731798   0.623794  -1.421447
 H  -3.310374   1.044046  -1.415324
 H  -5.194915  -2.574771   1.379514
 H  -2.762734  -2.129769   1.420201
 H  -0.464062   3.457597   1.418515
 H   0.366501   5.786436   1.377429
 H   3.406165   4.651435  -1.421960
 H   2.559376   2.344314  -1.415668
 N  -7.010198  -1.277277   0.047336
 C  -7.874309  -0.337398   0.044662
 C  -9.320434  -0.584238   0.006839
 C  -10.202126   0.504841   0.033917
 C  -11.579815   0.302831  -0.002008
 C  -12.089684  -0.992465  -0.065129
 C  -11.217315  -2.085056  -0.092209
 C  -9.843818  -1.885762  -0.057636
 H  -7.574286   0.719302   0.086498
 H  -9.804422   1.513701   0.083582
 H  -12.252609   1.152546   0.019260
 H  -13.161572  -1.153234  -0.093223
 H  -11.614990  -3.092559  -0.141545
 H  -9.155861  -2.722046  -0.078869
 N   4.611204  -5.431841   0.045678
 C   4.229551  -6.650196   0.044980
 C   5.166596  -7.778996   0.007184
 C   4.664665  -9.087191   0.037315
 C   5.528677  -10.179135   0.001674
 C   6.905192  -9.972788  -0.064304
 C   7.414827  -8.670910  -0.094517
 C   6.555263  -7.581240  -0.060183
 H   3.164533  -6.918893   0.088794
 H   3.592249  -9.247405   0.089139
 H   5.129510  -11.186718   0.025361
 H   7.580538  -10.820555  -0.092211
 H   8.486052  -8.511363  -0.146100
 H   6.935222  -6.567249  -0.083808
 N   2.398526   6.709404   0.046081
 C   3.644597   6.987538   0.044304
 C   4.154329   8.363199   0.007044
 C   5.538395   8.581743   0.035225
 C   6.052789   9.875690   0.000045
 C   5.186395   10.965212  -0.063441
 C   3.803950   10.756498  -0.091645
 C   3.289300   9.467536  -0.057802
 H   4.409476   6.199167   0.086749
 H   6.212904   7.732640   0.085164
 H   7.125100   10.033101   0.022183
 H   5.583477   11.973750  -0.090956
 H   3.130602   11.604898  -0.141268
 H   2.221030   9.290279  -0.079849





