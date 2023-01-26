%nproc=15
%mem=70GB
%chk=PMDA_PrimaryAmine_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from PMDA_PrimaryAmine_Imide.log

0 1
 C   2.668640  -0.613869  -0.744155
 C   1.220732  -0.390588  -0.458884
 C   0.111244  -0.955482  -1.076348
 H   0.194590  -1.669383  -1.886002
 C  -1.113509  -0.534600  -0.572213
 C  -2.488741  -0.932024  -0.994627
 C  -1.220949   0.385736   0.470875
 C  -2.668854   0.608978   0.756174
 C  -0.111466   0.950632   1.088335
 H  -0.194801   1.664535   1.897989
 C   1.113289   0.529749   0.584199
 C   2.488529   0.927120   1.006653
 N  -3.372744  -0.210518  -0.157770
 N   3.372537   0.205562   0.169838
 O  -3.157819   1.332882   1.585970
 O  -2.802839  -1.705467  -1.863425
 O   2.802627   1.700604   1.875414
 O   3.157624  -1.337817  -1.573899
 C  -4.795232  -0.298248  -0.226939
 C   4.795046   0.293274   0.239050
 C  -5.567743   0.861401  -0.156896
 C  -6.952730   0.774668  -0.212069
 H  -5.091777   1.824853  -0.040462
 C  -7.568211  -0.469473  -0.361993
 H  -7.551777   1.672407  -0.157947
 C  -6.793076  -1.627751  -0.444190
 C  -5.409140  -1.543495  -0.364506
 H  -7.269350  -2.591880  -0.550683
 H  -4.809456  -2.440496  -0.428597
 C   5.409045   1.538540   0.376128
 C   6.793001   1.622736   0.455801
 H   4.809434   2.435622   0.439749
 C   7.567982   0.464365   0.374102
 H   7.269372   2.586866   0.561868
 C   6.952477  -0.779814   0.224711
 C   5.567478  -0.866467   0.169482
 H   7.551504  -1.677598   0.171048
 H   5.091394  -1.829919   0.053491
 C  -11.205294  -0.136386  -0.954924
 C  -12.395666   0.371382  -1.448797
 H  -12.403291   1.238697  -2.097549
 C  -13.575374  -0.280370  -1.074026
 H  -14.528101   0.085911  -1.438544
 C  -13.547155  -1.399326  -0.233995
 H  -14.478455  -1.882087   0.038669
 C  -12.338203  -1.903394   0.257224
 H  -12.302003  -2.768962   0.907343
 C  -11.177039  -1.250259  -0.122036
 C  -9.761637  -1.544316   0.231744
 C  -9.809162   0.331334  -1.171842
 N  -8.987484  -0.557173  -0.431173
 O  -9.423810   1.264664  -1.831491
 O  -9.330227  -2.427151   0.931181
 C   11.205041   0.130679   0.966500
 C   12.395363  -0.377195   1.460352
 H   12.402924  -1.244330   2.109345
 C   13.575162   0.274192   1.085186
 H   14.527860  -0.092215   1.449658
 C   13.547062   1.392893   0.244827
 H   14.478419   1.875397  -0.028103
 C   12.338138   1.897091  -0.246366
 H   12.302050   2.762491  -0.896712
 C   11.176906   1.244313   0.133267
 C   9.761492   1.538576  -0.220303
 C   9.808789  -0.336594   1.183827
 N   8.987241   0.551920   0.443318
 O   9.423373  -1.269694   1.843814
 O   9.330066   2.421204  -0.919971





