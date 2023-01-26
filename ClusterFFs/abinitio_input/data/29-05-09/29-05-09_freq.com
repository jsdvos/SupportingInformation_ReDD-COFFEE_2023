%nproc=15
%mem=70GB
%chk=TetraPhenylSilane_Ketoenamine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from TetraPhenylSilane_Ketoenamine_Ketoenamine-2.log

0 1
Si   0.000004   0.000005   0.005716
 C  -0.446257  -3.810892  -2.741302
 C  -3.821340   0.426596   2.738841
 C   0.446251   3.810900  -2.741306
 C   3.821362  -0.426585   2.738822
 C  -1.234972  -3.667865  -1.592758
 C  -3.665350   1.237672   1.607822
 C   1.234972   3.667874  -1.592766
 C   3.665367  -1.237662   1.607804
 C  -1.124621  -2.536915  -0.791983
 C  -2.536052   1.125926   0.804811
 C   1.124625   2.536925  -0.791990
 C   2.536065  -1.125916   0.804798
 C  -0.213086  -1.514451  -1.099782
 C  -1.515290   0.211459   1.109749
 C   0.213089   1.514460  -1.099783
 C   1.515304  -0.211449   1.109742
 C   0.577917  -1.668409  -2.249704
 C  -1.668072  -0.579947   2.259651
 C  -0.577920   1.668418  -2.249701
 C   1.668091   0.579957   2.259642
 C   0.464982  -2.794892  -3.058919
 C  -2.803514  -0.481821   3.058211
 C  -0.464989   2.794901  -3.058917
 C   2.803538   0.481832   3.058197
 H  -1.933271  -4.458238  -1.346278
 H  -4.443830   1.954485   1.376958
 H   1.933273   4.458248  -1.346289
 H   4.443846  -1.954474   1.376936
 H  -1.761390  -2.446705   0.081776
 H  -2.444094   1.766651  -0.065882
 H   1.761398   2.446715   0.081766
 H   2.444102  -1.766641  -0.065894
 H   1.288889  -0.895521  -2.524741
 H  -0.885295  -1.275569   2.545648
 H  -1.288894   0.895530  -2.524734
 H   0.885316   1.275579   2.545643
 H   1.105802  -2.882073  -3.927715
 H  -2.875894  -1.099044   3.945369
 H  -1.105814   2.882082  -3.927711
 H   2.875922   1.099055   3.945354
 C  -0.608451  -5.058917  -3.560141
 O  -1.196722  -6.034531  -3.056955
 C  -0.099294  -5.095902  -4.913680
 C  -0.194181  -6.209617  -5.707076
 H   0.347250  -4.216187  -5.351613
 N  -0.755622  -7.376424  -5.328790
 H   0.190099  -6.169108  -6.719824
 C  -0.857002  -8.559681  -6.074988
 H  -1.151893  -7.346901  -4.385902
 C  -0.139981  -8.780241  -7.258627
 C  -1.708212  -9.565740  -5.594899
 C  -0.293681  -9.978021  -7.951567
 H   0.550615  -8.036328  -7.634981
 C  -1.144493  -10.974806  -7.477087
 H   0.268206  -10.134676  -8.865507
 C  -1.846148  -10.760920  -6.290564
 H  -2.263297  -9.399533  -4.677855
 H  -1.254464  -11.905903  -8.019841
 H  -2.509173  -11.526774  -5.904453
 C  -5.062105   0.596020   3.566773
 O  -5.742925   1.629142   3.424808
 C  -5.449742  -0.448665   4.489256
 C  -6.559352  -0.352682   5.288197
 H  -4.880912  -1.364509   4.543439
 N  -7.397567   0.704030   5.313800
 H  -6.808852  -1.178561   5.944377
 C  -8.528918   0.880988   6.123560
 H  -7.150338   1.431835   4.638425
 C  -8.813555   0.066464   7.227806
 C  -9.400011   1.934654   5.810036
 C  -9.959141   0.296872   7.984761
 H  -8.142730  -0.733466   7.514225
 C  -10.827467   1.340317   7.668832
 H  -10.165902  -0.341456   8.836424
 C  -10.535671   2.160256   6.578668
 H  -9.182098   2.568673   4.957242
 H  -11.714267   1.515521   8.265916
 H  -11.197644   2.979035   6.320565
 C   0.608441   5.058925  -3.560145
 O   1.196719   6.034538  -3.056964
 C   0.099282   5.095908  -4.913684
 C   0.194170   6.209622  -5.707082
 H  -0.347263   4.216193  -5.351615
 N   0.755611   7.376429  -5.328799
 H  -0.190112   6.169111  -6.719830
 C   0.856991   8.559684  -6.074999
 H   1.151881   7.346908  -4.385911
 C   0.139969   8.780242  -7.258638
 C   1.708201   9.565745  -5.594912
 C   0.293669   9.978021  -7.951581
 H  -0.550627   8.036329  -7.634991
 C   1.144481   10.974807  -7.477103
 H  -0.268218   10.134674  -8.865521
 C   1.846136   10.760923  -6.290580
 H   2.263286   9.399540  -4.677868
 H   1.254452   11.905903  -8.019859
 H   2.509162   11.526778  -5.904471
 C   5.062132  -0.596007   3.566747
 O   5.742937  -1.629141   3.424800
 C   5.449757   0.448667   4.489247
 C   6.559359   0.352676   5.288200
 H   4.880925   1.364509   4.543436
 N   7.397578  -0.704034   5.313795
 H   6.808849   1.178545   5.944394
 C   8.528920  -0.881001   6.123565
 H   7.150361  -1.431828   4.638403
 C   8.813539  -0.066495   7.227829
 C   9.400022  -1.934657   5.810032
 C   9.959117  -0.296911   7.984794
 H   8.142707   0.733426   7.514254
 C   10.827453  -1.340346   7.668856
 H   10.165865   0.341403   8.836470
 C   10.535674  -2.160267   6.578674
 H   9.182122  -2.568661   4.957225
 H   11.714247  -1.515556   8.265948
 H   11.197654  -2.979037   6.320565





