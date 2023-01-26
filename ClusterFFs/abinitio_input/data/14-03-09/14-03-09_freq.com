%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_PrimaryAmine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Triazine-TrisPhenyl_PrimaryAmine_Ketoenamine.log

0 1
 C   3.971295  -3.947409   0.287871
 C   2.613639  -4.302875   0.260726
 C   1.638236  -3.320426   0.184049
 C   1.976121  -1.960809   0.136423
 C   3.334690  -1.612527   0.169521
 C   4.315725  -2.585979   0.243603
 C   0.929023  -0.922885   0.056119
 N   1.317663   0.359278   0.017931
 C   0.333888   1.267067  -0.054598
 N  -0.970537   0.960901  -0.089530
 C  -1.266162  -0.346002  -0.047089
 N  -0.350402  -1.321991   0.025995
 C   0.710935   2.693949  -0.098476
 C  -0.267993   3.696051  -0.173359
 C   0.086389   5.033330  -0.214234
 C   1.437631   5.417090  -0.181685
 C   2.422278   4.419735  -0.103706
 C   2.057294   3.082710  -0.064391
 C  -2.690054  -0.735002  -0.083003
 C  -3.698601   0.235690  -0.157615
 C  -5.038650  -0.118753  -0.190911
 C  -5.410893  -1.471631  -0.152539
 C  -4.404260  -2.448870  -0.075719
 C  -3.069363  -2.085024  -0.041761
 H   0.592981  -3.599080   0.163810
 H   2.308712  -5.340257   0.305213
 H   5.362462  -2.302158   0.265773
 H   3.608831  -0.566596   0.134895
 H  -1.310818   3.409408  -0.199550
 H  -0.681293   5.797432  -0.273500
 H   3.473095   4.676042  -0.069567
 H   2.819743   2.317457  -0.003904
 H  -3.416358   1.279759  -0.187143
 H  -5.785139   0.662992  -0.242996
 H  -4.682865  -3.496784  -0.044686
 H  -2.300773  -2.844091   0.016616
 H   9.923688  -6.115973   0.159830
 C   9.777727  -7.154210   0.430579
 C   10.854917  -8.027060   0.530408
 H   11.858684  -7.671670   0.326221
 C   10.645939  -9.357823   0.896082
 H   11.486055  -10.038374   0.978324
 C   9.353621  -9.808691   1.160281
 H   9.186897  -10.838206   1.456373
 C   8.272889  -8.937313   1.048342
 H   7.278883  -9.302758   1.273852
 C   8.471932  -7.599852   0.679531
 C   7.354707  -6.605925   0.559791
 O   7.626707  -5.391612   0.548174
 C   5.988347  -7.076442   0.441874
 C   4.919276  -6.227338   0.348837
 H   5.780901  -8.134670   0.394252
 N   5.008810  -4.877900   0.361340
 H   3.923688  -6.642990   0.250801
 H   5.972966  -4.540542   0.424628
 H   0.367502   11.612166  -0.961981
 C   1.329859   12.030547  -0.694838
 C   1.547899   13.403028  -0.724935
 H   0.749018   14.070216  -1.028567
 C   2.791648   13.922286  -0.361952
 H   2.961418   14.993041  -0.381170
 C   3.814324   13.060075   0.030347
 H   4.778553   13.458754   0.324768
 C   3.599728   11.684063   0.048909
 H   4.401771   11.032527   0.372422
 C   2.355239   11.152557  -0.316112
 C   2.052655   9.683297  -0.300126
 O   0.865125   9.312095  -0.325079
 C   3.143490   8.728411  -0.278488
 C   2.941549   7.375353  -0.246950
 H   4.166184   9.071623  -0.315992
 N   1.726510   6.781716  -0.226561
 H   3.800301   6.715037  -0.242618
 H   0.952720   7.450835  -0.257040
 H  -10.228590  -5.553196  -0.656611
 C  -11.076594  -4.907694  -0.464896
 C  -12.373868  -5.406826  -0.474806
 H  -12.547202  -6.456027  -0.686168
 C  -13.451604  -4.560274  -0.209972
 H  -14.463593  -4.949599  -0.213390
 C  -13.222766  -3.212769   0.063940
 H  -14.055183  -2.553325   0.282062
 C  -11.923927  -2.709959   0.061917
 H  -11.766161  -1.663990   0.293186
 C  -10.835159  -3.551242  -0.205165
 C  -9.411579  -3.078026  -0.204396
 O  -8.496884  -3.918768  -0.132417
 C  -9.129037  -1.659670  -0.307467
 C  -7.856587  -1.156371  -0.296277
 H  -9.935896  -0.952654  -0.426665
 N  -6.736672  -1.906260  -0.184074
 H  -7.712817  -0.086551  -0.387554
 H  -6.930156  -2.909586  -0.126279





