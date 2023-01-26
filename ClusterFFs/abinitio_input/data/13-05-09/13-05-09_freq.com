%nproc=15
%mem=70GB
%chk=TrisPhenyl_Ketoenamine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from TrisPhenyl_Ketoenamine_Ketoenamine.log

0 1
 C   5.415300  -1.829499  -0.011901
 C   5.026510  -0.741084   0.779943
 C   3.708526  -0.299395   0.785434
 C   2.734445  -0.924462  -0.005858
 C   3.126970  -2.016401  -0.796035
 C   4.440172  -2.463872  -0.793464
 C   1.329609  -0.448384  -0.003393
 C   0.267971  -1.360203  -0.013161
 C  -1.062463  -0.926203  -0.008910
 C  -1.321341   0.449109   0.004797
 C  -0.280269   1.384252   0.014633
 C   1.040248   0.920731   0.010116
 C  -0.570107   2.838883   0.030920
 C   0.187387   3.736073  -0.738540
 C  -0.080998   5.097041  -0.718205
 C  -1.125387   5.613220   0.060850
 C  -1.881898   4.721022   0.831883
 C  -1.606061   3.358669   0.819788
 C  -2.177147  -1.904795  -0.017438
 C  -3.325957  -1.686187  -0.794120
 C  -4.369886  -2.599918  -0.797406
 C  -4.300975  -3.774349  -0.035768
 C  -3.157095  -3.994480   0.742629
 C  -2.115810  -3.073723   0.754277
 H   0.480253  -2.422550  -0.024340
 H   1.854254   1.635644   0.017065
 H  -2.347545   0.796473   0.007687
 H   0.978828   3.357301  -1.375268
 H  -2.183242   2.692802   1.451051
 H   0.506539   5.786168  -1.311964
 H  -2.674095   5.084771   1.474235
 H   4.738873  -3.307340  -1.403374
 H   2.398320  -2.502592  -1.434766
 H   5.742445  -0.246892   1.424680
 H   3.425126   0.523043   1.432201
 H  -1.256245  -3.249816   1.390793
 H  -3.081222  -4.872129   1.372285
 H  -5.255038  -2.427117  -1.396813
 H  -3.388410  -0.801795  -1.418003
 C   6.814399  -2.366843  -0.051500
 O   7.011109  -3.495674  -0.540844
 C   7.903612  -1.561663   0.457624
 C   9.201707  -2.000885   0.472307
 H   7.719765  -0.559609   0.814328
 N   9.609630  -3.208035   0.028605
 H   9.975898  -1.342827   0.849832
 C   10.910777  -3.731040   0.044218
 H   8.851923  -3.769325  -0.368474
 C   11.956913  -3.155379   0.777800
 C   11.158999  -4.892332  -0.702201
 C   13.226646  -3.725482   0.742741
 H   11.787447  -2.278937   1.390247
 C   13.473864  -4.875966  -0.004149
 H   14.025900  -3.269040   1.315808
 C   12.428513  -5.457461  -0.721822
 H   10.351442  -5.341971  -1.270002
 H   14.463716  -5.315833  -0.022149
 H   12.601546  -6.354676  -1.305174
 C  -1.358257   7.094143   0.041219
 O  -0.473330   7.835555  -0.427421
 C  -2.605220   7.627985   0.545140
 C  -2.873032   8.971638   0.577685
 H  -3.385426   6.962939   0.882995
 N  -2.026112   9.934464   0.157957
 H  -3.833830   9.308168   0.949930
 C  -2.222799   11.322576   0.193707
 H  -1.157370   9.564397  -0.235956
 C  -3.249320   11.929958   0.929408
 C  -1.335161   12.129072  -0.533608
 C  -3.389377   13.315107   0.914905
 H  -3.928220   11.335941   1.527917
 C  -2.510657   14.115350   0.186873
 H  -4.188145   13.770659   1.489356
 C  -1.479461   13.511321  -0.532666
 H  -0.538143   11.662852  -1.102924
 H  -2.623815   15.192759   0.184876
 H  -0.784280   14.118312  -1.101455
 C  -5.465869  -4.717058  -0.081228
 O  -6.546275  -4.315055  -0.554056
 C  -5.308129  -6.071555   0.402838
 C  -6.336748  -6.977172   0.409718
 H  -4.345276  -6.418783   0.745610
 N  -7.589887  -6.719798  -0.019057
 H  -6.150087  -7.983074   0.767792
 C  -8.692783  -7.585956  -0.011236
 H  -7.700879  -5.775720  -0.397569
 C  -8.709015  -8.796601   0.694463
 C  -9.830814  -7.203611  -0.736571
 C  -9.837843  -9.610540   0.652968
 H  -7.858368  -9.102340   1.290168
 C  -10.966040  -9.232641  -0.072943
 H  -9.835623  -10.544094   1.204352
 C  -10.954971  -8.020294  -0.762882
 H  -9.822921  -6.266406  -1.282755
 H  -11.841934  -9.869732  -0.096116
 H  -11.824877  -7.708420  -1.329659





