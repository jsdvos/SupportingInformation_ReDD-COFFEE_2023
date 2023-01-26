%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_Ketoenamine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Triazine-TrisPhenyl_Ketoenamine_Ketoenamine.log

0 1
 C   4.044033  -3.869545   0.325540
 C   2.693785  -4.246063   0.311870
 C   1.690986  -3.291522   0.235162
 C   2.011691  -1.926890   0.184657
 C   3.359390  -1.545233   0.214938
 C   4.361211  -2.504945   0.280480
 C   0.942639  -0.903368   0.103856
 N   1.308082   0.384701   0.065978
 C   0.310106   1.274448  -0.008379
 N  -0.987726   0.945177  -0.045073
 C  -1.260670  -0.365097  -0.002234
 N  -0.328275  -1.323881   0.071885
 C   0.663608   2.713171  -0.054237
 C  -0.338486   3.688234  -0.143477
 C  -0.006345   5.035803  -0.196272
 C   1.333442   5.445417  -0.152851
 C   2.331895   4.467207  -0.046625
 C   2.005056   3.120318  -0.005507
 C  -2.682958  -0.780604  -0.040178
 C  -3.705091   0.177770  -0.106324
 C  -5.034560  -0.214781  -0.139297
 C  -5.382967  -1.572526  -0.122454
 C  -4.359563  -2.527685  -0.051205
 C  -3.027156  -2.138281  -0.006508
 H   0.650572  -3.587325   0.215340
 H   2.453236  -5.300452   0.362777
 H   5.393138  -2.179689   0.319261
 H   3.608393  -0.492841   0.189034
 H  -1.374176   3.376875  -0.168166
 H  -0.802540   5.767616  -0.248463
 H   3.364408   4.789218   0.002632
 H   2.779534   2.368453   0.066767
 H  -3.440609   1.226527  -0.128925
 H  -5.829017   0.519730  -0.178411
 H  -4.596050  -3.583275  -0.006893
 H  -2.240903  -2.878491   0.057448
 C   5.079625  -4.953527   0.412997
 O   4.726958  -6.090367   0.779873
 C   6.447917  -4.660907   0.050387
 C   7.446350  -5.598092   0.123363
 H   6.713749  -3.686131  -0.329439
 N   7.275959  -6.869666   0.538064
 H   8.449467  -5.323632  -0.182072
 C   8.253337  -7.868221   0.664831
 H   6.303516  -7.094552   0.763341
 C   9.628196  -7.602150   0.615925
 C   7.819275  -9.186947   0.863259
 C   10.542563  -8.644238   0.743813
 H   9.993209  -6.589841   0.497852
 C   10.110433  -9.955513   0.934840
 H   11.603270  -8.423249   0.704569
 C   8.741849  -10.217483   0.998738
 H   6.755682  -9.396327   0.903805
 H   10.828775  -10.759808   1.038236
 H   8.387893  -11.230767   1.150841
 C   1.755800   6.886019  -0.189056
 O   2.905494   7.182505   0.187249
 C   0.833066   7.885499  -0.677835
 C   1.146924   9.219678  -0.722010
 H  -0.132291   7.593149  -1.062255
 N   2.321101   9.745297  -0.318334
 H   0.419838   9.918776  -1.118855
 C   2.696717   11.097091  -0.308596
 H   2.992765   9.040554  -0.003463
 C   1.783148   12.144987  -0.483881
 C   4.050149   11.398536  -0.098542
 C   2.227671   13.464278  -0.468016
 H   0.727331   11.944166  -0.613795
 C   3.573909   13.762988  -0.265383
 H   1.509286   14.264776  -0.604406
 C   4.480345   12.719872  -0.075550
 H   4.759873   10.589893   0.039662
 H   3.910585   14.792551  -0.249661
 H   5.530486   12.933884   0.087440
 C  -6.841372  -1.929030  -0.156555
 O  -7.679602  -1.048419   0.114011
 C  -7.237214  -3.270401  -0.521697
 C  -8.548963  -3.668715  -0.554243
 H  -6.494584  -3.995120  -0.819010
 N  -9.598562  -2.878464  -0.251280
 H  -8.784047  -4.682970  -0.855821
 C  -10.957129  -3.228662  -0.233535
 H  -9.329807  -1.916914  -0.027489
 C  -11.403683  -4.556211  -0.273250
 C  -11.899362  -2.192742  -0.157599
 C  -12.768461  -4.831006  -0.258025
 H  -10.698869  -5.377596  -0.296417
 C  -13.704607  -3.800711  -0.188995
 H  -13.099313  -5.863032  -0.288588
 C  -13.258934  -2.480090  -0.133418
 H  -11.557138  -1.163917  -0.124478
 H  -14.764652  -4.023503  -0.172520
 H  -13.972912  -1.666446  -0.075237





