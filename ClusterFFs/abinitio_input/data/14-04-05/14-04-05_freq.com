%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_Hydrazide_Hydrazone_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Triazine-TrisPhenyl_Hydrazide_Hydrazone.log

0 1
 C   0.950495  -5.518525  -0.574878
 C  -0.358920  -5.023908  -0.624681
 C  -0.594856  -3.654472  -0.633371
 C   0.474781  -2.751344  -0.590662
 C   1.785821  -3.248464  -0.557241
 C   2.019425  -4.615713  -0.557859
 C   0.223030  -1.290175  -0.591967
 N   1.281681  -0.470736  -0.590596
 C   1.006985   0.839980  -0.591191
 N  -0.231990   1.347107  -0.591413
 C  -1.229764   0.453861  -0.592808
 N  -1.049412  -0.872678  -0.592490
 C   2.146522   1.788592  -0.589193
 C   1.921579   3.172563  -0.556164
 C   2.988940   4.058378  -0.556136
 C   4.305264   3.583960  -0.572194
 C   4.531524   2.202657  -0.621692
 C   3.463451   1.313709  -0.630927
 C  -2.621080   0.966341  -0.592166
 C  -2.868411   2.344329  -0.631411
 C  -4.172377   2.824697  -0.623051
 C  -5.255456   1.937852  -0.577137
 C  -5.007978   0.560692  -0.563656
 C  -3.707115   0.079377  -0.562612
 H  -1.605436  -3.272002  -0.683612
 H  -1.202228  -5.701501  -0.697612
 H   3.028725  -5.008114  -0.549557
 H   2.611919  -2.550560  -0.534066
 H   0.904152   3.539149  -0.533886
 H   2.824282   5.128683  -0.548051
 H   5.539970   1.810907  -0.693889
 H   3.637408   0.247255  -0.680812
 H  -2.031822   3.028423  -0.678502
 H  -4.337358   3.894060  -0.693012
 H  -5.852474  -0.117194  -0.558357
 H  -3.515824  -0.985055  -0.542102
 C   1.285555  -6.984619  -0.587273
 O   2.314564  -7.411572  -1.066664
 N   0.326170  -7.789036   0.009034
 H  -0.429753  -7.354643   0.531281
 N   0.469557  -9.137308   0.042701
 C  -0.400809  -9.806425   0.700625
 C  -0.349126  -11.266013   0.792965
 H  -1.228656  -9.309323   1.225268
 C  -1.334943  -11.939756   1.527377
 C  -1.314026  -13.328370   1.635122
 H  -2.121427  -11.373202   2.016208
 C  -0.306535  -14.059178   1.009424
 H  -2.082277  -13.837378   2.205934
 C   0.680439  -13.394590   0.274975
 H  -0.287565  -15.139999   1.091453
 C   0.663397  -12.010967   0.164663
 H   1.465274  -13.961930  -0.212485
 H   1.424471  -11.488747  -0.402035
 C   5.407426   4.607204  -0.583759
 O   5.262504   5.712369  -1.061902
 N   6.584035   4.177908   0.011430
 H   6.586414   3.305064   0.531961
 N   7.680360   4.975713   0.044611
 C   8.695747   4.555143   0.700523
 C   9.934477   5.328951   0.791980
 H   8.679492   3.588930   1.223840
 C   11.012117   4.810148   1.523183
 C   12.204845   5.521731   1.629749
 H   10.915227   3.844868   2.010366
 C   12.333326   6.760729   1.006060
 H   13.030770   5.109408   2.198044
 C   11.263022   7.285128   0.274851
 H   13.260325   7.316927   1.087167
 C   10.072688   6.579395   0.165710
 H   11.361437   8.249343  -0.211017
 H   9.238937   6.978855  -0.398543
 C  -6.692773   2.380405  -0.589829
 O  -7.576537   1.703047  -1.070612
 N  -6.910650   3.612417   0.008213
 H  -6.156811   4.049965   0.530792
 N  -8.150263   4.161705   0.041962
 C  -8.295306   5.249401   0.700743
 C  -9.585694   5.933438   0.793526
 H  -7.451268   5.717891   1.225919
 C  -9.677182   7.123452   1.528789
 C  -10.890774   7.798555   1.637082
 H  -8.793590   7.521611   2.017904
 C  -12.027017   7.290946   1.011074
 H  -10.948215   8.717918   2.208570
 C  -11.944036   6.104506   0.275749
 H  -12.972952   7.814080   1.093550
 C  -10.736699   5.428558   0.164862
 H  -12.827487   5.708130  -0.211961
 H  -10.664206   4.508826  -0.402525






