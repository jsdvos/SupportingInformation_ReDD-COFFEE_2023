%nproc=15
%mem=70GB
%chk=F_PrimaryAmine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from F_PrimaryAmine_Ketoenamine.log

0 1
 F   1.126661  -2.475095   0.176926
 C   0.605817  -1.234122   0.091316
 C  -0.772579  -1.115577   0.119649
 F  -1.510863  -2.233014   0.236860
 C  -1.440451   0.114832   0.028968
 C  -0.605964   1.234592  -0.091277
 F  -1.126801   2.475568  -0.176886
 C   0.772432   1.116046  -0.119610
 F   1.510714   2.233483  -0.236820
 C   1.440304  -0.114364  -0.028930
 H  -7.164757  -2.467526   0.065815
 C  -7.801820  -1.601839   0.197011
 C  -9.183824  -1.732829   0.264360
 H  -9.638156  -2.713290   0.176586
 C  -9.985667  -0.605033   0.447538
 H  -11.063703  -0.706815   0.503868
 C  -9.396390   0.653002   0.562430
 H  -10.013870   1.530774   0.715665
 C  -8.012335   0.786014   0.483809
 H  -7.574321   1.770354   0.590604
 C  -7.198480  -0.339985   0.297798
 C  -5.703499  -0.269078   0.221083
 O  -5.041745  -1.316221   0.338413
 C  -5.052540   1.007923  -0.006453
 C  -3.695406   1.147418  -0.072892
 H  -5.640918   1.900061  -0.157722
 N  -2.825908   0.109631   0.073274
 H  -3.264174   2.120966  -0.252751
 H  -3.299973  -0.791659   0.214258
 H   7.165346   2.467421  -0.065798
 C   7.802207   1.601590  -0.197020
 C   9.184239   1.732267  -0.264435
 H   9.638797   2.712625  -0.176687
 C   9.985816   0.604288  -0.447646
 H   11.063873   0.705825  -0.504028
 C   9.396249  -0.653615  -0.562504
 H   10.013523  -1.531528  -0.715763
 C   8.012168  -0.786314  -0.483818
 H   7.573923  -1.770555  -0.590583
 C   7.198579   0.339872  -0.297775
 C   5.703592   0.269294  -0.220979
 O   5.042092   1.316598  -0.338331
 C   5.052389  -1.007606   0.006506
 C   3.695231  -1.146980   0.072923
 H   5.640645  -1.899827   0.157756
 N   2.825760  -0.109173  -0.073237
 H   3.263936  -2.120505   0.252760
 H   3.299852   0.792106  -0.214230






