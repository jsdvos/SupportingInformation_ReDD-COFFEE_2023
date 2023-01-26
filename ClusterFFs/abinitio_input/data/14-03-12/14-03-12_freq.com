%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_PrimaryAmine_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from Triazine-TrisPhenyl_PrimaryAmine_Imide.log

0 1
 C   4.014802  -3.876798   0.305846
 C   2.671532  -4.262633   0.306789
 C   1.678079  -3.297225   0.225440
 C   2.005122  -1.936830   0.153425
 C   3.355729  -1.564492   0.159215
 C   4.356052  -2.523691   0.229052
 C   0.941101  -0.909738   0.072112
 N   1.311904   0.376698   0.034883
 C   0.317109   1.270371  -0.039227
 N  -0.981977   0.946596  -0.075569
 C  -1.259910  -0.362893  -0.033407
 N  -0.331644  -1.325529   0.040215
 C   0.676065   2.706619  -0.084189
 C  -0.318946   3.686333  -0.197952
 C   0.013107   5.032787  -0.246769
 C   1.354022   5.419211  -0.170244
 C   2.356896   4.453189  -0.050702
 C   2.016370   3.108456  -0.014158
 C  -2.682904  -0.772049  -0.070402
 C  -3.700045   0.190565  -0.110251
 C  -5.034742  -0.188033  -0.138789
 C  -5.370521  -1.544683  -0.139498
 C  -4.366308  -2.516097  -0.105818
 C  -3.034573  -2.128194  -0.065824
 H   0.636644  -3.589165   0.222970
 H   2.410649  -5.310234   0.357770
 H   5.395477  -2.227609   0.237048
 H   3.612852  -0.515475   0.102082
 H  -1.354864   3.380495  -0.254086
 H  -0.760957   5.782382  -0.330745
 H   3.393608   4.754514  -0.000165
 H   2.787726   2.355584   0.075315
 H  -3.433107   1.238685  -0.112252
 H  -5.813185   0.560943  -0.173347
 H  -4.629164  -3.564417  -0.097983
 H  -2.252518  -2.874762  -0.036790
 C   6.274201  -6.797135   0.055583
 C   6.754770  -8.031548  -0.349553
 H   6.196648  -8.643228  -1.047790
 C   7.980786  -8.447637   0.180109
 H   8.391339  -9.407605  -0.110877
 C   8.687989  -7.644643   1.082210
 H   9.635307  -7.995190   1.475238
 C   8.192505  -6.399227   1.482492
 H   8.729748  -5.767362   2.179131
 C   6.978318  -5.997909   0.949953
 C   6.205861  -4.746770   1.180617
 C   5.020502  -6.092567  -0.327750
 N   5.037413  -4.863973   0.383088
 O   4.154955  -6.460663  -1.082451
 O   6.485853  -3.813631   1.891313
 C   2.768513   8.797135  -0.704171
 C   3.610578   9.787623  -1.182436
 H   4.437442   9.544948  -1.838630
 C   3.347044   11.101773  -0.781992
 H   3.982921   11.906071  -1.133457
 C   2.274662   11.396996   0.067446
 H   2.096653   12.425339   0.360009
 C   1.430620   10.387717   0.542583
 H   0.596736   10.602207   1.199709
 C   1.701156   9.090857   0.137807
 C   0.994582   7.823933   0.470777
 C   2.791795   7.329300  -0.949336
 N   1.699046   6.799537  -0.214307
 O   3.561914   6.694963  -1.626669
 O   0.027070   7.668046   1.173531
 C  -8.992872  -2.054344  -0.700798
 C  -10.261405  -1.865053  -1.224128
 H  -10.449624  -1.092110  -1.959193
 C  -11.277610  -2.708731  -0.763318
 H  -12.284569  -2.592888  -1.147218
 C  -11.016601  -3.701642   0.187844
 H  -11.825437  -4.339403   0.525200
 C  -9.730719  -3.883636   0.707632
 H  -9.514576  -4.648570   1.443392
 C  -8.732913  -3.042917   0.242390
 C  -7.289279  -2.991406   0.600802
 C  -7.726880  -1.326997  -0.989673
 N  -6.738169  -1.937862  -0.174641
 O  -7.546757  -0.410134  -1.752010
 O  -6.686418  -3.683223   1.383231






