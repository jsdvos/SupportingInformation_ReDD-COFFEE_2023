%nproc=15
%mem=70GB
%chk=PMDA_Nitrile_Triazine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from PMDA_Nitrile_Triazine.log

0 1
 C   2.664346  -0.787453  -0.577886
 C   1.218377  -0.493063  -0.356718
 C   0.105851  -1.161669  -0.853274
 H   0.185148  -2.028753  -1.496730
 C  -1.116397  -0.631739  -0.457549
 C  -2.493059  -1.093950  -0.800419
 C  -1.218578   0.486005   0.371166
 C  -2.664553   0.780378   0.592316
 C  -0.106059   1.154612   0.867714
 H  -0.185337   2.021722   1.511139
 C   1.116189   0.624662   0.472011
 C   2.492852   1.086800   0.814971
 N  -3.374760  -0.204126  -0.138341
 N   3.374567   0.197098   0.152709
 O  -3.148414   1.663989   1.252148
 O  -2.810580  -2.025226  -1.494799
 O   2.810403   2.018238   1.509109
 O   3.148184  -1.671023  -1.237779
 C  -4.796237  -0.288503  -0.199596
 C   4.796029   0.281584   0.213995
 C  -5.555001   0.874260  -0.355492
 C  -6.938733   0.788586  -0.410042
 H  -5.065706   1.835813  -0.420824
 C  -7.581301  -0.453589  -0.319038
 H  -7.534476   1.683767  -0.528324
 C  -6.806884  -1.611794  -0.167371
 C  -5.423122  -1.533421  -0.103189
 H  -7.300835  -2.571559  -0.095868
 H  -4.831269  -2.430893   0.008633
 C   5.422798   1.526572   0.117724
 C   6.806531   1.605105   0.182128
 H   4.830840   2.423967   0.005784
 C   7.581042   0.446965   0.333854
 H   7.300398   2.564911   0.110668
 C   6.938609  -0.795309   0.424521
 C   5.554890  -0.881133   0.369756
 H   7.534463  -1.690443   0.542557
 H   5.065657  -1.842731   0.434933
 C  -9.055563  -0.540916  -0.381977
 N  -9.626335  -1.754585  -0.283267
 C  -10.953988  -1.766966  -0.346470
 N  -9.754872   0.597553  -0.534824
 C  -11.075292   0.453178  -0.584411
 N  -11.743551  -0.700047  -0.496478
 H  -11.665160   1.357217  -0.707852
 H  -11.441601  -2.734794  -0.268935
 C   9.055282   0.534360   0.397266
 N   9.625930   1.748144   0.299662
 C   10.953577   1.760622   0.363044
 N   9.754684  -0.604202   0.549347
 C   11.075054  -0.459717   0.599326
 N   11.743206   0.693661   0.512302
 H   11.664996  -1.363798   0.722088
 H   11.441077   2.728564   0.286278






