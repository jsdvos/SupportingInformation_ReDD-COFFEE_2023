%nproc=15
%mem=70GB
%chk=ETTA_Aldehyde_Benzobisoxazole_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from ETTA_Aldehyde_Benzobisoxazole.log

0 1
 C  -0.017885   0.681611  -0.091603
 C   0.018041  -0.681588  -0.091570
 C  -1.287546   1.446507  -0.220767
 C  -2.229741   1.133287  -1.214208
 H  -2.031170   0.312798  -1.892463
 C  -3.406181   1.854280  -1.336676
 H  -4.126727   1.609771  -2.106989
 C  -3.677982   2.915933  -0.460092
 C  -2.737286   3.247793   0.524935
 H  -2.935978   4.069595   1.200988
 C  -1.554707   2.528513   0.630793
 H  -0.829057   2.800556   1.388492
 C  -1.210198  -1.511252   0.037669
 C  -2.163048  -1.250373   1.032912
 H  -2.005118  -0.423538   1.714136
 C  -3.304185  -2.030420   1.153169
 H  -4.029012  -1.816229   1.928074
 C  -3.524008  -3.098780   0.272472
 C  -2.568638  -3.378946  -0.716442
 H  -2.738417  -4.207698  -1.392311
 C  -1.425342  -2.602507  -0.820208
 H  -0.688897  -2.832808  -1.581377
 C   1.210312   1.511310   0.037609
 C   1.425385   2.602616  -0.820219
 H   0.688914   2.832920  -1.581364
 C   2.568644   3.379112  -0.716451
 H   2.738343   4.207903  -1.392292
 C   3.524061   3.098956   0.272414
 C   3.304319   2.030535   1.153066
 H   4.029181   1.816341   1.927938
 C   2.163223   1.250441   1.032812
 H   2.005354   0.423571   1.714009
 C   1.287663  -1.446523  -0.220697
 C   2.229908  -1.133339  -1.214119
 H   2.031391  -0.312831  -1.892368
 C   3.406306  -1.854389  -1.336584
 H   4.126875  -1.609903  -2.106883
 C   3.678038  -2.916084  -0.460019
 C   2.737311  -3.247901   0.524986
 H   2.935936  -4.069725   1.201033
 C   1.554770  -2.528556   0.630844
 H   0.829107  -2.800577   1.388540
 H  -8.910354   7.337508   0.284146
 C  -8.310809   6.524402  -0.107884
 C  -8.793782   5.777516  -1.196673
 H  -9.757977   6.032022  -1.621076
 C  -8.064890   4.722873  -1.740219
 H  -8.432738   4.144857  -2.578694
 C  -6.828826   4.432915  -1.160154
 N  -5.875146   3.465671  -1.458094
 C  -6.371247   5.192586  -0.075987
 O  -5.146984   4.694568   0.287540
 C  -7.076620   6.244457   0.481289
 H  -6.693903   6.812288   1.319457
 C  -4.923953   3.656550  -0.596509
 H  -9.556182  -5.893025   2.083891
 C  -8.628317  -5.729727   1.548467
 C  -8.294875  -6.577095   0.477299
 H  -8.975066  -7.377382   0.209990
 C  -7.116171  -6.412166  -0.245467
 H  -6.855471  -7.061942  -1.071479
 C  -6.272333  -5.367606   0.135675
 N  -5.048255  -4.940943  -0.367852
 C  -6.628318  -4.538268   1.206855
 O  -5.630227  -3.611152   1.358445
 C  -7.793230  -4.681756   1.939581
 H  -8.039209  -4.022215   2.761711
 C  -4.718527  -3.925448   0.369601
 H   9.556119   5.893448   2.083825
 C   8.628251   5.730123   1.548414
 C   8.294740   6.577520   0.477290
 H   8.974882   7.377855   0.210001
 C   7.116030   6.412560  -0.245457
 H   6.855276   7.062357  -1.071435
 C   6.272255   5.367937   0.135658
 N   5.048191   4.941232  -0.367860
 C   6.628309   4.538572   1.206795
 O   5.630272   3.611395   1.358366
 C   7.793230   4.682090   1.939501
 H   8.039262   4.022528   2.761598
 C   4.718535   3.925683   0.369553
 H   8.910118  -7.338011   0.284144
 C   8.310625  -6.524860  -0.107874
 C   8.793641  -5.777997  -1.196661
 H   9.757815  -6.032566  -1.621074
 C   8.064817  -4.723300  -1.740191
 H   8.432697  -4.145302  -2.578664
 C   6.828778  -4.433261  -1.160113
 N   5.875160  -3.465953  -1.458038
 C   6.371155  -5.192911  -0.075948
 O   5.146929  -4.694816   0.287591
 C   7.076462  -6.244836   0.481312
 H   6.693712  -6.812648   1.319478
 C   4.923958  -3.656777  -0.596448





