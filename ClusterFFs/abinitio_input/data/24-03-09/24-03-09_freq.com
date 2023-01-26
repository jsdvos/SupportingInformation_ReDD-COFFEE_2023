%nproc=15
%mem=70GB
%chk=P_PrimaryAmine_Ketoenamine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from P_PrimaryAmine_Ketoenamine-1.log

0 1
 C   3.947607   1.977798  -0.167394
 C   4.352373   0.686652  -0.160992
 C   3.139285  -0.124963  -0.138879
 C   2.488004   1.951750  -0.167329
 N   2.027531   0.666948  -0.146249
 C   1.698322   3.117086  -0.175820
 C   0.297673   3.146302  -0.160387
 N  -0.545921   2.061033  -0.138128
 C  -1.859083   2.467954  -0.130291
 C  -0.536874   4.311404  -0.173023
 C  -1.842037   3.900999  -0.144181
 C  -2.991024   1.643132  -0.103438
 C  -2.973438   0.235851  -0.082871
 N  -1.861885  -0.556340  -0.092091
 C  -3.781696  -1.866653  -0.039480
 C  -4.186531  -0.575654  -0.058807
 C  -0.132008  -3.035650  -0.075663
 N   0.711602  -1.950382  -0.098171
 C   2.024784  -2.357282  -0.105803
 C   2.007811  -3.790195  -0.082332
 C   0.702446  -4.200881  -0.074841
 C   3.156874  -1.532321  -0.122854
 C  -1.532735  -3.006510  -0.063870
 C  -2.322388  -1.841103  -0.070009
 H   4.562747   2.863414  -0.168590
 H   5.362915   0.310470  -0.168618
 C   2.392798   4.440901  -0.204149
 H  -0.241320   1.094860  -0.128206
 H  -0.170104   5.324127  -0.198801
 H  -2.722797   4.521457  -0.132205
 C  -4.318189   2.332462  -0.093175
 H  -4.396586  -2.751747  -0.004516
 H  -5.197103  -0.199480  -0.055471
 H   0.407021  -0.984188  -0.106734
 H   2.888756  -4.410402  -0.071777
 H   0.335573  -5.213860  -0.067452
 C   4.484123  -2.221465  -0.120404
 C  -2.227337  -4.330338  -0.043398
 C  -4.805785   2.974179  -1.235733
 C  -6.037579   3.620677  -1.237255
 H  -4.208144   2.969291  -2.140207
 C  -6.825856   3.633135  -0.079091
 H  -6.367478   4.116935  -2.140918
 C  -6.341038   2.997772   1.073419
 C  -5.107637   2.360539   1.062270
 H  -6.939317   3.005749   1.978273
 H  -4.748833   1.874170   1.961977
 C   3.052690   4.883231  -1.355074
 C   3.703539   6.111789  -1.394038
 H   3.058705   4.251410  -2.235876
 C   3.706007   6.942238  -0.265375
 H   4.206108   6.407847  -2.305775
 C   3.049157   6.504846   0.894083
 C   2.404645   5.275377   0.919419
 H   3.049899   7.134916   1.777167
 H   1.905936   4.952354   1.825998
 C   4.948681  -2.895189  -1.254127
 C   6.180378  -3.541809  -1.262605
 H   4.333209  -2.914873  -2.146362
 C   6.991671  -3.522870  -0.120507
 H   6.492571  -4.061314  -2.159438
 C   6.530313  -2.854690   1.023096
 C   5.297082  -2.217125   1.018965
 H   7.146502  -2.837962   1.915739
 H   4.956655  -1.705422   1.911733
 C  -2.912459  -4.801584  -1.167833
 C  -3.563449  -6.030667  -1.161354
 H  -2.937301  -4.192765  -2.064362
 C  -3.541724  -6.831762  -0.011868
 H  -4.083307  -6.350776  -2.055076
 C  -2.858855  -6.365560   1.121056
 C  -2.213722  -5.136358   1.100761
 H  -2.840533  -6.972674   2.019898
 H  -1.694940  -4.790822   1.987558
 H  -12.342518   5.475350   2.451858
 C  -12.735141   6.109533   1.666860
 C  -13.962284   6.747469   1.807205
 H  -14.545003   6.604590   2.710331
 C  -14.442512   7.573092   0.789412
 H  -15.397634   8.074423   0.899749
 C  -13.687526   7.755541  -0.368162
 H  -14.049976   8.405277  -1.156792
 C  -12.463472   7.107076  -0.513694
 H  -11.884407   7.274899  -1.413171
 C  -11.973770   6.273844   0.501246
 C  -10.655751   5.561515   0.414954
 O  -10.146474   5.109392   1.457540
 C  -10.015757   5.395808  -0.873253
 C  -8.803997   4.774070  -1.020569
 H  -10.506068   5.737084  -1.772295
 N  -8.077169   4.258523  -0.006312
 H  -8.385364   4.666457  -2.014421
 H  -8.538504   4.343361   0.903200
 H   5.320841   12.632491   2.012921
 C   6.031066   12.967895   1.267502
 C   6.655565   14.204957   1.377915
 H   6.425268   14.853239   2.215953
 C   7.579931   14.610974   0.414138
 H   8.070790   15.573862   0.501693
 C   7.874364   13.771957  -0.659464
 H   8.600647   14.076899  -1.404437
 C   7.239768   12.537608  -0.776398
 H   7.494540   11.893400  -1.608645
 C   6.308153   12.121850   0.184552
 C   5.603903   10.797973   0.127455
 O   5.051055   10.365493   1.156152
 C   5.562906   10.063277  -1.119564
 C   4.952488   8.842505  -1.237465
 H   5.997309   10.484333  -2.013503
 N   4.336540   8.192179  -0.227334
 H   4.945633   8.349765  -2.202537
 H   4.346043   8.715842   0.651912
 H   12.577740  -5.249291   2.336820
 C   12.948511  -5.919931   1.571746
 C   14.179715  -6.551080   1.707112
 H   14.787730  -6.365994   2.585561
 C   14.631551  -7.424062   0.716189
 H   15.589847  -7.920094   0.822894
 C   13.844162  -7.660590  -0.409578
 H   14.184604  -8.347062  -1.176599
 C   12.615938  -7.019086  -0.550759
 H   12.011678  -7.228862  -1.424498
 C   12.154548  -6.138658   0.437239
 C   10.833973  -5.430670   0.354909
 O   10.353048  -4.931510   1.389475
 C   10.158767  -5.323564  -0.921488
 C   8.943100  -4.708323  -1.063900
 H   10.624198  -5.706533  -1.816959
 N   8.244070  -4.146676  -0.054606
 H   8.497229  -4.646771  -2.049742
 H   8.729479  -4.192559   0.845144
 H  -5.052820  -12.485338   2.420895
 C  -5.795002  -12.832095   1.712735
 C  -6.413048  -14.067407   1.869356
 H  -6.145478  -14.702869   2.706141
 C  -7.378749  -14.488104   0.953595
 H  -7.864538  -15.449589   1.077350
 C  -7.720937  -13.665527  -0.118560
 H  -8.479174  -13.981821  -0.826019
 C  -7.093060  -12.433045  -0.282128
 H  -7.384851  -11.801586  -1.111977
 C  -6.120284  -12.002623   0.630366
 C  -5.420332  -10.679658   0.522237
 O  -4.823379  -10.231372   1.519046
 C  -5.434775  -9.963993  -0.736376
 C  -4.830715  -8.745266  -0.899652
 H  -5.909019  -10.397948  -1.603494
 N  -4.171189  -8.079928   0.072456
 H  -4.868007  -8.266701  -1.871112
 H  -4.144566  -8.589103   0.959842






