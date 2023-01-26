%nproc=15
%mem=70GB
%chk=Triazine-TrisPhenyl_PrimaryAmine_Imine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Triazine-TrisPhenyl_PrimaryAmine_Imine.log

0 1
 C   4.817268   2.869522  -0.312350
 C   4.822328   1.464034  -0.308874
 C   3.628280   0.757070  -0.315839
 C   2.397767   1.428441  -0.325282
 C   2.395870   2.832261  -0.347079
 C   3.585829   3.541924  -0.358442
 C   1.126969   0.671718  -0.327818
 N  -0.018435   1.366959  -0.328137
 C  -1.144851   0.640587  -0.328218
 N  -1.174246  -0.698995  -0.327920
 C   0.018003  -1.311316  -0.327692
 N   1.192802  -0.666971  -0.327599
 C  -2.435645   1.362663  -0.325614
 C  -3.650300   0.658875  -0.346440
 C  -4.860012   1.334299  -0.357620
 C  -4.893675   2.736982  -0.312423
 C  -3.679142   3.444354  -0.309987
 C  -2.469729   2.764019  -0.317027
 C   0.038022  -2.790222  -0.325053
 C  -1.158572  -3.520384  -0.316420
 C  -1.143109  -4.907944  -0.309462
 C   0.076730  -5.606097  -0.311943
 C   1.274685  -4.875646  -0.357159
 C   1.254828  -3.490286  -0.345989
 H   3.633941  -0.324993  -0.320877
 H   5.766792   0.932621  -0.333334
 H   3.587224   4.624937  -0.390103
 H   1.447892   3.353371  -0.365608
 H  -3.627355  -0.422659  -0.364261
 H  -5.798514   0.793758  -0.388425
 H  -3.691382   4.527973  -0.335092
 H  -1.535587   3.310152  -0.322770
 H  -2.098594  -2.984428  -0.322135
 H  -2.075437  -5.460346  -0.334520
 H   2.212027  -5.418192  -0.388014
 H   2.180009  -2.929689  -0.363879
 N   5.991155   3.633392  -0.329961
 C   6.977072   3.316864   0.417162
 C   8.257079   4.033191   0.394779
 C   9.278270   3.636886   1.269039
 C   10.505552   4.295386   1.269392
 C   10.722769   5.357842   0.394279
 C   9.709129   5.760228  -0.481033
 C   8.485341   5.104849  -0.483546
 H   6.907610   2.488401   1.136083
 H   9.107887   2.809336   1.950551
 H   11.288947   3.980916   1.949395
 H   11.677065   5.872498   0.391957
 H   9.879674   6.586966  -1.161418
 H   7.690935   5.404732  -1.155940
 N  -6.142336   3.371311  -0.329703
 C  -6.360998   4.384105   0.416529
 C  -7.621812   5.133698   0.394488
 C  -7.788938   6.217341   1.267430
 C  -8.973405   6.950073   1.268170
 C  -10.002933   6.604942   0.394798
 C  -9.844850   5.524790  -0.479180
 C  -8.664829   4.793544  -0.482104
 H  -5.608277   4.739501   1.134249
 H  -6.986462   6.485088   1.947632
 H  -9.092572   7.786610   1.947144
 H  -10.926221   7.173358   0.392800
 H  -10.646748   5.257523  -1.158175
 H  -8.527555   3.954708  -1.153394
 N   0.151684  -7.004629  -0.329209
 C  -0.616391  -7.700338   0.416755
 C  -0.635572  -9.167025   0.394411
 C  -1.491267  -9.853489   1.266646
 C  -1.534051  -11.245617   1.267070
 C  -0.720057  -11.964743   0.394090
 C   0.137115  -11.287858  -0.479205
 C   0.180836  -9.900324  -0.481812
 H  -1.300990  -7.226080   1.133989
 H  -2.124698  -9.292331   1.946497
 H  -2.199564  -11.767004   1.945489
 H  -0.751027  -13.048531   0.391855
 H   0.769777  -11.848751  -1.157912
 H   0.839228  -9.362122  -1.152616






