%nproc=15
%mem=70GB
%chk=TetraPhenylMethane_Aldehyde_Azine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TetraPhenylMethane_Aldehyde_Azine.log

0 1
 C  -0.000573   0.000000  -0.000000
 C  -2.921005  -3.241499   0.349758
 C   2.919993   0.349790   3.241391
 C  -2.921005   3.241499  -0.349758
 C   2.919994  -0.349789  -3.241390
 C  -2.148697  -2.867188   1.457903
 C   2.147697   1.457940   2.867074
 C  -2.148699   2.867188  -1.457903
 C   2.147698  -1.457939  -2.867074
 C  -1.194062  -1.863532   1.350663
 C   1.193001   1.350684   1.863475
 C  -1.194063   1.863531  -1.350663
 C   1.193002  -1.350684  -1.863475
 C  -0.969216  -1.200597   0.137652
 C   0.968093   0.137655   1.200594
 C  -0.969216   1.200597  -0.137652
 C   0.968094  -0.137655  -1.200594
 C  -1.763474  -1.558804  -0.960460
 C   1.762317  -0.960472   1.558828
 C  -1.763473   1.558804   0.960460
 C   1.762318   0.960473  -1.558828
 C  -2.713829  -2.564205  -0.861738
 C   2.712736  -0.861731   2.564166
 C  -2.713829   2.564205   0.861737
 C   2.712736   0.861732  -2.564166
 H  -2.308997  -3.370478   2.403271
 H   2.308032   2.403319   3.370332
 H  -2.308998   3.370477  -2.403271
 H   2.308034  -2.403318  -3.370331
 H  -0.623662  -1.593182   2.228989
 H   0.622613   2.229019   1.593128
 H  -0.623663   1.593181  -2.228990
 H   0.622614  -2.229018  -1.593128
 H  -1.641697  -1.035284  -1.900942
 H   1.640490  -1.900977   1.035360
 H  -1.641697   1.035285   1.900942
 H   1.640490   1.900978  -1.035360
 H  -3.310504  -2.823931  -1.730158
 H   3.309375  -1.730167   2.823920
 H  -3.310503   2.823931   1.730158
 H   3.309376   1.730168  -2.823919
 C  -3.931512  -4.293590   0.418304
 N  -4.164776  -4.953990   1.495052
 H  -4.500590  -4.514405  -0.489507
 N  -5.165035  -5.904473   1.334525
 C  -5.402088  -6.561265   2.412385
 C  -6.414463  -7.613367   2.482941
 H  -4.834587  -6.338169   3.320664
 C  -6.612337  -8.285534   3.698165
 C  -7.567267  -9.293381   3.804822
 H  -6.013267  -8.014427   4.561518
 C  -8.336708  -9.641605   2.696451
 H  -7.710428  -9.805114   4.749682
 C  -8.147216  -8.977541   1.480591
 H  -9.081096  -10.425837   2.776424
 C  -7.196086  -7.972414   1.371029
 H  -8.746070  -9.248391   0.618242
 H  -7.041740  -7.451902   0.434152
 C   3.930573   0.418348   4.293411
 N   4.164188   1.495218   4.953489
 H   4.499710  -0.489443   4.514153
 N   5.164699   1.334759   5.903714
 C   5.402240   2.412777   6.560071
 C   6.415218   2.483510   7.611580
 H   4.835019   3.321173   6.336737
 C   6.613774   3.698956   8.283144
 C   7.569290   3.805777   9.290419
 H   6.014749   4.562342   8.012044
 C   8.338657   2.697356   9.638651
 H   7.712968   4.750805   9.801697
 C   8.148484   1.481275   8.975188
 H   9.083505   2.777459   10.422432
 C   7.196760   1.371546   7.970642
 H   8.747272   0.618884   9.246052
 H   7.041858   0.434487   7.450622
 C  -3.931513   4.293590  -0.418304
 N  -4.164777   4.953989  -1.495052
 H  -4.500591   4.514404   0.489507
 N  -5.165037   5.904471  -1.334526
 C  -5.402091   6.561264  -2.412385
 C  -6.414466   7.613365  -2.482941
 H  -4.834590   6.338166  -3.320664
 C  -6.612340   8.285532  -3.698166
 C  -7.567271   9.293378  -3.804823
 H  -6.013271   8.014424  -4.561518
 C  -8.336711   9.641602  -2.696451
 H  -7.710432   9.805111  -4.749683
 C  -8.147219   8.977538  -1.480591
 H  -9.081100   10.425833  -2.776424
 C  -7.196089   7.972412  -1.371029
 H  -8.746073   9.248389  -0.618242
 H  -7.041742   7.451900  -0.434152
 C   3.930575  -0.418347  -4.293410
 N   4.164189  -1.495217  -4.953488
 H   4.499711   0.489445  -4.514152
 N   5.164701  -1.334758  -5.903713
 C   5.402241  -2.412776  -6.560070
 C   6.415220  -2.483508  -7.611579
 H   4.835021  -3.321171  -6.336736
 C   6.613776  -3.698955  -8.283144
 C   7.569293  -3.805775  -9.290418
 H   6.014751  -4.562341  -8.012044
 C   8.338660  -2.697354  -9.638650
 H   7.712971  -4.750803  -9.801697
 C   8.148486  -1.481273  -8.975187
 H   9.083507  -2.777457  -10.422432
 C   7.196762  -1.371544  -7.970641
 H   8.747274  -0.618882  -9.246051
 H   7.041860  -0.434485  -7.450622





