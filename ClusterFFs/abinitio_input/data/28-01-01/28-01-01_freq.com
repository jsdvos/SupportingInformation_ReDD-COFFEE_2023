%nproc=15
%mem=70GB
%chk=TetraPhenylMethane_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from TetraPhenylMethane_BoronicAcid_BoronateEster.log

0 1
 C   0.000031  -0.000143   0.000186
 C   2.897979   2.350322   2.300237
 C  -2.898387  -2.300202   2.350036
 C   2.897812  -2.350881  -2.299773
 C  -2.897366   2.300570  -2.350250
 C   2.072021   1.420156   2.943370
 C  -2.071863  -2.943366   1.420395
 C   2.071844  -1.420744  -2.942935
 C  -2.070913   2.943536  -1.420405
 C   1.123257   0.684130   2.235675
 C  -1.122952  -2.235618   0.684612
 C   1.123134  -0.684645  -2.235246
 C  -1.122326   2.235564  -0.684421
 C   0.964919   0.856591   0.858357
 C  -0.964995  -0.858233   0.856833
 C   0.964854  -0.857011  -0.857913
 C  -0.964658   0.858146  -0.856623
 C   1.810429   1.769368   0.206937
 C  -1.811103  -0.206775   1.769032
 C   1.810377  -1.769753  -0.206460
 C  -1.810657   0.206893  -1.769066
 C   2.749175   2.507259   0.911091
 C  -2.750014  -0.910975   2.506670
 C   2.749077  -2.507711  -0.910606
 C  -2.749243   0.911317  -2.506905
 H   2.173686   1.265613   4.012144
 H  -2.173195  -4.012204   1.266075
 H   2.173456  -1.266285  -4.011726
 H  -2.172041   4.012393  -1.266091
 H   0.512551  -0.031792   2.768555
 H  -0.511772  -2.768514  -0.030902
 H   0.512396   0.031245  -2.768140
 H  -0.511170   2.768306   0.031232
 H   1.733855   1.891118  -0.867001
 H  -1.734899   0.867219   1.890501
 H   1.733867  -1.891410   0.867493
 H  -1.734627  -0.867111  -1.890571
 H   3.384662   3.209059   0.382072
 H  -3.385972  -0.381929   3.208023
 H   3.384581  -3.209479  -0.381567
 H  -3.385134   0.382433  -3.208439
 B   3.941042   3.161376   3.083351
 O   4.157403   3.060398   4.456857
 O   4.801808   4.100915   2.517892
 C   5.169042   3.952000   4.739943
 C   5.560836   4.584524   3.561177
 C   5.759285   4.237904   5.955614
 C   6.562864   5.534974   3.538045
 C   6.777413   5.200510   5.946599
 C   7.170344   5.834638   4.764588
 H   5.445909   3.739574   6.864018
 H   6.858077   6.018657   2.615745
 H   7.269437   5.457142   6.877207
 H   7.961646   6.574162   4.794985
 B  -3.941619  -3.083368   3.160824
 O  -4.157596  -4.456953   3.060099
 O  -4.802943  -2.517882   4.099835
 C  -5.169561  -4.740062   3.951323
 C  -5.561928  -3.561232   4.583373
 C  -5.759630  -5.955806   4.237274
 C  -6.564384  -3.538106   5.533372
 C  -6.778188  -5.946798   5.199425
 C  -7.171693  -4.764723   5.833078
 H  -5.445809  -6.864259   3.739314
 H  -6.860040  -2.615756   6.016688
 H  -7.270096  -6.877462   5.456075
 H  -7.963315  -4.795127   6.572259
 B   3.940818  -3.162017  -3.082879
 O   4.157114  -3.061144  -4.456403
 O   4.801591  -4.101533  -2.517393
 C   5.168718  -3.952790  -4.739474
 C   5.560557  -4.585236  -3.560680
 C   5.758895  -4.238796  -5.955153
 C   6.562565  -5.535706  -3.537528
 C   6.777002  -5.201424  -5.946117
 C   7.169978  -5.835474  -4.764079
 H   5.445485  -3.740525  -6.863578
 H   6.857814  -6.019327  -2.615207
 H   7.268975  -5.458135  -6.876731
 H   7.961262  -6.575018  -4.794460
 B  -3.940237   3.083984  -3.161262
 O  -4.155944   4.457612  -3.060546
 O  -4.801460   2.518710  -4.100493
 C  -5.167638   4.740964  -3.952001
 C  -5.560107   3.562237  -4.584179
 C  -5.757380   5.956843  -4.238055
 C  -6.562344   3.539354  -5.534415
 C  -6.775712   5.948081  -5.200447
 C  -7.169320   4.766109  -5.834229
 H  -5.443482   6.865214  -3.739994
 H  -6.858083   2.617081  -6.017829
 H  -7.267361   6.878858  -5.457185
 H  -7.960762   4.796705  -6.573595





