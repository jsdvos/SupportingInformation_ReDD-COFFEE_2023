%nproc=15
%mem=70GB
%chk=TetraPhenylMethane_BoronicAcid_Boroxine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    -0.000000000000000    0.000000000000000    -0.000000000000000
 C    -0.334711000000000    -3.282421000000000    -2.523411000000000
 C    -3.282421000000000    0.334711000000000    2.523412000000000
 C    0.334711000000000    3.282422000000000    -2.523411000000000
 C    3.282421000000000    -0.334711000000000    2.523412000000000
 C    -1.442915000000000    -2.894269000000000    -1.866366000000000
 C    -2.894268000000000    1.442915000000000    1.866365000000000
 C    1.442915000000000    2.894269000000000    -1.866366000000000
 C    2.894269000000000    -1.442914000000000    1.866365000000000
 C    -1.355721000000000    -1.850757000000000    -1.055712000000000
 C    -1.850756000000000    1.355721000000000    1.055713000000000
 C    1.355722000000000    1.850757000000000    -1.055712000000000
 C    1.850757000000000    -1.355721000000000    1.055713000000000
 C    -0.132197000000000    -1.201023000000000    -0.847056000000000
 C    -1.201023000000000    0.132197000000000    0.847056000000000
 C    0.132198000000000    1.201023000000000    -0.847056000000000
 C    1.201024000000000    -0.132198000000000    0.847056000000000
 C    0.964755000000000    -1.566674000000000    -1.552049000000000
 C    -1.566674000000000    -0.964756000000000    1.552050000000000
 C    -0.964756000000000    1.566674000000000    -1.552049000000000
 C    1.566675000000000    0.964754000000000    1.552050000000000
 C    0.874751000000000    -2.604561000000000    -2.363589000000000
 C    -2.604559000000000    -0.874752000000000    2.363589000000000
 C    -0.874750000000000    2.604559000000000    -2.363589000000000
 C    2.604560000000000    0.874750000000000    2.363589000000000
 H    -2.387983000000000    -3.392116000000000    -1.994223000000000
 H    -3.392115000000000    2.387984000000000    1.994223000000000
 H    2.387983000000000    3.392116000000000    -1.994223000000000
 H    3.392116000000000    -2.387983000000000    1.994223000000000
 H    -2.258597000000000    -1.552610000000000    -0.616202000000000
 H    -1.552610000000000    2.258597000000000    0.616202000000000
 H    2.258598000000000    1.552609000000000    -0.616202000000000
 H    1.552610000000000    -2.258598000000000    0.616202000000000
 H    1.907010000000000    -1.051950000000000    -1.461484000000000
 H    -1.051950000000000    -1.907011000000000    1.461484000000000
 H    -1.907010000000000    1.051952000000000    -1.461484000000000
 H    1.051951000000000    1.907009000000000    1.461484000000000
 H    1.760751000000000    -2.888643000000000    -2.847495000000000
 H    -2.888643000000000    -1.760751000000000    2.847496000000000
 H    -1.760749000000000    2.888644000000000    -2.847495000000000
 H    2.888643000000000    1.760751000000000    2.847496000000000
 B    -0.459514354000000    -4.506111948000000    -3.464159955000000
 O    -1.668688360000000    -5.167855689000000    -3.610950386000000
 B    -1.785360479000000    -6.254250784000000    -4.444150895000000
 O    0.635389805000000    -4.966029308000000    -4.179150563000000
 B    0.530626081000000    -6.051377308000000    -5.015292921000000
 O    -0.682471079000000    -6.693112739000000    -5.145085501000000
 H    -2.825675883000000    -6.810591310999999    -4.559637032000000
 H    1.475486211000000    -6.433819319000000    -5.620342880000000
 B    -4.506135179000000    0.459505235000000    3.464131946000000
 O    -5.167684283000000    1.668756232000000    3.611165257000000
 B    -6.253973180000000    1.785470416000000    4.444498328000000
 O    -4.965744409000000    -0.635277003000000    4.179507282000000
 B    -6.050993302000000    -0.530474020000000    5.015773349000000
 O    -6.692785376000000    0.682600747000000    5.145495131000000
 H    -6.809780492000000    2.825996718000000    4.560650015000000
 H    -6.432815578000000    -1.475088995000000    5.621597010000000
 B    0.459514259000000    4.506113161000000    -3.464159691000000
 O    1.668687781000000    5.167858120000000    -3.610948614000000
 B    1.785359815000000    6.254253410000000    -4.444148881000000
 O    -0.635389504000000    4.966029498000000    -4.179151563000000
 B    -0.530625861000000    6.051377681000000    -5.015293694000000
 O    0.682470815000000    6.693114332000000    -5.145084764000000
 H    2.825674802000000    6.810594984000000    -4.559633723000000
 H    -1.475485649000000    6.433818808000000    -5.620344746000000
 B    4.506134677000000    -0.459505170000000    3.464132609000000
 O    5.167682695000000    -1.668756555000000    3.611167614000000
 B    6.253971142000000    -1.785470682000000    4.444501279000000
 O    4.965744531000000    0.635277517000000    4.179506857000000
 B    6.050992983000000    0.530474593000000    5.015773505000000
 O    6.692783975000000    -0.682600565000000    5.145496977000000
 H    6.809777524000000    -2.825997320000000    4.560654419000000
 H    6.432815802000001    1.475089955000000    5.621596221000000




