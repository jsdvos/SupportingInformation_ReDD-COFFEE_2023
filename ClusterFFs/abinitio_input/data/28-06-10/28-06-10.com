%nproc=15
%mem=70GB
%chk=TetraPhenylMethane_Nitrile_Triazine.chk
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
 C    -0.453960354000000    -4.451835705000000    -3.422415232000000
 N    0.640420532000000    -4.854590133000000    -4.091782273000000
 C    0.469438657000000    -5.909377011000000    -4.882570951000000
 N    -1.651034760000000    -5.055353904000000    -3.526609170000000
 C    -1.694095482000000    -6.098893557000000    -4.348917133000000
 N    -0.671198735000000    -6.581424625000000    -5.059542795000000
 H    -2.649619389000000    -6.605661552000000    -4.452137346000000
 H    1.337908428000000    -6.256310804000000    -5.435646433000000
 C    -4.451835767000000    0.453960396000000    3.422416145000000
 N    -4.854589117000000    -0.640420015000000    4.091784613000000
 C    -5.909376112000000    -0.469438128000000    4.882573132000000
 N    -5.055355099000000    1.651034361000000    3.526608581000000
 C    -6.098894746000000    1.694095147000000    4.348916548000000
 N    -6.581424801000000    0.671198852000000    5.059543550000000
 H    -6.605663644000000    2.649618704000000    4.452135564000000
 H    -6.256309048000000    -1.337907518000000    5.435649748000000
 C    0.453960263000000    4.451836908000000    -3.422414979000000
 N    -0.640420224000000    4.854590310000000    -4.091783290000000
 C    -0.469438455000000    5.909377432000000    -4.882571664000000
 N    1.651034192000000    5.055356309000000    -3.526607430000000
 C    1.694094858000000    6.098896083000000    -4.348915243000000
 N    0.671198479000000    6.581426199000001    -5.059542082000000
 H    2.649618384000000    6.605665039000000    -4.452134267000000
 H    -1.337907910000000    6.256310412999999    -5.435648151000000
 C    4.451835287000001    -0.453960333000000    3.422416778000000
 N    4.854589280000000    0.640420519000000    4.091784135000000
 C    5.909375798000000    0.469438665000000    4.882573299000000
 N    5.055353561000000    -1.651034688000000    3.526610869000000
 C    6.098892824999999    -1.694095394000000    4.348919328000000
 N    6.581423446000000    -0.671198677000000    5.059545336000000
 H    6.605660871000000    -2.649619259000000    4.452139675000000
 H    6.256309235000000    1.337908410000000    5.435649046000000



