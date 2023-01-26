%nproc=15
%mem=70GB
%chk=TetraPhenylSilane_PrimaryAmine_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
Si    -0.000000000000000    0.000000000000000    -0.000000000000000
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
 C    -1.310539960000000    -6.080358618000000    -4.705151020000000
 C    -2.133517319000000    -6.946775353000000    -5.401181093000000
 H    -3.213963260000000    -6.851772711000000    -5.355991010000000
 C    -1.518200752000000    -7.944622486000000    -6.163244874000000
 H    -2.129224558000000    -8.647373677999999    -6.727210861000000
 C    -0.113911637000000    -8.051328700999999    -6.210421825000000
 H    0.345026335000000    -8.835865424000000    -6.809701675000000
 C    0.694425853000000    -7.160893947000000    -5.497150051000000
 H    1.777407303000000    -7.230691172000000    -5.524093945000000
 C    0.059961317000000    -6.183007277000000    -4.753095596000000
 C    0.614583096000000    -5.123699936999999    -3.917517836000000
 C    -1.641922142000000    -4.953648651000000    -3.839893087000000
 N    -0.449853679000000    -4.412753503000000    -3.394501588000000
 O    -2.805380192000000    -4.649073516000000    -3.644300250000000
 O    1.820636607000000    -5.011663832000000    -3.785375414000000
 C    -6.309870950000000    1.219766294000000    4.418645630000000
 C    -7.417893878000000    1.947186666000000    4.813072239000000
 H    -7.666138889000001    2.891875614000000    4.339396947000000
 C    -8.202162997000000    1.416341842000000    5.841751152000000
 H    -9.083882650000001    1.956582338000000    6.182306028000000
 C    -7.864466087000000    0.187817458000000    6.443688387000000
 H    -8.488432937000001    -0.207613896000000    7.243411893000000
 C    -6.738528788000000    -0.527377085000000    6.024400732000001
 H    -6.466128479000000    -1.475016873000000    6.478518943000000
 C    -5.981656718000000    0.019674485000000    5.004447856000000
 C    -4.772704768000000    -0.475761804000000    4.355675464000000
 C    -5.314038573000000    1.499385355000000    3.390009714000000
 N    -4.414678343000000    0.449092455000000    3.392099750000000
 O    -5.392974649000000    2.511161867000000    2.715669598000000
 O    -4.270264589000000    -1.527407622000000    4.710884867000000
 C    1.310539780000000    6.080360796000000    -4.705149590000000
 C    2.133517108000000    6.946778416000000    -5.401178598000000
 H    3.213963090000000    6.851776846000000    -5.355987233000000
 C    1.518200458000000    7.944625008000000    -6.163243021000000
 H    2.129224238000000    8.647376862000000    -6.727208210000000
 C    0.113911293000000    8.051329828000000    -6.210421639000000
 H    -0.345026744000000    8.835866151999999    -6.809701964000000
 C    -0.694426163000000    7.160894202000000    -5.497150915000000
 H    -1.777407649000000    7.230690351000000    -5.524096096000000
 C    -0.059961542000000    6.183008094000000    -4.753095794000000
 C    -0.614583264000000    5.123700123000000    -3.917518797000000
 C    1.641922050000000    4.953651077000000    -3.839891368000000
 N    0.449853595000000    4.412754700000000    -3.394501343000000
 O    2.805380169999999    4.649077082000000    -3.644297169000000
 O    -1.820636822000000    5.011662804000000    -3.785377826000000
 C    6.309869402000000    -1.219766638000000    4.418647943000000
 C    7.417891547000000    -1.947187487000000    4.813075871000000
 H    7.666135959000000    -2.891877199000000    4.339401789000000
 C    8.202160681000001    -1.416342122000000    5.841754494000000
 H    9.083879733000000    -1.956582966000000    6.182310374000000
 C    7.864464557000000    -0.187816748000000    6.443690148000000
 H    8.488431404000000    0.207615014000000    7.243413455000000
 C    6.738528040000000    0.527378257000000    6.024401181000000
 H    6.466128342000000    1.475018807000000    6.478518168000000
 C    5.981655938000000    -0.019673866000000    5.004448626000000
 C    4.772704681000000    0.475762687000000    4.355675144000000
 C    5.314037222000000    -1.499386061000000    3.390011934000000
 N    4.414677880000000    -0.449092397000000    3.392100361000000
 O    5.392972726000000    -2.511163428000000    2.715673035000000
 O    4.270265242000000    1.527409346000000    4.710883104000000



