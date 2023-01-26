%nproc=15
%mem=70GB
%chk=TetraPhenylSilane_Aldehyde_Benzobisoxazole.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

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
 H    0.189283951000000    -8.433662325000000    -7.121587622000000
 C    -0.230186288000000    -7.758403755000000    -6.377711087000000
 C    -1.570252202000000    -7.904779387999999    -5.995434591000000
 H    -2.174679832000000    -8.689852489000000    -6.445844673000000
 C    -2.149321184000000    -7.054813230000000    -5.040367496000000
 H    -3.185927613000000    -7.164214999000000    -4.742246440000000
 C    -1.349679045000000    -6.051362917000000    -4.478711936000000
 N    -1.632610953000000    -5.088764770000000    -3.544102419000000
 C    -0.030529743000000    -5.940042021000000    -4.884352688000000
 O    0.528978004000000    -4.898961864000000    -4.204608159000000
 C    0.581599402000000    -6.760624790000000    -5.823701945000000
 H    1.618350069000000    -6.631303390000000    -6.106732312000000
 C    -0.493564976000000    -4.428588211000000    -3.416109487000000
 H    -8.714749879999999    -0.300455822000000    6.770699628000000
 C    -7.975073720000000    0.144491996000000    6.107237244000000
 C    -8.052510549000001    1.511823690000000    5.811018782000000
 H    -8.849548886999999    2.111519013000000    6.246492298000000
 C    -7.119116019999999    2.123889140000000    4.960097487000000
 H    -7.175791839000000    3.181349051000000    4.727795554000000
 C    -6.105393276000000    1.328309782000000    4.411265335000000
 N    -5.074871231000000    1.638106026000000    3.561446996000000
 C    -6.064767633000000    -0.018799986000000    4.728655429000000
 O    -5.002505117000000    -0.569930003000000    4.075353440000000
 C    -6.966337924000000    -0.662960227000000    5.566905864000000
 H    -6.890497224000000    -1.720862885000000    5.783174933999999
 C    -4.451455822000000    0.484520732000000    3.387564231000000
 H    -0.189283587000000    8.433663235999999    -7.121587763000000
 C    0.230186435000000    7.758405015000000    -6.377710790000000
 C    1.570251747000000    7.904781946000000    -5.995432679000000
 H    2.174679133000000    8.689855692000000    -6.445841964000000
 C    2.149320434000000    7.054816274000000    -5.040364972000000
 H    3.185926398000000    7.164219048000000    -4.742242667000000
 C    1.349678623000000    6.051365112000000    -4.478710462000000
 N    1.632610373000000    5.088767159000000    -3.544100698000000
 C    0.030529917000000    5.940042941000000    -4.884352801000000
 O    -0.528977605000000    4.898962162000000    -4.204609039000000
 C    -0.581598923000000    6.760625189000000    -5.823702712000000
 H    -1.618349123000000    6.631302782000000    -6.106734330000000
 C    0.493564901000000    4.428589454000000    -3.416109189000000
 H    8.714748625999999    0.300456194000000    6.770701176000000
 C    7.975072369000000    -0.144491774000000    6.107239002000000
 C    8.052508166000001    -1.511823879000000    5.811022172000000
 H    8.849545812000002    -2.111519368000000    6.246496726000000
 C    7.119113477000000    -2.123889536000000    4.960101200000001
 H    7.175788500000000    -3.181349766999999    4.727800528000000
 C    6.105391638000000    -1.328309962000000    4.411267689000000
 N    5.074869690000000    -1.638106328000000    3.561449277000000
 C    6.064767002000000    0.018800212000000    4.728656189000000
 O    5.002505228000000    0.569930364000000    4.075353107000000
 C    6.966337484000000    0.662960670000000    5.566906251000000
 H    6.890497588000000    1.720863646000000    5.783174051000000
 C    4.451455331000000    -0.484520710000000    3.387564899000000




