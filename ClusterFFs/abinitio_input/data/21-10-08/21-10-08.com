%nproc=15
%mem=70GB
%chk=TriPhenanthrene_Cyanohydrine_Benzobisoxazole.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    -5.583474167000000    -2.420772783000000    0.000000000000000
 C    4.888188833000000    -3.625045783000000    0.000000000000000
 C    0.695285833000000    6.045818216999999    0.000000000000000
 C    -0.695287167000000    6.045818216999999    0.000000000000000
 C    5.583474833000000    -2.420772783000000    0.000000000000000
 C    -4.888187167000000    -3.625045783000000    0.000000000000000
 C    -4.886262167000000    -1.199823783000000    0.000000000000000
 C    3.482207833000000    -3.631715783000000    0.000000000000000
 C    1.404053833000000    4.831539217000000    0.000000000000000
 C    -1.404054167000000    4.831539217000000    0.000000000000000
 C    4.886261833000000    -1.199823783000000    0.000000000000000
 C    -3.482208167000000    -3.631715783000000    0.000000000000000
 C    -5.591177167000000    0.021127217000000    0.000000000000000
 C    2.777292833000000    -4.852666783000000    0.000000000000000
 C    2.813885833000000    4.831539217000000    0.000000000000000
 C    -2.813885167000000    4.831539217000000    0.000000000000000
 C    5.591178833000000    0.021127217000000    0.000000000000000
 C    -2.777292167000000    -4.852666783000000    0.000000000000000
 C    -4.882410167000000    1.248750217000000    0.000000000000000
 C    1.359754833000000    -4.852666783000000    0.000000000000000
 C    3.522654833000000    3.603916217000000    0.000000000000000
 C    -3.522655167000000    3.603916217000000    0.000000000000000
 C    4.882409833000000    1.248750217000000    0.000000000000000
 C    -1.359756167000000    -4.852666783000000    0.000000000000000
 C    -3.464874167000000    1.248750217000000    0.000000000000000
 C    0.650987833000000    -3.625045783000000    0.000000000000000
 C    2.813885833000000    2.376295217000000    0.000000000000000
 C    -2.813887167000000    2.376295217000000    0.000000000000000
 C    3.464873833000000    1.248750217000000    0.000000000000000
 C    -0.650987167000000    -3.625045783000000    0.000000000000000
 C    -2.758032167000000    0.024463217000000    0.000000000000000
 C    1.357830833000000    -2.400758783000000    0.000000000000000
 C    1.400200833000000    2.376295217000000    0.000000000000000
 C    -1.400202167000000    2.376295217000000    0.000000000000000
 C    2.758031833000000    0.024463217000000    0.000000000000000
 C    -1.357829167000000    -2.400758783000000    0.000000000000000
 C    -3.468725167000000    -1.199823783000000    0.000000000000000
 C    2.773439832999999    -2.404092783000000    0.000000000000000
 C    0.695285833000000    3.603916217000000    0.000000000000000
 C    -0.695286167000000    3.603916217000000    0.000000000000000
 C    3.468725833000000    -1.199823783000000    0.000000000000000
 C    -2.773440167000000    -2.404092783000000    0.000000000000000
 H    3.354423833000000    5.778069217000001    0.000000000000000
 H    4.612654833000000    3.603915217000000    0.000000000000000
 H    0.855198833000000    1.432329217000000    0.000000000000000
 H    -3.354422167000000    5.778070217000000    0.000000000000000
 H    -4.612655167000000    3.603915217000000    0.000000000000000
 H    -0.855201167000000    1.432328217000000    0.000000000000000
 H    -6.681165167000000    0.015983217000000    0.000000000000000
 H    -5.427411167000000    2.192718217000000    0.000000000000000
 H    -1.668032167000000    0.024463217000000    0.000000000000000
 H    -0.812828167000000    -1.456791783000000    0.000000000000000
 H    -0.814756167000000    -5.796634783000000    0.000000000000000
 H    -3.326741167000000    -5.794051783000000    0.000000000000000
 H    3.326742833000000    -5.794050783000000    0.000000000000000
 H    0.814752833000000    -5.796632783000000    0.000000000000000
 H    0.812831833000000    -1.456790783000000    0.000000000000000
 H    6.681166833000000    0.015983217000000    0.000000000000000
 H    5.427408833000001    2.192719217000000    0.000000000000000
 H    1.668031833000000    0.024463217000000    0.000000000000000
 N    6.942173225000000    -2.724937881000000    0.064989441000000
 O    5.748582580000000    -4.651731448000000    -0.022003196000000
 C    6.981328434000000    -4.046659170000000    0.030920069000000
 C    8.250415350000001    -6.207220332000000    -0.032420345000000
 C    9.461898885999998    -6.904949285000000    -0.038546109000000
 H    7.321328985000000    -6.771193312000000    -0.081511553000000
 C    10.668226104000000    -6.210332627000000    0.021256462000000
 H    9.461985946000000    -7.990782301000000    -0.091450314000000
 C    10.666027568000001    -4.819097386000000    0.088568927000000
 H    11.610040991000000    -6.753094740000000    0.014871726000000
 C    9.454889188999999    -4.122410421000000    0.094243010000000
 H    11.605080945999999    -4.273603682000000    0.135334292000000
 H    9.468317275000000    -3.034560310000000    0.145564429000000
 C    8.230305338000001    -4.805538801000000    0.032608119000000
 N    -5.830950233000000    -4.649630096000000    0.064989441000000
 O    -6.902807487000000    -2.652553708000000    -0.022003196000000
 C    -6.995172176000000    -4.022679092000000    0.030920069000000
 C    -9.500816481999999    -4.041460523000000    -0.032420345000000
 C    -10.710809108000001    -4.741771808000000    -0.038546109000000
 H    -9.524688447000001    -2.954861643000000    -0.081511553000000
 C    -10.712416765999999    -6.133790153000000    0.021256462000000
 H    -11.651211722999999    -4.198930885000000    -0.091450314000000
 C    -9.506472297000000    -6.827503543000000    0.088568927000000
 H    -11.653369877999998    -6.678044904000000    0.014871726000000
 C    -8.297554637999999    -6.126970179000000    0.094243010000000
 H    -9.503587362999999    -7.913494475000000    0.135334292000000
 H    -7.362162738000000    -6.682524110000000    0.145564429000000
 C    -8.276869526000000    -4.724885262000000    0.032608119000000
 N    -1.111222382000000    7.374567328000000    0.064989441000000
 O    1.154225128000000    7.304284659000000    -0.022003196000000
 C    0.013844113000000    8.069337561999999    0.030920069000000
 C    1.250401153000000    10.248679852000000    -0.032420345000000
 C    1.248910157000000    11.646719847000000    -0.038546109000000
 H    2.203359349000000    9.726054102999999    -0.081511553000000
 C    0.044190762000000    12.344121338000001    0.021256462000000
 H    2.189225514000000    12.189711903999999    -0.091450314000000
 C    -1.159554918000000    11.646599534000000    0.088568927000000
 H    0.043328919000000    13.431138013000000    0.014871726000000
 C    -1.157334111000000    10.249379447999999    0.094243010000000
 H    -2.101493100000000    12.187096609999999    0.135334292000000
 H    -2.106153899000000    9.717083302000001    0.145564429000000
 C    0.046564462000000    9.530423109999999    0.032608119000000




