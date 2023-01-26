%nproc=15
%mem=70GB
%chk=TriPhenanthrene_CarboxylicAnhydride_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

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
 C    7.005038075000000    -2.736692476000000    0.002452522000000
 C    5.873427131000001    -4.697868290000000    0.004694686000000
 N    7.126873392000000    -4.114060910000000    0.007821995000000
 O    5.509412768000000    -5.860627988000000    0.016553678000000
 O    7.829616047000000    -1.839768646000000    -0.010873866000000
 C    8.366899730000000    -4.829558686000000    0.018109317000000
 C    8.429983614999999    -6.200251773000000    -0.285500427000000
 H    7.545564934000000    -6.771372770000000    -0.550788722000000
 C    9.642472114000000    -6.902737722000000    -0.273387072000000
 H    9.653004180000000    -7.962669271000000    -0.515876054000000
 C    10.826029356999999    -6.250352721000000    0.045306300000000
 H    11.766639789999999    -6.794024167000000    0.056403362000000
 C    10.795034325000000    -4.895986703000000    0.349730245000000
 H    11.714372340000001    -4.373439770000000    0.602945519000000
 C    9.580923277000000    -4.196361882000000    0.334668267000000
 H    9.629680522999999    -3.142432448000000    0.590794534000000
 C    -5.872562426000000    -4.698195365000000    0.002452522000000
 C    -7.005185424000000    -2.737603860000000    0.004694686000000
 N    -7.126316256000000    -4.115023880000000    0.007821995000000
 O    -7.830157859000000    -1.840978492000000    0.016553678000000
 O    -5.508092356000000    -5.860762677000000    -0.010873866000000
 C    -8.365968531000000    -4.831169551000000    0.018109317000000
 C    -9.584565634000001    -4.200455499000000    -0.285500427000000
 H    -9.636961797000000    -3.148965965000000    -0.550788722000000
 C    -10.799180421000001    -4.899258610000000    -0.273387072000000
 H    -11.722374206000000    -4.378414058000000    -0.515876054000000
 C    -10.825976787000000    -6.250441755000000    0.045306300000000
 H    -11.767115177999999    -6.793198752000000    0.056403362000000
 C    -9.637563761999999    -6.900782040000000    0.349730245000000
 H    -9.644693639000000    -7.958225584000000    0.602945519000000
 C    -8.424615512000001    -6.199143197000000    0.334668267000000
 H    -7.536264357000000    -6.768332749000000    0.590794534000000
 C    -1.132475040000000    7.434887180000000    0.002452522000000
 C    1.131758509000000    7.435471630000000    0.004694686000000
 N    -0.000556773000000    8.229084060000000    0.007821995000000
 O    2.320745083000000    7.701605986000000    0.016553678000000
 O    -2.321522890000000    7.700530542000000    -0.010873866000000
 C    -0.000930925000000    9.660727259000000    0.018109317000000
 C    1.154582047000000    10.400706237000000    -0.285500427000000
 H    2.091396758000000    9.920337841000000    -0.550788722000000
 C    1.156708247000000    11.801995053000001    -0.273387072000000
 H    2.069369774000000    12.341082012999999    -0.515876054000000
 C    -0.000052471000000    12.500793004000000    0.045306300000000
 H    0.000475418000000    13.587221256999999    0.056403362000000
 C    -1.157470219000000    11.796767321000001    0.349730245000000
 H    -2.069678232000000    12.331663782000000    0.602945519000000
 C    -1.156307335000000    10.395503901000000    0.334668267000000
 H    -2.093415543000000    9.910764045000001    0.590794534000000



