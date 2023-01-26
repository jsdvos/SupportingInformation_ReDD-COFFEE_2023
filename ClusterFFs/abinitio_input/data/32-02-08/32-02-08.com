%nproc=15
%mem=70GB
%chk=T-brick_Aldehyde_Benzobisoxazole.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 N    1.048041667000000    -2.362353333000000    -0.827760000000000
 C    0.687191667000000    -1.111843333000000    -0.380820000000000
 C    -0.707988333000000    -1.156633333000000    -0.350020000000000
 N    -1.140578333000000    -2.423203333000000    -0.704650000000000
 C    -1.452748333000000    -0.004173333000000    0.016440000000000
 C    -2.937038333000000    0.018916667000000    0.013290000000000
 C    -0.690808333000000    1.125556667000000    0.389200000000000
 H    -1.190148333000000    2.031256667000000    0.727650000000000
 C    0.707131667000000    1.132496667000000    0.363290000000000
 H    1.218751667000000    2.041706667000000    0.676780000000000
 C    1.457221667000000    0.014596667000000    -0.038090000000000
 C    2.930271667000000    0.035566667000000    -0.088570000000000
 C    -3.644438333000000    1.216186667000000    -0.227900000000000
 C    3.614231667000000    1.150886667000000    -0.607650000000000
 C    3.709801667000000    -1.028743333000000    0.397570000000000
 C    -3.714278333000000    -1.126953333000000    0.257540000000000
 C    -5.114658333000000    -1.083173333000000    0.260860000000000
 H    -3.237768333000000    -2.085023333000000    0.454110000000000
 C    5.107441667000000    -0.993503333000000    0.339110000000000
 H    3.237601667000000    -1.893393333000000    0.858000000000000
 C    -5.778958333000000    0.116916667000000    0.034260000000000
 H    -5.680408333000000    -1.992973333000000    0.442850000000000
 C    -5.043148333000000    1.269606667000000    -0.208850000000000
 H    -3.111798333000000    2.138136667000000    -0.449260000000000
 C    5.011001667000000    1.198966667000000    -0.646670000000000
 H    3.058831667000001    2.004406667000000    -0.992550000000000
 C    5.757241667000000    0.121996667000000    -0.180560000000000
 H    5.688571667000000    -1.831683333000000    0.715280000000000
 H    5.511931667000000    2.079726667000000    -1.040580000000000
 H    -5.551588333000000    2.212836667000000    -0.390060000000000
 C    -0.066158333000000    -3.157953333000000    -0.916690000000000
 H    1.993111667000000    -2.657013333000000    -1.011250000000000
 C    -0.040848333000000    -4.599613333000000    -1.196060000000000
 C    -1.251718333000000    -5.281883333000000    -1.390290000000000
 C    -1.273518333000000    -6.658563333000001    -1.631550000000000
 H    -2.193238333000000    -4.735723333000000    -1.348080000000000
 C    1.151281667000000    -5.339883333000000    -1.242140000000000
 C    -0.081488333000000    -7.375973333000000    -1.679350000000000
 H    -2.223038333000000    -7.167013333000000    -1.777580000000000
 C    1.130121667000000    -6.718103333000000    -1.483600000000000
 H    2.116521667000000    -4.871123333000000    -1.078620000000000
 H    2.060961667000000    -7.279213333000000    -1.512890000000000
 H    -12.623331915000000    -0.934913392000000    0.070175805000000
 C    -11.661472349000000    -0.427766708000000    0.016230154000000
 C    -11.626130096000001    0.956919513000000    -0.195271132000000
 H    -12.558544507000001    1.507834484000000    -0.303149939000000
 C    -10.407122724000002    1.648375146000000    -0.271603622000000
 H    -10.375295737000000    2.719558537000000    -0.435771176000000
 C    -9.222285877999997    0.914551411000000    -0.132099415000000
 N    -7.908619653000000    1.306208710000000    -0.164133994000000
 C    -9.300882573000001    -0.452224340000000    0.074874616000000
 O    -8.029034566000000    -0.933476501000000    0.173608805000000
 C    -10.485021653000000    -1.174166543000000    0.158834663000000
 H    -10.491906863000001    -2.244233139000000    0.321905916000000
 C    -7.238948513000000    0.180021007000000    0.017202268000000
 H    12.532023783000000    1.236874546000000    -1.081926044000000
 C    11.598957989000001    0.758448603000000    -0.789046276000000
 C    11.636500313999999    -0.495906920000000    -0.165732075000000
 H    12.595813367000000    -0.976160938000000    0.017615467000000
 C    10.456537051000000    -1.146635089000000    0.226895054000000
 H    10.481090135000001    -2.117466932000000    0.708849905000000
 C    9.235690753000000    -0.507384263000000    -0.023484157000000
 N    7.945436675000000    -0.890901050000000    0.237858328000000
 C    9.242235171000001    0.731214859000000    -0.642290286000000
 O    7.947769205000000    1.137642465000000    -0.776915321000000
 C    10.385844768000000    1.410047164000000    -1.044507875000000
 H    10.336433907000000    2.378480097000000    -1.525525939000000
 C    7.217877760000000    0.110694221000000    -0.228087973000000
 H    1.020087626000000    -13.997765089000000    -3.379788324000000
 C    0.502719697000000    -13.071746647000001    -3.134631258000000
 C    -0.898264071000000    -13.049704469000000    -3.124811633000000
 H    -1.451464077000000    -13.956458221000000    -3.362107151000000
 C    -1.603317207000000    -11.876436166000000    -2.814556777000000
 H    -2.687230541000000    -11.854894154000000    -2.806455745000000
 C    -0.866621439000000    -10.723285216999999    -2.515693460000000
 N    -1.269427287000000    -9.454903884000000    -2.185395336000000
 C    0.516327766000000    -10.787166665000001    -2.537194596000000
 O    0.996472656000000    -9.551110336000001    -2.220058863000000
 C    1.252388326000000    -11.926910560000000    -2.836583810000000
 H    2.334822218000000    -11.923984730000001    -2.839818904000000
 C    -0.134345845000000    -8.795398180999999    -2.023273879000000




