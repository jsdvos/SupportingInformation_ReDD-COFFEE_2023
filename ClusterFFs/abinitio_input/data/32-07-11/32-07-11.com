%nproc=15
%mem=70GB
%chk=T-brick_AmineBorane_Borazine.chk
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
 N    -7.236510675000000    0.143962933000000    0.060423832000000
 B    -8.067414042999999    -1.080377185000000    0.477270819000000
 B    -8.035882307000000    1.398388581000000    -0.327646562000000
 H    -7.572991096000000    -2.092308174000000    0.808425939000000
 N    -9.592966988000001    -1.068799845000000    0.507425453000000
 H    -7.516446420000000    2.391841533000000    -0.676007357000000
 N    -9.560985860000001    1.442557912000000    -0.305621891000000
 B    -10.325335743000000    0.200432317000000    0.113340337000000
 H    -10.125392763000001    -1.919082262000000    0.798350076000000
 H    -10.071637550000000    2.311802448000000    -0.578997040000000
 H    -11.506297369000000    0.221869797000000    0.133084259000000
 N    7.214278621999999    0.164946515000000    -0.213324296000000
 B    7.983806116000000    1.433740543000000    -0.614771299000000
 B    8.074431005999999    -1.055953235000000    0.150981918000000
 H    7.440442265000000    2.434203720000000    -0.901042580000000
 N    9.507943886000000    1.493959045000000    -0.656399586000000
 H    7.604332391000000    -2.085993885000000    0.461048366000000
 N    9.599773163000000    -1.027059700000000    0.121429723000000
 B    10.301915434000000    0.255201412000000    -0.285311914000000
 H    9.997850252999998    2.374407384000000    -0.931893698000000
 H    10.152530834000000    -1.877126088000000    0.372290795000000
 H    11.482404816000001    0.289552897000000    -0.313321651000000
 N    -0.092078044000000    -8.801255848000000    -1.986485945000000
 B    1.188737519000000    -9.650906836000001    -1.958642919000000
 B    -1.384649611000000    -9.545668198000000    -2.358354659000000
 H    2.235479477000000    -9.200859498000000    -1.675958013000000
 N    1.194610957000000    -11.142583943000000    -2.279921670000000
 H    -2.424203374000000    -9.004543154000000    -2.426393700000000
 N    -1.412234403000000    -11.036484011000001    -2.682521519999999
 B    -0.114072971000000    -11.821084727000001    -2.639922250000000
 H    2.083964846000000    -11.689559836000001    -2.248912597000000
 H    -2.308959223000000    -11.509208739000000    -2.934515788000000
 H    -0.122406732000000    -12.975549534000002    -2.890244660000000




