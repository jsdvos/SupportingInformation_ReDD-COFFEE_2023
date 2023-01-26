%nproc=15
%mem=70GB
%chk=T-brick_BoronicAcid_Borosilicate.chk
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
 B    -7.354386730000000    0.144031424000000    0.080148531000000
 O    -8.052090553999999    -0.982344060000000    0.285076258000000
 O    -8.024238801999999    1.289055420000000    -0.117780421000000
Si    -9.537342432999999    -1.571613699000000    0.240459031000000
Si    -9.497674250999999    1.865681210000000    -0.343411442000000
 O    -10.394385875999999    -0.780003910000000    1.332883396000000
 O    -10.153667177000001    -1.209039298000000    -1.189703169000000
 H    -9.549761518000000    -3.021430237000000    0.481168051000000
 O    -10.138729891000001    1.067877667000000    -1.571348387000000
 O    -10.357596862999998    1.497820408000000    0.953170277000000
 H    -9.477230584000001    3.314319218000000    -0.590871743000000
 B    -10.763655381000000    0.456802718000000    1.693400266000000
 H    -11.194861911000000    0.643978724000000    2.776170734000000
 B    -10.419660642000000    -0.171829621000000    -1.996002799000000
 H    -10.597297139000000    -0.363245391000000    -3.147133611000000
 B    7.332336524000000    0.180284605000000    -0.202793037000000
 O    7.970230645000000    1.242509393000000    -0.714923216000000
 O    8.060800907999999    -0.846244258000000    0.260387009000000
Si    9.419310002999998    1.749270020000000    -1.160529802000000
Si    9.559344221000000    -1.396713086000000    0.336352056000000
 O    10.340013837000001    1.760498648000000    0.145892367000000
 O    10.026994334999998    0.651131664000000    -2.151232688000000
 H    9.355832111000000    3.072865648000000    -1.796240610000000
 O    10.131174245000000    -1.429480794000000    -1.156031497000000
 O    10.422467721000000    -0.320492839000000    1.144238450000000
 H    9.614615132999999    -2.723506141000000    0.966250414000000
 B    10.784336913000001    0.970033822000000    1.132687878000000
 H    11.246176976999998    1.454985311000000    2.104743237000000
 B    10.334521074000001    -0.651253802000000    -2.228011176000000
 H    10.478909104000000    -1.151197563000000    -3.287494634999999
 B    -0.088126841000000    -8.920422431000000    -1.994704586000000
 O    1.062127426000000    -9.596689028000000    -2.125419314000000
 O    -1.244374895000000    -9.577162295999999    -2.170035159000000
Si    1.649786385000000    -11.006034793000000    -2.598287170000000
Si    -1.836636668000000    -10.984740824999999    -2.641593593000000
 O    1.043164883000000    -12.122087882000001    -1.628237496000000
 O    1.076229555000000    -11.297227583000000    -4.062158622000000
 H    3.119431606000000    -11.009559298999999    -2.584374493000000
 O    -1.232347390000000    -11.295316077000001    -4.088538461000000
 O    -1.266192125000000    -12.099392626000000    -1.647583026000000
 H    -3.306224737000000    -10.971826462999999    -2.660321866000000
 B    -0.120182390000000    -12.616983102000001    -1.183574859000000
 H    -0.134637741000000    -13.290193751000000    -0.214067624000000
 B    -0.070622875000000    -11.418385483000000    -4.745331950999999
 H    -0.057461447000000    -11.321883835000000    -5.921686947000000



