%nproc=15
%mem=70GB
%chk=T-brick_Ketoenamine_Ketoenamine.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

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
 C    -7.272725721000000    0.234915213000000    0.002304620000000
 O    -7.804252559000000    1.332379557000000    0.158203109000000
 C    -8.095908644000000    -0.985704967000000    -0.233764105000000
 C    -9.436648047000000    -0.970417807000000    -0.230510552000000
 H    -7.605232147000000    -1.926902659000000    -0.428908047000000
 N    -10.275670796000000    0.092805089000000    -0.032484374000000
 H    -9.944586835000003    -1.911498231000000    -0.415174413000000
 C    -11.679358302000001    0.175729487000000    0.105053092000000
 H    -9.792203410999999    0.983873570000000    0.112822932000000
 C    -12.496713597999999    -0.938841356000000    0.335670806000000
 C    -12.310489980000000    1.408061322000000    -0.125348311000000
 C    -13.891986354000000    -0.828682220000000    0.338092966000000
 H    -12.069646071999999    -1.917562757000000    0.523935705000000
 C    -14.496257462000003    0.402769475000000    0.107403529000000
 H    -14.502466963000000    -1.708923312000000    0.521317813000000
 C    -13.704914767000000    1.523280095000000    -0.124734732000000
 H    -11.713757898000001    2.297953172000000    -0.311896338000000
 H    -15.579411818000001    0.488027065000000    0.108765352000000
 H    -14.168933904999999    2.488843675000000    -0.306727413000000
 C    7.272725721000000    -0.234915213000000    -0.002304620000000
 O    7.804252559000000    -1.332379557000000    -0.158203109000000
 C    8.095908644000000    0.985704967000000    0.233764105000000
 C    9.436648047000000    0.970417807000000    0.230510552000000
 H    7.605232147000000    1.926902659000000    0.428908047000000
 N    10.275670796000000    -0.092805089000000    0.032484374000000
 H    9.944586835000003    1.911498231000000    0.415174413000000
 C    11.679358302000001    -0.175729487000000    -0.105053092000000
 H    9.792203410999999    -0.983873570000000    -0.112822932000000
 C    12.496713597999999    0.938841356000000    -0.335670806000000
 C    12.310489980000000    -1.408061322000000    0.125348311000000
 C    13.891986354000000    0.828682220000000    -0.338092966000000
 H    12.069646071999999    1.917562757000000    -0.523935705000000
 C    14.496257462000003    -0.402769475000000    -0.107403529000000
 H    14.502466963000000    1.708923312000000    -0.521317813000000
 C    13.704914767000000    -1.523280095000000    0.124734732000000
 H    11.713757898000001    -2.297953172000000    0.311896338000000
 H    15.579411818000001    -0.488027065000000    -0.108765352000000
 H    14.168933904999999    -2.488843675000000    0.306727413000000
 C    -0.190598025000000    -8.827275494000000    -2.037264691000000
 O    -1.246465967000000    -9.427581740999999    -1.847453198000000
 C    0.985838022000000    -9.521536992000000    -2.634986525000000
 C    0.982365404000000    -10.827455927000001    -2.938943933000000
 H    1.881602124000000    -8.958884120000000    -2.848823137000000
 N    -0.030625361000000    -11.735010937000000    -2.784638775000000
 H    1.887918792000000    -11.238819227000000    -3.373096228000001
 C    -0.079703535000000    -13.135569738999999    -2.963965296000000
 H    -0.892629493000000    -11.336356405000000    -2.401387678000000
 C    1.063840677000000    -13.934594000000001    -3.095833563000000
 C    -1.327539689000000    -13.751074745000000    -3.148660377000000
 C    0.966957740000000    -15.297515884999999    -3.399123192000000
 H    2.056305202000000    -13.519130107000001    -2.961186457000000
 C    -0.280276197000000    -15.886782401000000    -3.578028044000000
 H    1.869958394000000    -15.894713661000001    -3.494133361000000
 C    -1.429707432000000    -15.112981838000000    -3.452744337000000
 H    -2.240473533000000    -13.166920826000000    -3.058589054000000
 H    -0.355310464000000    -16.944693824000002    -3.814009713000000
 H    -2.407898303000000    -15.565248132000001    -3.591834028000000




