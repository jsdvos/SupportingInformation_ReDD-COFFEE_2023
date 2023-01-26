%nproc=15
%mem=70GB
%chk=TPG_BoronicAcid_BoronateEster.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    1.214670000000000    -0.706990000000000    -0.100210000000000
 C    -0.003630000000000    -1.377070000000000    -0.034780000000000
 O    -0.065580000000000    -2.740060000000000    -0.060670000000000
 C    -1.202870000000000    -0.677230000000000    0.059480000000000
 C    -1.171390000000000    0.714080000000000    0.087980000000000
 O    -2.318460000000000    1.447630000000000    0.179540000000000
 C    0.034640000000000    1.405600000000000    0.023800000000000
 C    1.221460000000000    0.684370000000000    -0.070160000000000
 O    2.430470000000000    1.313820000000000    -0.136030000000000
 H    -3.072450000000000    0.835100000000000    0.215630000000000
 H    2.278920000000000    2.273550000000000    -0.105640000000000
 H    0.839910000000000    -3.087250000000000    -0.127960000000000
 B    2.552879680000001    -1.464143700000000    -0.155066429000000
 O    2.677244097000000    -2.851413181000000    -0.169786435000000
 O    3.805235283000000    -0.855828570000000    -0.196781570000000
 C    4.031351708000000    -3.101338386000000    -0.221522754000000
 C    4.717170981000000    -1.888026764000000    -0.237935847000000
 C    4.689310814000000    -4.315586884000000    -0.254646019000000
 C    6.096325693000000    -1.826377885000000    -0.288318835000000
 C    6.088389595000000    -4.267673118000000    -0.305863732000000
 C    6.776316775000000    -3.050632309000000    -0.322327272000000
 H    4.145308019000000    -5.251282263000000    -0.241431641000000
 H    6.617442028000000    -0.877734489000000    -0.300595049000000
 H    6.647687100000000    -5.195243328000000    -0.333255864000000
 H    7.859070183000000    -3.052132709000000    -0.362246786000000
 B    -2.548508447000000    -1.422386817000000    0.092564559000000
 O    -3.796118066000000    -0.803037813000000    0.101331098000000
 O    -2.685271332000000    -2.808330300000000    0.117834722000000
 C    -4.717703866000000    -1.826976264000000    0.132577500000000
 C    -4.042312641000000    -3.046192636000000    0.142611733000000
 C    -6.097033617000000    -1.753084824000000    0.152483390000000
 C    -4.711412738000000    -4.254407906000000    0.173069449000000
 C    -6.788267175000000    -2.971106994000000    0.183444330000000
 C    -6.110800093000000    -4.194070702000000    0.193509404000000
 H    -6.609985717000000    -0.799956058000000    0.144408575000000
 H    -4.175448025000000    -5.194792229000000    0.180578445000000
 H    -7.871601574000000    -2.962990450000000    0.199913211000000
 H    -6.678636131000000    -5.116529838000000    0.217636969000000
 B    0.076858326000000    2.943502581000000    0.036814004000000
 O    1.244383077000000    3.703156324000000    0.040270475000000
 O    -1.047222864000000    3.766013882000000    0.046746067000000
 C    0.830974019000000    5.017572026000000    0.052558253000000
 C    -0.562319312000000    5.055791770000000    0.056495425000000
 C    1.595693191000000    6.168028434000000    0.060393530000000
 C    -1.262763149000000    6.246439395000000    0.068470965000000
 C    0.898053457000000    7.382726349000000    0.072567031000000
 C    -0.499522246000000    7.421063564000000    0.076516305000000
 H    2.677263782000000    6.125197149000000    0.057224905000000
 H    -2.345054842000000    6.262969506000000    0.071417022000000
 H    1.455678442000000    8.311684305000000    0.079048921000000
 H    -1.005339014000000    8.379192030000000    0.086003263000000



