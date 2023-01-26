%nproc=15
%mem=70GB
%chk=TriPhenanthrene_long_CarboxylicAnhydride_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 H    -3.354391864000000    7.227269985000000    0.000000000000000
 C    -2.813854864000000    6.280738985000000    0.000000000000000
 C    -1.404023864000000    6.280738985000000    0.000000000000000
 H    -4.612624864000000    5.053114985000000    0.000000000000000
 C    -3.522624864000000    5.053115985000000    0.000000000000000
 C    -2.813856864000000    3.825494985000000    0.000000000000000
 H    -0.855170864000000    2.881527985000000    0.000000000000000
 C    -1.400171864000000    3.825494985000000    0.000000000000000
 C    -0.695255864000000    5.053115985000000    0.000000000000000
 C    0.695316136000000    7.495017985000000    0.000000000000000
 C    -0.695256864000000    7.495017985000000    0.000000000000000
 C    1.404084136000000    6.280738985000000    0.000000000000000
 C    2.813916136000000    6.280738985000000    0.000000000000000
 H    3.354454136000000    7.227268985000000    0.000000000000000
 H    4.612685136000000    5.053114985000000    0.000000000000000
 C    3.522685136000000    5.053115985000000    0.000000000000000
 C    2.813916136000000    3.825494985000000    0.000000000000000
 H    0.855229136000000    2.881528985000000    0.000000000000000
 C    1.400231136000000    3.825494985000000    0.000000000000000
 C    0.695316136000000    5.053115985000000    0.000000000000000
 C    4.719904136000000    0.524226985000000    0.000000000000000
 C    6.137440136000000    0.524226985000000    0.000000000000000
 H    6.682439136000000    1.468194985000000    0.000000000000000
 H    7.936197136000000    -0.708541015000000    0.000000000000000
 C    6.846209136000000    -0.703396015000000    0.000000000000000
 C    6.141292136000000    -1.924347015000000    0.000000000000000
 C    4.723756136000000    -1.924347015000000    0.000000000000000
 C    4.013062136000000    -0.700060015000000    0.000000000000000
 H    2.923062136000000    -0.700060015000000    0.000000000000000
 C    4.028470136000000    -3.128616015000000    0.000000000000000
 C    2.612861136000000    -3.125282015000000    0.000000000000000
 H    2.067862136000000    -2.181314015000000    0.000000000000000
 C    1.906018136000000    -4.349569015000000    0.000000000000000
 C    2.614785136000000    -5.577190015000000    0.000000000000000
 H    2.069783136000000    -6.521157015000000    0.000000000000000
 H    4.581773136000000    -6.518574015000000    0.000000000000000
 C    4.032323136000000    -5.577190015000000    0.000000000000000
 C    4.737238135999999    -4.356239015000000    0.000000000000000
 C    6.143219136000000    -4.349569015000000    0.000000000000000
 C    6.838505136000000    -3.145296015000000    0.000000000000000
 H    -6.682379864000000    1.468193985000000    0.000000000000000
 C    -6.137379864000000    0.524226985000000    0.000000000000000
 C    -4.719843864000000    0.524226985000000    0.000000000000000
 H    -2.923001864000000    -0.700060015000000    0.000000000000000
 C    -4.013001864000000    -0.700060015000000    0.000000000000000
 C    -4.723694864000000    -1.924347015000000    0.000000000000000
 H    -7.936134864000000    -0.708541015000000    0.000000000000000
 C    -6.846146864000000    -0.703396015000000    0.000000000000000
 C    -6.141231864000000    -1.924347015000000    0.000000000000000
 H    -2.067797864000000    -2.181315015000000    0.000000000000000
 C    -2.612798864000000    -3.125282015000000    0.000000000000000
 C    -4.028409864000000    -3.128616015000000    0.000000000000000
 C    -1.905956864000000    -4.349569015000000    0.000000000000000
 C    -2.614725864000000    -5.577190015000000    0.000000000000000
 H    -2.069725864000000    -6.521158015000000    0.000000000000000
 H    -4.581710864000000    -6.518576015000001    0.000000000000000
 C    -4.032261864000000    -5.577190015000000    0.000000000000000
 C    -4.737177864000000    -4.356239015000000    0.000000000000000
 C    -6.838443864000000    -3.145296015000000    0.000000000000000
 C    -6.143156864000000    -4.349569015000000    0.000000000000000
 C    -0.605969864000000    -4.349549015000000    0.000000000000000
 C    0.604030136000000    -4.349549015000000    0.000000000000000
 C    -4.069839864000000    1.650063985000000    0.000000000000000
 C    -3.464839864000000    2.697954985000000    0.000000000000000
 C    4.069900136000000    1.650063985000000    0.000000000000000
 C    3.464900136000000    2.697954985000000    0.000000000000000
 C    -1.132439091000000    8.884091550999999    0.002452522000000
 C    1.131794460000000    8.884666798000000    0.004694686000000
 N    -0.000517597000000    9.678283831000000    0.007821995000000
 O    2.320782116000000    9.150796321999998    0.016553678000000
 O    -2.321485862000000    9.149739746000000    -0.010873866000000
 C    -0.000885929000000    11.109927030000000    0.018109317000000
 C    1.154630050000000    11.849901312000000    -0.285500427000000
 H    2.091442808000000    11.369529109000000    -0.550788722000000
 C    1.156761946000000    13.251190120000002    -0.273387072000000
 H    2.069425663000000    13.790273370000000    -0.515876054000000
 C    0.000004067000000    13.949992772000000    0.045306300000000
 H    0.000536373000000    15.036421023000001    0.056403362000000
 C    -1.157416542000000    13.245971794000001    0.349730245000000
 H    -2.069622381000000    13.780871962000001    0.602945519000000
 C    -1.156259354000000    11.844708368999999    0.334668267000000
 H    -2.093369531000000    11.359972322000001    0.590794534000000
 C    8.260066097999999    -3.461201614999999    0.002452522000000
 C    7.128470776000000    -5.422386443000000    0.004694686000000
 N    8.381912387000000    -4.838569078000000    0.007821995000000
 O    6.764465675000000    -6.585149040000000    0.016553678000000
 O    9.084636925000000    -2.564271217000000    -0.010873866000000
 C    9.621944425000001    -5.554056977000000    0.018109317000000
 C    9.685039228000001    -6.924749561000000    -0.285500427000000
 H    8.800625095999999    -7.495877603000000    -0.550788722000000
 C    10.897533321999999    -7.627225852000000    -0.273387072000000
 H    10.908073831999999    -8.687157317000000    -0.515876054000000
 C    12.081085368000000    -6.974831424000000    0.045306300000000
 H    13.021700131999999    -7.518495376000000    0.056403362000000
 C    12.050079547999999    -5.620465652000000    0.349730245000000
 H    12.969413401000001    -5.097911396000000    0.602945519000000
 C    10.835962928000001    -4.920850502000000    0.334668267000000
 H    10.884711778000000    -3.866920680000000    0.590794534000000
 C    -7.127538725000000    -5.422716087000000    0.002452522000000
 C    -8.260153997000000    -3.462120119000000    0.004694686000000
 N    -8.381290257000000    -4.839539661000000    0.007821995000000
 O    -9.085122898000000    -2.565491499000000    0.016553678000000
 O    -6.763073238000000    -6.585284836000000    -0.010873866000000
 C    -9.620945355000000    -5.555680446000000    0.018109317000000
 C    -10.839539972000001    -4.924961591000000    -0.285500427000000
 H    -10.891931990000000    -3.873471851000000    -0.550788722000000
 C    -12.054157513000000    -5.623759915000000    -0.273387072000000
 H    -12.977349245999999    -5.102911725000000    -0.515876054000000
 C    -12.080959204999999    -6.974942954000000    0.045306300000000
 H    -13.022099733999999    -7.517696242000000    0.056403362000000
 C    -10.892548743000001    -7.625287924000000    0.349730245000000
 H    -10.899682787000000    -8.682731440000000    0.602945519000000
 C    -9.679597727000001    -6.923653861000000    0.334668267000000
 H    -8.791248816000000    -7.492846914000001    0.590794534000000




