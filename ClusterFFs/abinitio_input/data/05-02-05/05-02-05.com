%nproc=15
%mem=70GB
%chk=PMDA_Aldehyde_Hydrazone.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 C    2.594721975000000    -0.803042738000000    -0.463296744000000
 C    1.192502366000000    -0.497212196000000    -0.218045132000000
 C    0.065053697000000    -1.207654423000000    -0.591075722000000
 H    0.117972119000000    -2.139694743000000    -1.142514618000000
 C    -1.136014647000000    -0.628168196000000    -0.220766459000000
 C    -2.493796812000000    -1.086064864000000    -0.477627101000000
 C    -1.212416318000000    0.572343425000000    0.469936694000000
 C    -2.634581919000000    0.879079250000000    0.641083051000000
 C    -0.077288290000000    1.277983164000000    0.852371574000000
 H    -0.130718641000000    2.219027793000000    1.389272092000000
 C    1.131286774000000    0.698479109000000    0.482625637000000
 C    2.510597082000000    1.145897664000000    0.689299179000000
 N    -3.354717750000000    -0.136510250000000    0.035418250000000
 N    3.344032250000000    0.208749750000000    0.102548250000000
 O    -2.977386949000000    1.885929367000000    1.234317595000000
 O    -2.674622336000000    -2.137303305000000    -1.063707033000000
 O    2.734903026000000    2.186746473000000    1.279923781000000
 O    2.894996663000000    -1.814846508000000    -1.070204593000000
 C    -4.779157750000000    -0.244520250000000    -0.079881750000000
 C    4.776282250000000    0.250709750000000    0.090598250000000
 C    -5.598447750000000    0.885339750000000    -0.236421750000000
 C    -6.987337750000000    0.768939750000000    -0.396281750000000
 H    -5.191987750000000    1.892079750000000    -0.255001750000000
 C    -7.593107750000001    -0.481480250000000    -0.397281750000000
 H    -7.592377750000001    1.663179750000000    -0.526491750000000
 C    -6.810587750000000    -1.614680250000000    -0.229201750000000
 C    -5.424057750000000    -1.494860250000000    -0.070921750000000
 H    -7.269907750000000    -2.600490250000000    -0.224171750000000
 H    -4.878867750000000    -2.424760250000000    0.057408250000000
 C    5.483312250000000    1.458609750000000    0.205788250000000
 C    6.885602250000000    1.492349750000000    0.200838250000000
 H    4.976532250000000    2.412839750000000    0.314018250000000
 C    7.616882250000000    0.317529750000000    0.076798250000000
 H    7.402702250000000    2.444229750000000    0.296648250000000
 C    6.945222250000000    -0.890780250000000    -0.041811750000000
 C    5.544762250000000    -0.921110250000000    -0.033961750000000
 H    7.503602250000000    -1.818700250000000    -0.141941750000000
 H    5.089732250000000    -1.901230250000000    -0.138591750000000
 H    -15.706863694999999    -1.566379893000000    -1.882992967000000
 C    -15.339845386000000    -0.593117394000000    -1.565452561000000
 C    -16.245563462000000    0.416369514000000    -1.245383568000000
 H    -17.314768715000000    0.230286572000000    -1.310363717000000
 C    -15.780033258000000    1.668336886000000    -0.852607737000000
 H    -16.482452210000002    2.462674602000000    -0.612586486000000
 C    -14.406531497000000    1.907449714000000    -0.772572146000000
 H    -14.056746356000000    2.893470436000000    -0.469241759000000
 C    -13.482047886000000    0.897905596000000    -1.085116140000000
 C    -12.025484312000000    1.237160526000000    -0.995395851000000
 C    -13.963097592000000    -0.354981159000000    -1.492784550000000
 H    -13.300919102000000    -1.161790121000000    -1.784028536000000
 O    -11.672927312000001    2.399610392000000    -1.182171921000000
 N    -11.157894711000001    0.194079845000000    -0.698412012000000
 H    -11.489830849000001    -0.733182554000000    -0.478385815000000
 N    -9.815949766999999    0.431464179000000    -0.659951025000000
 C    -9.063177536000000    -0.599916173000000    -0.451420168000000
 H    -9.485953726000000    -1.611674711000000    -0.316932415000000
 H    15.706863694999999    1.566379893000000    1.882992967000000
 C    15.339845386000000    0.593117394000000    1.565452561000000
 C    16.245563462000000    -0.416369514000000    1.245383568000000
 H    17.314768715000000    -0.230286572000000    1.310363717000000
 C    15.780033258000000    -1.668336886000000    0.852607737000000
 H    16.482452210000002    -2.462674602000000    0.612586486000000
 C    14.406531497000000    -1.907449714000000    0.772572146000000
 H    14.056746356000000    -2.893470436000000    0.469241759000000
 C    13.482047886000000    -0.897905596000000    1.085116140000000
 C    12.025484312000000    -1.237160526000000    0.995395851000000
 C    13.963097592000000    0.354981159000000    1.492784550000000
 H    13.300919102000000    1.161790121000000    1.784028536000000
 O    11.672927312000001    -2.399610392000000    1.182171921000000
 N    11.157894711000001    -0.194079845000000    0.698412012000000
 H    11.489830849000001    0.733182554000000    0.478385815000000
 N    9.815949766999999    -0.431464179000000    0.659951025000000
 C    9.063177536000000    0.599916173000000    0.451420168000000
 H    9.485953726000000    1.611674711000000    0.316932415000000



