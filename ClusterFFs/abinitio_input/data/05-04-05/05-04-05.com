%nproc=15
%mem=70GB
%chk=PMDA_Hydrazide_Hydrazone.chk
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
 C    -9.077803676000000    -0.656973599000000    -0.495267331000000
 O    -9.526038949000000    -1.745406962000000    -0.848635954000000
 N    -9.860090933000000    0.443694644000000    -0.170483570000000
 H    -9.460505338000001    1.298756465000000    0.186740896000000
 N    -11.215105164000001    0.360809437000000    -0.295836185000000
 C    -11.875122404000001    1.445671390000000    -0.048150156000000
 C    -13.346120213000001    1.493549743000000    -0.157403626000000
 H    -11.367267687000000    2.382445939000000    0.243430078000000
 C    -14.003639846000000    2.709459525000000    0.050245886000000
 C    -15.394259138000001    2.769514060000000    -0.060088672000000
 H    -13.449499932000000    3.612082726000000    0.291252071000000
 C    -16.121715867999999    1.620060851000000    -0.373554947000000
 H    -15.913303783000000    3.712020976000000    0.095889669000000
 C    -15.463037011000001    0.407442698000000    -0.578584805000000
 H    -17.204560988000001    1.668270203000000    -0.460045634000000
 C    -14.073851534999999    0.342024897000000    -0.472102669000000
 H    -16.028739738999999    -0.488062820000000    -0.823499396000000
 H    -13.564015040999999    -0.605738143000000    -0.636084241000000
 C    9.077803676000000    0.656973599000000    0.495267331000000
 O    9.526038949000000    1.745406962000000    0.848635954000000
 N    9.860090933000000    -0.443694644000000    0.170483570000000
 H    9.460505338000001    -1.298756465000000    -0.186740896000000
 N    11.215105164000001    -0.360809437000000    0.295836185000000
 C    11.875122404000001    -1.445671390000000    0.048150156000000
 C    13.346120213000001    -1.493549743000000    0.157403626000000
 H    11.367267687000000    -2.382445939000000    -0.243430078000000
 C    14.003639846000000    -2.709459525000000    -0.050245886000000
 C    15.394259138000001    -2.769514060000000    0.060088672000000
 H    13.449499932000000    -3.612082726000000    -0.291252071000000
 C    16.121715867999999    -1.620060851000000    0.373554947000000
 H    15.913303783000000    -3.712020976000000    -0.095889669000000
 C    15.463037011000001    -0.407442698000000    0.578584805000000
 H    17.204560988000001    -1.668270203000000    0.460045634000000
 C    14.073851534999999    -0.342024897000000    0.472102669000000
 H    16.028739738999999    0.488062820000000    0.823499396000000
 H    13.564015040999999    0.605738143000000    0.636084241000000




