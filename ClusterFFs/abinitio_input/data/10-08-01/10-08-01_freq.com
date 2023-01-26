%nproc=15
%mem=70GB
%chk=An_Catechol_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from An_Catechol_BoronateEster.log

0 1
 C   3.611535   0.711332   0.000002
 C   3.611534  -0.711342   0.000002
 C   2.467917  -1.441066   0.000002
 C   1.225458  -0.724493   0.000002
 C   2.467922   1.441062   0.000002
 C   1.225460   0.724492   0.000002
 C   0.000002  -1.397945   0.000002
 C  -1.225464  -0.724486   0.000002
 C   0.000006   1.397946   0.000002
 C  -1.225462   0.724490   0.000002
 C  -2.467915  -1.441057   0.000002
 C  -3.611536  -0.711329   0.000002
 C  -3.611534   0.711336   0.000002
 C  -2.467912   1.441063   0.000002
 H   2.483807  -2.523685   0.000002
 H   2.483818   2.523680   0.000002
 H  -0.000007  -2.483451   0.000002
 H  -0.000001   2.483452   0.000002
 H  -2.483815  -2.523676   0.000002
 H  -2.483813   2.523682   0.000002
 B   5.708124  -0.000005   0.000006
 C   7.243839  -0.000004   0.000000
 C   7.961048   1.208275  -0.000002
 C   7.961056  -1.208276  -0.000002
 C   9.352885   1.209402  -0.000005
 C   9.352894  -1.209394  -0.000005
 C   10.049291   0.000006  -0.000007
 O   4.917168   1.147853   0.000002
 O   4.917166  -1.147863   0.000002
 H   9.895848   2.147919  -0.000007
 H   9.895863  -2.147907  -0.000007
 H   7.419019  -2.147665  -0.000000
 H   7.419004   2.147660  -0.000000
 H   11.133848   0.000010  -0.000010
 B  -5.708124   0.000007   0.000006
 C  -7.243840   0.000004   0.000000
 C  -7.961047  -1.208275  -0.000002
 C  -7.961058   1.208276  -0.000002
 C  -9.352885  -1.209403  -0.000005
 C  -9.352896   1.209392  -0.000005
 C  -10.049292  -0.000008  -0.000007
 O  -4.917169  -1.147854   0.000002
 O  -4.917167   1.147865   0.000002
 H  -9.895847  -2.147921  -0.000007
 H  -9.895866   2.147906  -0.000007
 H  -7.419022   2.147665  -0.000000
 H  -7.419003  -2.147659  -0.000000
 H  -11.133849  -0.000013  -0.000010





