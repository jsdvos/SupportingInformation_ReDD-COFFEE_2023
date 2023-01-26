%nproc=15
%mem=70GB
%chk=Pyrene_H_Catechol_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from Pyrene_H_Catechol_BoronateEster.log

0 1
 C   0.000002   3.548247  -0.000001
 C  -1.210778   2.861265  -0.000001
 C  -1.225105   1.461005  -0.000002
 C   0.000000   0.718378  -0.000002
 C   1.225106   1.461004  -0.000002
 C   1.210781   2.861263  -0.000001
 H  -2.152577   3.395941  -0.000001
 H   2.152580   3.395939  -0.000001
 C  -0.000000  -0.718377  -0.000002
 C  -1.225106  -1.461003  -0.000002
 C  -1.210781  -2.861263  -0.000001
 C  -0.000002  -3.548246  -0.000001
 C   1.210778  -2.861264  -0.000001
 C   1.225105  -1.461005  -0.000002
 H  -2.152580  -3.395938  -0.000001
 H   2.152577  -3.395941  -0.000001
 C   2.410758   0.679647  -0.000002
 C   2.410757  -0.679649  -0.000002
 C  -2.410757   0.679650  -0.000002
 C  -2.410758  -0.679647  -0.000002
 H   0.000002   4.632262  -0.000000
 H  -0.000002  -4.632262  -0.000000
 B   4.507178  -0.000002  -0.000007
 C   6.043435  -0.000002  -0.000001
 C   6.761467   1.207727   0.000001
 C   6.761470  -1.207729   0.000001
 C   8.153377   1.208987   0.000004
 C   8.153380  -1.208985   0.000004
 C   8.850371   0.000002   0.000006
 O   3.707828   1.143744  -0.000001
 O   3.707827  -1.143747  -0.000000
 H   8.695960   2.147769   0.000006
 H   8.695965  -2.147766   0.000006
 H   6.220232  -2.147642  -0.000001
 H   6.220227   2.147639  -0.000001
 H   9.934897   0.000003   0.000009
 B  -4.507178   0.000003  -0.000007
 C  -6.043435   0.000002  -0.000001
 C  -6.761467  -1.207727   0.000001
 C  -6.761471   1.207728   0.000001
 C  -8.153376  -1.208987   0.000004
 C  -8.153380   1.208984   0.000004
 C  -8.850371  -0.000003   0.000006
 O  -3.707828  -1.143743  -0.000001
 O  -3.707827   1.143747  -0.000000
 H  -8.695959  -2.147770   0.000006
 H  -8.695966   2.147765   0.000006
 H  -6.220233   2.147642  -0.000001
 H  -6.220226  -2.147639  -0.000001
 H  -9.934897  -0.000004   0.000009





