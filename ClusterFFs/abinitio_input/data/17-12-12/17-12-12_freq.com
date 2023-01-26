%nproc=15
%mem=70GB
%chk=3_Phenyl_Double_CarboxylicAnhydride_Imide_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from 3_Phenyl_Double_CarboxylicAnhydride_Imide.log

0 1
 C   0.692958  -1.147662   0.021732
 C   1.394918   0.068517   0.023197
 C   0.701936   1.268441   0.021829
 C  -0.702242   1.268286   0.021764
 C  -1.394957   0.068208   0.020398
 C  -0.692726  -1.147816   0.021875
 C   2.866211  -0.233876   0.002823
 C   1.690861  -2.270346   0.043395
 N   2.956523  -1.643444   0.023752
 O   1.495303  -3.454401   0.076831
 O   3.793680   0.527697  -0.030517
 C   4.194614  -2.357864   0.024457
 C   4.363632  -3.447632  -0.830543
 H   3.556551  -3.755048  -1.481363
 C   5.569584  -4.142509  -0.821542
 H   5.699521  -4.992561  -1.481032
 C   6.603675  -3.747667   0.025801
 H   7.542477  -4.289281   0.026340
 C   6.427141  -2.654407   0.872453
 H   7.227521  -2.341053   1.532455
 C   5.221904  -1.958265   0.880116
 H   5.083498  -1.105348   1.530378
 C  -1.175934   2.693752   0.000012
 C   1.175312   2.694013   0.043583
 N  -0.000397   3.476696   0.021745
 O   2.298456   3.116624   0.078469
 O  -2.299169   3.116112  -0.034979
 C  -0.000552   4.906129   0.021671
 C   0.860107   5.596774  -0.832416
 H   1.530734   5.051088  -1.481988
 C   0.859219   6.988602  -0.824153
 H   1.531525   7.525634  -1.482943
 C  -0.000848   7.687402   0.021533
 H  -0.000963   8.771233   0.021479
 C  -0.860765   6.988502   0.867289
 H  -1.533184   7.525457   1.526026
 C  -0.861357   5.596675   0.875690
 H  -1.531867   5.050912   1.525317
 C  -1.690379  -2.270721   0.000171
 C  -2.866184  -0.234514   0.040783
 N  -2.956180  -1.644103   0.019925
 O  -3.793822   0.526850   0.074166
 O  -1.494557  -3.454737  -0.033095
 C  -4.194108  -2.358804   0.019367
 C  -5.221573  -1.959467  -0.836205
 H  -5.083426  -1.106537  -1.486505
 C  -6.426648  -2.655888  -0.828404
 H  -7.227164  -2.342738  -1.488338
 C  -6.602847  -3.749165   0.018297
 H  -7.541523  -4.290996   0.017862
 C  -5.568583  -4.143743   0.865550
 H  -5.698258  -4.993807   1.525076
 C  -4.362790  -3.448587   0.874414
 H  -3.555576  -3.755797   1.525165






