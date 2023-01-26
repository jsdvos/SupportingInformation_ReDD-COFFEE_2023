%nproc=15
%mem=70GB
%chk=F_BoronicAcid_BoronateEster_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC

Restart from F_BoronicAcid_BoronateEster.log

0 1
 F   1.316016  -2.372708   0.000001
 C   0.694311  -1.185467   0.000000
 C  -0.694311  -1.185467   0.000000
 F  -1.316018  -2.372707   0.000001
 C  -1.437209  -0.000000  -0.000000
 C  -0.694311   1.185467  -0.000001
 F  -1.316017   2.372707  -0.000001
 C   0.694311   1.185467  -0.000000
 F   1.316018   2.372707  -0.000001
 C   1.437209  -0.000000   0.000000
 B  -2.989756  -0.000000  -0.000000
 O  -3.772065   1.143725   0.000001
 O  -3.772065  -1.143724  -0.000001
 C  -5.076294   0.696073   0.000001
 C  -5.076294  -0.696073  -0.000001
 C  -6.246043   1.431403   0.000001
 C  -6.246043  -1.431402  -0.000001
 C  -7.439571   0.699538   0.000001
 C  -7.439571  -0.699537  -0.000000
 H  -6.231365   2.513552   0.000002
 H  -6.231365  -2.513551  -0.000002
 H  -8.383857   1.230683   0.000001
 H  -8.383857  -1.230682  -0.000001
 B   2.989756   0.000000   0.000000
 O   3.772065  -1.143725  -0.000001
 O   3.772064   1.143725   0.000001
 C   5.076294  -0.696073  -0.000000
 C   5.076294   0.696073   0.000000
 C   6.246043  -1.431403  -0.000001
 C   6.246043   1.431403   0.000001
 C   7.439571  -0.699538  -0.000000
 C   7.439571   0.699538   0.000000
 H   6.231365  -2.513552  -0.000002
 H   6.231365   2.513552   0.000001
 H   8.383857  -1.230683  -0.000001
 H   8.383857   1.230683   0.000001






