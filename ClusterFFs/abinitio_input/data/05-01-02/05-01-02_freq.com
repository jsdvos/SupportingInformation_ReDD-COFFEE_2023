%nproc=15
%mem=70GB
%chk=PMDA_BoronicAcid_Boroxine_freq.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 freq(noraman) SCF=YQC NoSymm

Restart from PMDA_BoronicAcid_Boroxine.log

0 1
 C   2.664908  -0.781637  -0.584221
 C   1.218537  -0.489753  -0.360659
 C   0.106105  -1.154605  -0.862415
 H   0.185573  -2.016437  -1.512877
 C  -1.116319  -0.628490  -0.462157
 C  -2.493214  -1.088283  -0.808360
 C  -1.218757   0.482315   0.375741
 C  -2.665094   0.774219   0.599326
 C  -0.106317   1.147184   0.877494
 H  -0.185808   2.009000   1.527970
 C   1.116097   0.621069   0.477231
 C   2.493003   1.080855   0.823430
 N  -3.374569  -0.204272  -0.139160
 N   3.374334   0.196912   0.154176
 O  -3.149107   1.652217   1.266670
 O  -2.810502  -2.013985  -1.510431
 O   2.810266   2.006630   1.525430
 O   3.148973  -1.659544  -1.251650
 C  -4.797222  -0.288833  -0.200857
 C   4.797009   0.281721   0.215639
 C  -5.552985   0.874946  -0.363404
 C  -6.937983   0.784363  -0.417653
 H  -5.060060   1.834657  -0.433220
 C  -7.591078  -0.454737  -0.321489
 H  -7.526481   1.686295  -0.540532
 C  -6.806881  -1.608505  -0.163416
 C  -5.421360  -1.534576  -0.098061
 H  -7.293082  -2.574248  -0.086905
 H  -4.826345  -2.429690   0.018662
 C   5.420911   1.527554   0.112452
 C   6.806409   1.601759   0.177724
 H   4.825745   2.422528  -0.004559
 C   7.590817   0.448168   0.336087
 H   7.292450   2.567561   0.100912
 C   6.937957  -0.791032   0.432521
 C   5.552958  -0.881903   0.378348
 H   7.526640  -1.692831   0.555518
 H   5.060196  -1.841681   0.448304
 B  -9.132839  -0.546220  -0.387631
 O  -9.904501   0.594063  -0.549298
 B  -11.275312   0.518653  -0.608438
 O  -9.775519  -1.770376  -0.286407
 B  -11.145720  -1.857416  -0.343657
 O  -11.892029  -0.709764  -0.504875
 H  -11.922951   1.501604  -0.747521
 H  -11.682346  -2.910581  -0.255055
 B   9.132564   0.539868   0.402283
 O   9.904418  -0.600231   0.564246
 B   11.275227  -0.524533   0.623366
 O   9.775029   1.764151   0.301000
 B   11.145201   1.851476   0.358255
 O   11.891723   0.703945   0.519402
 H   11.923006  -1.507346   0.762777
 H   11.681644   2.904731   0.269638





