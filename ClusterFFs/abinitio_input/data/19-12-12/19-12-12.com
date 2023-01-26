%nproc=15
%mem=70GB
%chk=DBA12_CarboxylicAnhydride_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 C    2.750545300000000    0.714052633000000    0.000000000000000
 C    3.967385300000000    1.427232633000000    0.000000000000000
 C    5.134923300000000    0.697167633000000    0.000000000000000
 C    5.134905300000000    -0.697294367000000    0.000000000000000
 C    3.967338300000000    -1.427295367000000    0.000000000000000
 C    2.750523300000000    -0.714054367000000    0.000000000000000
 H    3.973029300000000    2.508793633000000    0.000000000000000
 H    3.972857300000000    -2.508858367000000    0.000000000000000
 C    1.523744300000000    -1.428965367000000    0.000000000000000
 C    0.475695300000000    -2.034128367000000    0.000000000000000
 C    -0.756889700000000    -2.739018366999999    0.000000000000000
 C    -0.747637700000000    -4.149421367000000    0.000000000000000
 C    -1.963640700000000    -4.795554367000000    0.000000000000000
 C    -3.171301700000000    -4.098371367000000    0.000000000000000
 C    -3.219770700000000    -2.722223367000000    0.000000000000000
 C    -1.993726700000000    -2.024989367000000    0.000000000000000
 H    0.186211300000000    -4.695066367000000    0.000000000000000
 H    -4.159213700000000    -2.186260367000000    0.000000000000000
 C    -1.999634700000000    -0.605105367000000    0.000000000000000
 C    1.523820300000000    1.429081633000000    0.000000000000000
 C    0.475673300000000    2.034067633000000    0.000000000000000
 C    -0.756847700000000    2.739067633000000    0.000000000000000
 C    -0.747585700000000    4.149487633000000    0.000000000000000
 C    -1.963575700000000    4.795616633000000    0.000000000000000
 C    -3.171234700000000    4.098376633000000    0.000000000000000
 C    -3.219715700000000    2.722243633000000    0.000000000000000
 C    -1.993596700000000    2.025034633000000    0.000000000000000
 H    0.186303300000000    4.695069633000000    0.000000000000000
 H    -4.159103700000000    2.186199633000000    0.000000000000000
 C    -1.999479700000000    0.605114633000000    0.000000000000000
 C    6.523997363000000    1.132393711000000    0.002452522000000
 C    6.524553622000000    -1.131839845000000    0.004694686000000
 N    7.318180150000000    0.000465556000000    0.007821995000000
 O    6.790673174000000    -2.320829733000000    0.016553678000000
 O    6.789655530000000    2.321438254000000    -0.010873866000000
 C    8.749823353000000    0.000821882000000    0.018109317000000
 C    9.489787944000000    -1.154700302000000    -0.285500427000000
 H    9.009407885000000    -2.091509032000000    -0.550788722000000
 C    10.891076734000000    -1.156843950000000    -0.273387072000000
 H    11.430152330000000    -2.069512188000000    -0.515876054000000
 C    11.589889087000001    -0.000091932000000    0.045306300000000
 H    12.676317333000000    -0.000633348000000    0.056403362000000
 C    10.885877815000001    1.157334582000000    0.349730245000000
 H    11.420785634000000    2.069535935000000    0.602945519000000
 C    9.484614381000000    1.156189145000000    0.334668267000000
 H    8.999886193000000    2.093303387000000    0.590794534000000
 C    -2.281259072000000    -6.216169866000000    0.002452522000000
 C    -4.242431894000000    -5.084553735000000    0.004694686000000
 N    -3.658627828000000    -6.338001541000000    0.007821995000000
 O    -5.405190627999999    -4.720536297000000    0.016553678000000
 O    -1.384337423000000    -7.040750210000000    -0.010873866000000
 C    -4.374128884000000    -7.578025986000001    0.018109317000000
 C    -5.744822138000001    -7.641106246000000    -0.285500427000000
 H    -6.315940796000000    -6.756686054000000    -0.550788722000000
 C    -6.447311294000000    -8.853592887000000    -0.273387072000000
 H    -7.507242870000000    -8.864122150000000    -0.515876054000000
 C    -5.794929423000000    -10.037151854999999    0.045306300000000
 H    -6.338603356000000    -10.977760849999999    0.056403362000000
 C    -4.440563323000000    -10.006160405000001    0.349730245000000
 H    -3.918018821000000    -10.925499801999999    0.602945519000000
 C    -3.740935291000000    -8.792051208000000    0.334668267000000
 H    -2.687005986000000    -8.840811241000001    0.590794534000000
 C    -4.242683723000000    5.083748279000000    0.002452522000000
 C    -2.282078774000000    6.216348003000000    0.004694686000000
 N    -3.659497355000000    6.337495186000000    0.007821995000000
 O    -1.385443612000000    7.041309794000000    0.016553678000000
 O    -5.405255362000000    4.719292009000000    -0.010873866000000
 C    -4.375628311000000    7.577155962000000    0.018109317000000
 C    -3.744899793000000    8.795745578000000    -0.285500427000000
 H    -2.693409637000000    8.848129259000000    -0.550788722000000
 C    -4.443688486000000    10.010368659999999    -0.273387072000000
 H    -3.922832975000000    10.933556263000000    -0.515876054000000
 C    -5.794871313000000    10.037181066000000    0.045306300000000
 H    -6.337617138000000    10.978325899000000    0.056403362000000
 C    -6.445225706000000    8.848775761000001    0.349730245000000
 H    -7.502669165000000    8.855918190000001    0.602945519000000
 C    -5.743601261000000    7.635819182000000    0.334668267000000
 H    -6.312801358000000    6.747474783000000    0.590794534000000



