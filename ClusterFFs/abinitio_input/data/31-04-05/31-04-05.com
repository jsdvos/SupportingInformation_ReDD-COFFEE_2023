%nproc=15
%mem=70GB
%chk=ETTA_Hydrazide_Hydrazone.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 C    -0.673043696000000    -0.061281087000000    0.099838696000000
 C    0.676216304000000    0.084438913000000    -0.077921304000000
 C    -1.445563696000000    0.859668913000000    0.987758696000000
 C    -1.208593696000000    0.910808913000000    2.368608696000000
 H    -0.435593696000000    0.291168913000000    2.815748696000000
 C    -1.956453696000000    1.762468913000000    3.184708696000000
 H    -1.756043696000000    1.803278913000000    4.251888696000000
 C    -2.958213696000000    2.558248913000000    2.631708696000000
 C    -3.215433696000000    2.502618913000000    1.263248696000000
 H    -4.001643696000000    3.116068913000000    0.831638696000000
 C    -2.465153696000000    1.655188913000000    0.445338696000000
 H    -2.681203696000000    1.618718913000000    -0.619611304000000
 C    1.423956304000000    1.217358913000000    0.545288696000000
 C    1.186576304000000    2.544388913000000    0.160958696000000
 H    0.432056304000000    2.771628913000000    -0.587751304000000
 C    1.910456304000000    3.588888913000000    0.739918696000000
 H    1.711366304000000    4.615018913000000    0.443058696000000
 C    2.886286304000000    3.315958913000000    1.697168696000000
 C    3.142086304000000    1.999178913000000    2.075248696000000
 H    3.907026304000000    1.784298913000000    2.816488696000000
 C    2.417466304000000    0.953908913000000    1.499138696000000
 H    2.631376304000000    -0.068851087000000    1.799648696000000
 C    -1.412513696000000    -1.154241087000000    -0.600071304000000
 C    -1.922953696000000    -2.232601087000000    0.134698696000000
 H    -1.770443696000000    -2.283151087000000    1.209878696000000
 C    -2.626883696000000    -3.253351087000000    -0.507091304000000
 H    -3.013833696000000    -4.089711087000000    0.068518696000000
 C    -2.834543696000000    -3.198741087000000    -1.884751304000000
 C    -2.343603696000000    -2.122491087000001    -2.622201304000000
 H    -2.507323696000000    -2.077791087000000    -3.695421304000000
 C    -1.638713696000000    -1.101381087000000    -1.982691304000000
 H    -1.261323696000000    -0.266561087000000    -2.567421304000000
 C    1.439886304000000    -0.894691087000000    -0.907871304000000
 C    1.681336304000000    -2.196721087000000    -0.447761304000000
 H    1.299246304000000    -2.515401087000000    0.518518696000000
 C    2.406656304000000    -3.098111087000000    -1.228831304000000
 H    2.581246304000000    -4.108341087000000    -0.868861304000000
 C    2.903116304000000    -2.703191087000000    -2.470121304000000
 C    2.679966304000000    -1.406181087000000    -2.930431304000000
 H    3.069876304000000    -1.097111087000000    -3.896341304000000
 C    1.955636304000000    -0.502771087000000    -2.150401304000000
 H    1.790596304000000    0.507088913000000    -2.517471304000000
 C    -3.921791891000000    3.411535557000000    3.398611589000000
 O    -4.539533006000000    4.292334676000000    2.804422850000000
 N    -4.054867861000000    3.140373390000000    4.754225948000000
 H    -3.587034992000000    2.361612000000000    5.193625296000000
 N    -4.874186930000000    3.920937994000000    5.514542965000000
 C    -4.892480857000000    3.658591166000000    6.781324737000000
 C    -5.728744009000000    4.436089113000000    7.716324252000000
 H    -4.276577827000000    2.851987623000000    7.217829244000000
 C    -5.655968926000000    4.158816475000000    9.084429020000000
 C    -6.434930339000000    4.894316615000000    9.979895902999999
 H    -4.998031699000000    3.382619090000000    9.464586742000000
 C    -7.282591725000000    5.898503351000000    9.508964641000000
 H    -6.381454217000000    4.687895061000000    11.046003926999999
 C    -7.354543620000000    6.171959395000000    8.142808508000000
 H    -7.888084988000000    6.471177435000000    10.207377583000001
 C    -6.576884450999999    5.442124913000000    7.243896907000000
 H    -8.013862139000000    6.954358334000000    7.775270949000000
 H    -6.634032621000000    5.661245782000000    6.179097340000000
 C    3.786965477000000    4.416328754000000    2.169063093000000
 O    3.957205523000000    5.396375331000000    1.447260860000000
 N    4.376266156000000    4.250402185000000    3.415707192000000
 H    4.155875369000000    3.467351357000000    4.012924058000000
 N    5.249878959000000    5.192080891000000    3.872531763000000
 C    5.817486621000000    4.936541764000000    5.006735395000000
 C    6.791338749000000    5.868717299000000    5.607338437000000
 H    5.607284139000000    4.007793745000000    5.566846129000000
 C    7.429665853999999    5.510321037000000    6.798127089000000
 C    8.360309950000000    6.379540808000000    7.370805735000000
 H    7.217700552000000    4.562295054000000    7.284132352000000
 C    8.648044057000000    7.599943345000000    6.757190373000000
 H    8.864348697000000    6.106869244000000    8.294738109000001
 C    8.008183241999999    7.955875329999999    5.569659984000000
 H    9.373611294000000    8.275637782000000    7.203673909000000
 C    7.079461829000000    7.090462565000000    4.991740004000001
 H    8.232243369000001    8.905789731000000    5.090710555000000
 H    6.585062338000000    7.371123760000000    4.063367006000000
 C    -3.762310673000000    -4.169496359000000    -2.549284011000000
 O    -4.398786375000000    -3.804285458000000    -3.535180105000000
 N    -3.837767621000000    -5.441023520000000    -1.995701721000000
 H    -3.245580362000000    -5.728765017000000    -1.230896017000000
 N    -4.707400932000000    -6.348906994000000    -2.523084225000000
 C    -4.779230375000000    -7.486970625000000    -1.911898259000000
 C    -5.689263402999999    -8.552836317000001    -2.374319521000000
 H    -4.173381279000000    -7.705014442000000    -1.014186853000000
 C    -5.781326923000000    -9.735673781999999    -1.635193803000000
 C    -6.646161225000000    -10.747415014000000    -2.057145914000000
 H    -5.194162854000000    -9.879590576000000    -0.732747139000000
 C    -7.411494435000001    -10.577340302000000    -3.212151000000000
 H    -6.726876974000000    -11.668985665999999    -1.485964354000000
 C    -7.316338437000000    -9.396455339999999    -3.948896484000000
 H    -8.085271071999999    -11.365615287000001    -3.539268602000000
 C    -6.455945347000000    -8.381383463000001    -3.530797169000000
 H    -7.912584284000000    -9.262867075999999    -4.848106098000000
 H    -6.387436660000000    -7.460425121000000    -4.107182699000000
 C    3.821546363000000    -3.519296414000000    -3.327547337000000
 O    3.960137866000000    -3.213211896999999    -4.509738563000000
 N    4.467755416000000    -4.584456556000000    -2.713726387000000
 H    4.387296555000000    -4.761655089000000    -1.723506344000000
 N    5.271368201000000    -5.396761179000000    -3.457375829000000
 C    5.781262213000000    -6.411637854000000    -2.837754967000000
 C    6.659948487000000    -7.374138665000000    -3.530261884000000
 H    5.572494266000000    -6.607303655000000    -1.770696936000000
 C    7.125157890000000    -8.492827825000001    -2.833123618000000
 C    7.949078252000000    -9.415728702999999    -3.480434579000000
 H    6.851605154000000    -8.659925388000000    -1.795274412000000
 C    8.306825924000000    -9.218871139999999    -4.815335618000000
 H    8.312107027000000    -10.290674382000001    -2.946827944000000
 C    7.842486335000000    -8.100547919000000    -5.508213518000000
 H    8.947819800000000    -9.938734649000001    -5.318606632000000
 C    7.017161317000001    -7.176596075000000    -4.867428199999999
 H    8.119991100000000    -7.946866006000000    -6.548072086000000
 H    6.654713555000000    -6.307149518000000    -5.413113357000000




