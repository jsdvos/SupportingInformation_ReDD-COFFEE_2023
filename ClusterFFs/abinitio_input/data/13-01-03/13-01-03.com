%nproc=15
%mem=70GB
%chk=TrisPhenyl_BoronicAcid_Borosilicate.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

Comment

0 1
 C    5.413628598000000    -1.872546979000000    -0.010594081000000
 C    5.035722949999999    -0.755661278000000    0.753536436000000
 C    3.723673695000000    -0.300430655000000    0.759525264000000
 C    2.735803795999999    -0.944625681000000    0.000449131000000
 C    3.108315992000000    -2.059076085000000    -0.765208093000000
 C    4.420719166000000    -2.513357956000000    -0.770224134000000
 C    1.334053259000000    -0.458848890000000    0.000402830000000
 C    0.268355259000000    -1.366873890000000    0.008860830000000
 C    -1.059173741000000    -0.922488890000000    0.008337830000000
 C    -1.309241741000000    0.453656110000000    0.006809830000000
 C    -0.264218741000000    1.383363110000000    0.003920830000000
 C    1.053270259000000    0.911980110000000    -0.001354170000000
 C    -0.547166741000000    2.832910110000000    0.023508830000000
 C    0.132648259000000    3.710843110000000    -0.833951170000000
 C    -0.134981741000000    5.073690110000000    -0.817152170000000
 C    -1.095013741000000    5.616633110000000    0.054395830000000
 C    -1.773249741000000    4.731796110000000    0.910653830000000
 C    -1.504821741000000    3.368908110000000    0.897304830000000
 C    -2.182774741000000    -1.891955890000000    -0.017346170000000
 C    -3.315488741000000    -1.662366890000000    -0.812060170000000
 C    -4.366923741000000    -2.569668890000000    -0.831200170000000
 C    -4.330026741000000    -3.741083890000000    -0.056511170000000
 C    -3.194045741000000    -3.964973890000000    0.739344830000000
 C    -2.141176741000000    -3.059528890000000    0.758579830000000
 H    0.474380259000000    -2.429560890000000    -0.040814170000000
 H    1.873134259000000    1.618471110000000    0.074149830000000
 H    -2.335361741000000    0.804210110000000    -0.020704170000000
 H    0.839454259000000    3.316212110000000    -1.556644170000000
 H    -2.004881741000000    2.721095110000000    1.609926830000000
 H    0.396903259000000    5.726432110000000    -1.500808170000000
 H    -2.509284741000000    5.119474110000000    1.606548830000000
 H    4.680359519000000    -3.384593294000000    -1.361290287000000
 H    2.355934167000000    -2.588063678000000    -1.339179593000000
 H    5.783140304000000    -0.232983007000000    1.339887964000000
 H    3.464345167000000    0.579450163000000    1.337108744000000
 H    -1.287700741000000    -3.244167890000000    1.400987830000000
 H    -3.142543741000000    -4.854378890000000    1.357384830000000
 H    -5.225637741000000    -2.375433890000000    -1.464217170000000
 H    -3.358715741000000    -0.779456890000000    -1.439657170000000
 B    6.898713777000001    -2.400130116000000    0.020928054000000
 O    7.835878244000000    -1.762695787000000    0.737103987000000
 O    7.256853171000000    -3.476646662000000    -0.694478070000000
Si    9.414625050000000    -1.705119262000000    0.980919525000000
Si    8.536940512000001    -4.311200712000000    -1.162788456000000
 O    9.855399372999999    -3.124153933000000    1.569663200000000
 O    10.114811548000000    -1.554251952000000    -0.448646222000000
 H    9.781541855000000    -0.602069593000000    1.880206213000000
 O    9.543453764000001    -3.286351460000000    -1.864135055000000
 O    9.262717437999999    -4.851168526000000    0.155390302000000
 H    8.163703026000000    -5.407755177000000    -2.067481061000000
 B    9.820227557999999    -4.436775858000000    1.301689989000000
 H    10.020078549999999    -5.213927477000000    2.167404003000000
 B    10.207616808999999    -2.142236927000000    -1.649619321000000
 H    10.598386613000001    -1.517857548000000    -2.571978371000000
 B    -1.379776469000000    7.166295728000000    0.102056719000000
 O    -2.312017877000000    7.665302186000000    0.926329879000000
 O    -0.717105278000000    8.008050228000000    -0.704958694000000
Si    -3.121401361000000    9.004892146000000    1.251290893000000
Si    -0.693854394000000    9.528771916000000    -1.196164988000000
 O    -2.054752852000000    10.104364080000000    1.707342360000000
 O    -3.766174014000000    9.519107531000000    -0.118574454000000
 H    -4.147535888000000    8.779110459000000    2.278974513000000
 O    -2.157640777000000    9.877768310000000    -1.735426436000000
 O    -0.439798801000000    10.442040956000000    0.091199777000000
 H    0.329232088000000    9.745768405000000    -2.228837830000000
 B    -0.941134385000000    10.729234423999999    1.300631187000000
 H    -0.273275556000000    11.301457932000000    2.087962537000000
 B    -3.446788269000000    9.880945242999999    -1.369071054000000
 H    -4.284900191000000    9.895648686000001    -2.200124049999999
 B    -5.531888726000000    -4.760888272000000    -0.038007331000000
 O    -5.467732525000000    -5.880269127000000    0.697096888000000
 O    -6.625452846000000    -4.541965992000000    -0.783089992000000
Si    -6.215582162000000    -7.271215636000001    0.944465597000000
Si    -7.978064135000000    -5.238167410000000    -1.273156279000000
 O    -7.677846392000000    -6.932137627000000    1.493437310000000
 O    -6.402267604000000    -7.974443456000000    -0.479367076000000
 H    -5.467109395000000    -8.127891313999999    1.875034435000000
 O    -7.579911487000000    -6.633429069000000    -1.943970217000000
 O    -8.840426453999999    -5.575205895000000    0.030165981000000
 H    -8.717620988000000    -4.379362429000000    -2.208953421000000
 B    -8.788975075000000    -6.247748929000000    1.188432105000000
 H    -9.581897844000000    -6.017805099000000    2.032093845000000
 B    -6.928711855000000    -7.778310831000000    -1.696356794000000
 H    -6.562789476000000    -8.443489847000000    -2.600201115000000



