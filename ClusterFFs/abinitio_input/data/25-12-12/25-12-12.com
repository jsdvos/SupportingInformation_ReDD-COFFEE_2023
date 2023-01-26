%nproc=15
%mem=70GB
%chk=Pc_CarboxylicAnhydride_Imide.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC NoSymm

Comment

0 1
 N    2.352349000000000    2.353581000000000    -0.106935000000000
 C    6.545223000000000    -0.623519000000000    -0.004651000000000
 C    6.506949000000000    0.766356000000000    -0.044129000000000
 C    5.306615000000000    1.457271000000000    -0.077805000000000
 C    0.694402000000000    4.134140000000000    -0.072345000000000
 C    1.449041000000000    5.302996000000000    -0.081392000000000
 C    0.768886000000000    6.499946000000000    -0.043424000000000
 C    -0.626235000000000    6.535277000000000    -0.005295000000000
 H    5.286896000000000    2.448611000000000    -0.109470000000000
 H    2.441236000000000    5.283090000000000    -0.114024000000000
 C    1.088817000000000    2.739747000000000    -0.086507000000000
 C    2.736369000000000    1.084715000000000    -0.079459000000000
 C    5.384667000000000    -1.376044000000000    0.003193000000000
 C    4.142797000000000    0.701090000000000    -0.064827000000000
 C    -0.702135000000000    4.169767000000000    -0.039486000000000
 C    -1.384691000000000    5.381008000000000    -0.007912000000000
 N    -0.045630000000000    1.976161000000000    -0.050973000000000
 N    1.967433000000000    -0.039198000000000    -0.045501000000000
 C    4.174325000000000    -0.691644000000000    -0.027748000000000
 H    5.418593000000000    -2.371732000000000    0.032937000000000
 C    -1.152033000000000    2.783343000000000    -0.027790000000000
 H    -2.386427000000000    5.423716000000000    0.012658000000000
 C    2.783446000000000    -1.134264000000000    -0.023299000000000
 N    -2.427713000000000    2.413713000000000    -0.001479000000000
 N    2.427713000000000    -2.413713000000000    0.001480000000000
 C    -2.783447000000000    1.134264000000000    0.023299000000000
 C    1.152032000000000    -2.783343000000000    0.027790000000000
 C    -4.174325000000000    0.691644000000000    0.027749000000000
 N    -1.967432000000000    0.039195000000000    0.045502000000000
 C    0.702135000000000    -4.169766000000000    0.039488000000000
 N    0.045631000000000    -1.976159000000000    0.050973000000000
 C    -5.384669000000000    1.376042000000000    -0.003193000000000
 C    -4.142797000000000    -0.701091000000000    0.064827000000000
 C    -2.736369000000000    -1.084715000000000    0.079459000000000
 C    -0.694402000000000    -4.134140000000000    0.072346000000000
 C    1.384691000000000    -5.381008000000000    0.007912000000000
 C    -1.088817000000000    -2.739746000000000    0.086509000000000
 C    -6.545223000000000    0.623518000000000    0.004651000000000
 H    -5.418594000000000    2.371732000000000    -0.032936000000000
 C    -5.306616000000000    -1.457273000000000    0.077805000000000
 N    -2.352349000000000    -2.353581000000000    0.106936000000000
 C    -1.449040000000000    -5.302996000000000    0.081393000000000
 C    0.626234000000000    -6.535278000000000    0.005295000000000
 H    2.386428000000000    -5.423717000000001    -0.012658000000000
 C    -6.506949000000000    -0.766357000000000    0.044129000000000
 H    -5.286899000000000    -2.448610000000000    0.109470000000000
 C    -0.768887000000000    -6.499946000000000    0.043426000000000
 H    -2.441237000000000    -5.283089000000000    0.114025000000000
 H    1.073860000000000    -0.054754000000000    -0.036968000000000
 H    -1.073859000000000    0.054753000000000    0.036969000000000
 C    7.901963953000000    1.219885779000000    -0.025238407000000
 C    7.928744718000000    -1.044187628000000    -0.021587518000000
 N    8.709210107000001    0.097233092000000    -0.020648294000000
 O    8.208636829000000    -2.230008506000000    -0.009485561000000
 O    8.153809618000000    2.411917849000000    -0.039801542000000
 C    10.140769856000000    0.114173104000000    -0.013036576000000
 C    10.893499633999999    -1.032898873000000    -0.317309071000000
 H    10.423507281999999    -1.975377859000000    -0.581115867000000
 C    12.294739407000000    -1.018808966000000    -0.307813042000000
 H    12.843895473000000    -1.925330059000000    -0.550740693000000
 C    12.980701441000001    0.146162163000000    0.008853595000000
 H    14.067081882000000    0.158207750000000    0.017920725000000
 C    12.263901970999999    1.295555171000000    0.313871916000000
 H    12.788680819000000    2.214052038000000    0.565519139000000
 C    10.862720152000000    1.278174795000000    0.301429270000000
 H    10.367650050000000    2.209778324999999    0.557877398000000
 C    -1.046755279000000    7.918294097000000    -0.025208633000000
 C    1.217361978000000    7.895520257000000    -0.021557648000000
 N    0.093294744000000    8.700778189999999    -0.020618574000000
 O    2.409024166000000    8.149394199000000    -0.009455674000000
 O    -2.232990337000000    8.196166510999999    -0.039771853000000
 C    0.107687807000000    10.132365812000000    -0.013007039000000
 C    1.270958219000000    10.859812141000001    -0.317279580000000
 H    2.202925933000000    10.369306602000000    -0.581086274000000
 C    1.287537165000000    12.261024675999998    -0.307783730000000
 H    2.205859185000000    12.790210397999999    -0.550711411000000
 C    0.137856987000000    12.972317315000000    0.008882768000000
 H    0.149589215000000    14.058701187000000    0.017949759000000
 C    -1.026947582000000    12.280843486000000    0.313901130000000
 H    -1.933739943000000    12.825597560000000    0.565548245000000
 C    -1.040235579000000    10.879616879000000    0.301458664000000
 H    -1.982450384000000    10.405053087000001    0.557906813000000
 C    -7.903384388000000    -1.218167818000000    0.030143342000000
 C    -7.927336446000000    1.045940072000000    0.030975894000000
 N    -8.709211442000001    -0.094503527000000    0.036291407000000
 O    -8.205707009999999    2.232123918000000    0.042591310000000
 O    -8.156764389999999    -2.409902355000000    0.018054112000000
 C    -10.140752873000000    -0.109647033000000    0.049253737000000
 C    -10.893183816000001    1.037971831000000    -0.253692813000000
 H    -10.423005726000000    1.979523939000000    -0.520461754000000
 C    -12.294394944000000    1.025643003000000    -0.238962724000000
 H    -12.843324369999999    1.932536070000000    -0.481012659000000
 C    -12.980620823000001    -0.138063461000000    0.081756326000000
 H    -14.066974102000000    -0.148741492000000    0.094882853000000
 C    -12.264120021000000    -1.287957550000000    0.385585435000000
 H    -12.789099649000001    -2.205474815000000    0.640367924000000
 C    -10.862973728000000    -1.272341970000000    0.367904727000000
 H    -10.368110683999999    -2.204232568000000    0.623708690000000
 C    1.045006524000000    -7.919676690000000    0.030113568000000
 C    -1.219083301000000    -7.894073872000000    0.030946023000000
 N    -0.096023871000000    -8.700719785000000    0.036261687000000
 O    -2.411075094000000    -8.146418583000001    0.042561423000000
 O    2.230910519000000    -8.199076667000000    0.018024423000000
 C    -0.112212733000000    -10.132249774000000    0.049224200000000
 C    -1.276023397000000    -10.859385349000000    -0.253722305000000
 H    -2.207060327000000    -10.368714389999999    -0.520491347000000
 C    -1.294362496000000    -12.260530701000000    -0.238992036000000
 H    -2.213051466000000    -12.789481681000000    -0.481041941000000
 C    -0.145952541000000    -12.972059463000001    0.081727153000000
 H    -0.159051496000000    -14.058386246000000    0.094853819000000
 C    1.019346487000000    -12.280895215999999    0.385556221000000
 H    1.925155019000000    -12.825828596999999    0.640338818000000
 C    1.034398201000000    -10.879742754000000    0.367875333000000
 H    1.976895492000000    -10.405392267000000    0.623679275000000



