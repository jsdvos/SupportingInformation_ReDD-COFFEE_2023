%nproc=15
%mem=70GB
%chk=ETTA_BoronicAcid_BoronateEster.chk
#P B3LYP/6-311++G(d,p) EmpiricalDispersion=GD3 opt SCF=YQC

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
 B    -3.923705512000000    3.393203134000000    3.490645890000000
 O    -3.816290194000000    3.557028817000000    4.869711841000000
 O    -5.024395349000000    4.088360336000000    2.995246704000000
 C    -4.870531166000000    4.368100134000000    5.229553433000000
 C    -5.605059037000001    4.691150318000000    4.089882453000000
 C    -5.219190815000000    4.833755505000000    6.482553143000000
 C    -6.726135406000000    5.496519639000000    4.144424004000000
 C    -6.354273419000000    5.651717346000000    6.552312198000000
 C    -7.091058906000000    5.975760445000000    5.409138371000000
 H    -4.640552005000000    4.575757143000000    7.360194001000000
 H    -7.288261932000000    5.740238434000001    3.252094384000000
 H    -6.666310015000000    6.040698033999999    7.514225765000000
 H    -7.963728854999999    6.611311099000000    5.501188841000000
 B    3.828626243000000    4.398573831000000    2.251277420000000
 O    4.817062600000000    4.173922738000000    3.206643976000000
 O    3.809655993000000    5.736991925000000    1.865973457000000
 C    5.417461973000000    5.396901139000000    3.413020091000000
 C    4.804961571000000    6.347246860000000    2.597895055000000
 C    6.456759327000000    5.721231594000000    4.263359739000000
 C    5.200164165000000    7.670944328000000    2.591063407000000
 C    6.867676342000000    7.060439555000000    4.266467920000000
 C    6.253293382999999    8.013706221000000    3.448837548000000
 H    6.924324046000000    4.974002453000000    4.891588476000000
 H    4.716483987000000    8.399658006999999    1.953358858000000
 H    7.679575434000000    7.361971634000000    4.917504626000000
 H    6.597696274000000    9.040596192000001    3.477723514000000
 B    -3.768174330000000    -4.252332679000000    -2.505536852000000
 O    -4.290549698000000    -5.347840130000000    -1.822025916000000
 O    -4.206301015000000    -4.240745389000000    -3.827691438000000
 C    -5.064513827000000    -6.022894073000000    -2.740736087000000
 C    -5.013292214000000    -5.349783593000000    -3.960179068000000
 C    -5.802065186000000    -7.178737417000000    -2.572062120000000
 C    -5.696979817000001    -5.797795675000000    -5.073850108000000
 C    -6.500467681000000    -7.644301157000000    -3.693589996000000
 C    -6.449088636000000    -6.969121830000000    -4.916781009000000
 H    -5.833855545000000    -7.692416120999999    -1.619821045000000
 H    -5.649222576000000    -5.266098930000000    -6.015473733000000
 H    -7.092822401000000    -8.547586711999999    -3.609081920000000
 H    -7.002347463000000    -7.358650009000000    -5.763021654000000
 B    3.858938630000000    -3.593196166000000    -3.283380637000000
 O    4.185559155000000    -4.907647165000000    -2.958207755000000
 O    4.515648720000000    -3.194373274000000    -4.445212882000000
 C    5.059522484000000    -5.325386429000000    -3.938222390000000
 C    5.260218016000000    -4.283719564000000    -4.842320836000000
 C    5.681508196000000    -6.550503029000000    -4.081853309000000
 C    6.093251655000000    -4.413437424000000    -5.936685941000000
 C    6.529217281000000    -6.695774746000000    -5.187577440000000
 C    6.730529663000000    -5.650906254000000    -6.094454688000000
 H    5.517487982000000    -7.351775220999999    -3.372840088000000
 H    6.240925302000000    -3.596945084000000    -6.631790239000000
 H    7.039047977000000    -7.639347941000000    -5.341502128000000
 H    7.393542129000000    -5.799418717000000    -6.938438859000001




