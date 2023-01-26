%nproc=15
%mem=70GB
%chk=ETTA_Nitrile_Triazine.chk
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
 C    -3.886879675000000    3.361355527000000    3.457880006000000
 N    -4.910543459000000    3.981378950000000    2.845004569000000
 C    -5.708867476000000    4.686581376000000    3.640319095000000
 N    -3.662276237000000    3.432375054000000    4.781733188000000
 C    -4.530251240999999    4.168197414000000    5.468904540000000
 N    -5.578000198000000    4.823847681000000    4.962458425000000
 H    -4.372428486000000    4.242848014000000    6.541291324000000
 H    -6.544639998000000    5.198218815000000    3.171057217000000
 C    3.792680309000000    4.357280981000000    2.230139641000000
 N    3.662239155000000    5.612788199000000    1.766869071000000
 C    4.504509328000000    6.501942961000001    2.283726554000000
 N    4.703098058000000    3.997760173000000    3.152073498000000
 C    5.487232915000000    4.977035956000000    3.591585825000000
 N    5.443319359000000    6.253543370000000    3.200762027000000
 H    6.232348388000000    4.714365240000000    4.337441827000000
 H    4.421083664000000    7.524805333000000    1.926952096000000
 C    -3.732561936000000    -4.212143034000000    -2.481860594000000
 N    -4.075415848000000    -4.076244133000000    -3.774886337000000
 C    -4.883533865000000    -5.017396536000000    -4.252729740000000
 N    -4.162443642000000    -5.220101059000000    -1.702566000000000
 C    -4.965672832000000    -6.097364786000000    -2.296073239000000
 N    -5.367915959000000    -6.057654158000000    -3.569166461000000
 H    -5.328107904000000    -6.922533747000000    -1.689252711000000
 H    -5.176674444000000    -4.932035809000000    -5.295455704000000
 C    3.822480357000000    -3.559245465000000    -3.252361937000000
 N    4.388563473000000    -3.042497945000000    -4.357103896000000
 C    5.203876941000001    -3.859971918000000    -5.016009916000000
 N    4.047485498000000    -4.812678815000000    -2.820673101000000
 C    4.881807780000000    -5.531305875000000    -3.565321407000000
 N    5.496692077000000    -5.118237321000000    -4.676822414000000
 H    5.081649268000000    -6.548314994000000    -3.239361316000000
 H    5.675194240000000    -3.467903153000000    -5.913025733000000



