from yaff import System, ForceField, StrainCellDOF, CGOptimizer, angstrom

def get_ff_kwargs():
    # Return force field arguments
    kwargs = {'rcut': 11.0*angstrom,
              'alpha_scale': 2.86,
              'gcut_scale': 1.0,
              'smooth_ei': True,
              'tailcorrections': True}
    return kwargs

for cof in ['COF-300', 'LZU-111', 'COF-320_89K', 'COF-320_298K']:
    for ff_label, fn_pars in (['qff', 'pars_cluster.txt'], ['uff', 'pars_uff.txt']):
    sys = System.from_file('../data/{}/{}_exp.chk'.format(cof, cof))
    ff = ForceField.generate(sys, '../data/{}/{}'.format(cof, fn_pars), **get_ff_kwargs())
    dof = StrainCellDOF(ff)
    opt = CGOptimizer(dof)
    opt.run()
    opt.dof.ff.system.to_file('{}_{}_opt.chk'.format(cof, ff_label))


