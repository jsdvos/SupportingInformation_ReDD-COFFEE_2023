import numpy as np

from yaff import *

class LJCrossGenerator(NonbondedGenerator):
    prefix = 'LJCROSS'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 2)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        nffatypes = system.ffatype_ids.max() + 1
        sigmas = np.zeros([nffatypes, nffatypes])
        epsilons = np.zeros([nffatypes, nffatypes])
        for i in range(system.natom):
            for j in range(system.natom):
                ffa_i, ffa_j = system.ffatype_ids[i], system.ffatype_ids[j]
                key = (system.get_ffatype(i), system.get_ffatype(j))
                par_list = par_table.get(key, [])
                if len(par_list) > 2:
                    raise TypeError('Superposition should not be allowed for non-covalent terms.')
                elif len(par_list) == 1:
                    sigmas[ffa_i,ffa_j], epsilons[ffa_i,ffa_j] = par_list[0]
                elif len(par_list) == 0:
                    if log.do_high:
                        log('No LJCross parameters found for ffatypes %s,%s. Parameters set to zero.' % (system.ffatypes[i0], system.ffatypes[i1]))

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotLJCross)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJCross part should not be present yet.')

        pair_pot = PairPotLJCross(system.ffatype_ids, epsilons, sigmas, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)

def apply_generators(system, parameters, ff_args):
    '''Populate the attributes of ff_args, prepares arguments for ForceField

       **Arguments:**

       system
            A System instance for which the force field object is being made

       ff_args
            An instance of the FFArgs class.

       parameters
            An instance of the Parameters, typically made by
            ``Parameters.from_file('parameters.txt')``.
    '''

    # Collect all the generators that have a prefix.
    import yaff.pes.generator as yaff_gen
    generators = {}

    for name, gen in yaff_gen.__dict__.items():
        if isinstance(gen, type) and issubclass(gen, Generator) and gen.prefix is not None:
            if name == 'LJCrossGenerator':
                generators[LJCrossGenerator.prefix] = LJCrossGenerator()
            else:
                generators[gen.prefix] = gen()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if log.do_warning:
                log.warn('There is no generator named %s. It will be ignored.' % prefix)
        else:
            generator(system, section, ff_args)

    # If tail corrections are requested, go through all parts and add when necessary
    if ff_args.tailcorrections:
        if system.cell.nvec==0:
            log.warn('Tail corrections were requested, but this makes no sense for non-periodic system. Not adding tail corrections...')
        elif system.cell.nvec==3:
            for part in ff_args.parts:
                # Only add tail correction to pair potentials
                if isinstance(part,ForcePartPair):
                    # Don't add tail corrections to electrostatic parts whose
                    # long-range interactions are treated using for instance Ewald
                    if isinstance(part.pair_pot,PairPotEI) or isinstance(part.pair_pot,PairPotEIDip):
                        continue
                    else:
                        part_tailcorrection = ForcePartTailCorrection(system, part)
                        ff_args.parts.append(part_tailcorrection)
        else:
            raise ValueError('Tail corrections not available for 1-D and 2-D periodic systems')

    part_valence = ff_args.get_part(ForcePartValence)
    if part_valence is not None and log.do_warning:
        # Basic check for missing terms
        groups = set([])
        nv = part_valence.vlist.nv
        for iv in range(nv):
            # Get the atoms in the energy term.
            atoms = part_valence.vlist.lookup_atoms(iv)
            # Reduce it to a set of atom indices.
            atoms = frozenset(sum(sum(atoms, []), []))
            # Keep all two- and three-body terms.
            if len(atoms) <= 3:
                groups.add(atoms)
        # Check if some are missing
        for i0, i1 in system.iter_bonds():
            if frozenset([i0, i1]) not in groups:
                log.warn('No covalent two-body term for atoms ({}, {})'.format(i0, i1))
        for i0, i1, i2 in system.iter_angles():
            if frozenset([i0, i1, i2]) not in groups:
                log.warn('No covalent three-body term for atoms ({}, {} {})'.format(i0, i1, i2))


class ForceField(ForceField):
    @classmethod
    def generate(cls, system, parameters, **kwargs):
        """Create a force field for the given system with the given parameters.

           **Arguments:**

           system
                An instance of the System class

           parameters
                Three types are accepted: (i) the filename of the parameter
                file, which is a text file that adheres to YAFF parameter
                format, (ii) a list of such filenames, or (iii) an instance of
                the Parameters class.

           See the constructor of the :class:`yaff.pes.generator.FFArgs` class
           for the available optional arguments.

           This method takes care of setting up the FF object, and configuring
           all the necessary FF parts. This is a lot easier than creating an FF
           with the default constructor. Parameters for atom types that are not
           present in the system, are simply ignored.
        """
        if system.ffatype_ids is None:
            raise ValueError('The generators needs ffatype_ids in the system object.')
        with log.section('GEN'), timer.section('Generator'):
            from yaff.pes.parameters import Parameters
            if log.do_medium:
                log('Generating force field from %s' % str(parameters))
            if not isinstance(parameters, Parameters):
                parameters = Parameters.from_file(parameters)
            ff_args = FFArgs(**kwargs)
            apply_generators(system, parameters, ff_args)
            return ForceField(system, ff_args.parts, ff_args.nlist)


