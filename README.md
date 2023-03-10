This file relates to the research done in the following paper:

J. S. De Vos, S. Borgmans, P. Van Der Voort, S. M. J. Rogge, V. Van Speybroeck, _ReDD-COFFEE: A ready-to-use database of covalent organic framework structures and accurate force fields to enable high-throughput screenings_ (2023)

For the structures and force fields of the ReDD-COFFEE database, which are not included in this storage, we refer to its landing page on the Materials Cloud:
https://doi.org/doi.org/10.24435/materialscloud:nw-3j

This file is part of the midterm storage (publically available under the [CC BY-SA license](https://creativecommons.org/licenses/by-sa/4.0/)) of the input files relevant in this work. In the remainder of this file, we outline in detail the workflow used in this work including references to the relevant input and output files.

# Software
The following software packages are used to perform all relevant calculations.

- Gaussian (Gaussian 16, Revision C.01)
- HORTON (version 2.0.0)
- Molden (version 6.8)
- QuickFF (version 2.2.4)
- TAMkin (version 1.2.6)
- Yaff (version 1.6.0)
- Zeo++ (version 0.3)
- RASPA (version 2.0.41)
- Shapely (version 1.8.2)
- Scikit-learn (version 1.0.1)
- Dask (version 2021.03.0)

# Workflow

The SBU nomenclature within this data archive is somewhat different than the nomenclature used in the manuscript. More specific, the following adaptions have to be made to the core linkers used in this data archive (data archive core linker id -> manuscript core linker id): 28 -> 26, 29 -> 27, 30 -> 28, 31 -> 29, 32 -> 30, 33 -> 31, 34 -> 32, 35 -> 34, 36 -> 33. Furthermore, the imide terminations are also given a separate id (data archive termination id -> manuscript termination id): 03-12 -> 03-07, 12-12 -> 09-07.

## STEP 1 - Cluster force field development

### Step 1a - QuickFF cluster force fields
#### (i) *Ab initio* input

The *ab initio* hessian for each SBU cluster is calculated with Gaussian (Gaussian 16, Revision C.01). For each molecular building block, an XYZ file is generated with Avogadro and converted to a Gaussian COM file using the `g16-xyztocom.py` script. The cluster is optimized with the B3LYP functional with additional Grimme D3 dispersion correction. If the optimization does not converge (as seen in the log file without `Normal termination`), this procedure is repeated, using the `g16-restart.py` script to create a new Gaussian COM file. Afterwards, the hessian is computed by a frequency calculation. For each COM file, a corresponding job script can be created using the `js-g16.py` script.

**input**
`SBU.xyz`

**command line**
`python g16-xyztocom.py SBU.xyz`
`python js-g16.py SBU.com`
`qsub SBU.sh`
`python g16-restart.py SBU.log`
`python g16-restart.py -j 'freq(noraman)' -o SBU_freq.com SBU.com`

**output**
`SBU.com`, `SBU.sh`, `SBU.fchk`, `SBU.log`
`SBU_b3lyp.com`, `SBU_b3lyp.sh`, `SBU_b3lyp.fchk`, `SBU_b3lyp.log`
`SBU_freq.com`, `SBU_freq.sh`, `SBU_freq.fchk`, `SBU_freq.log`


#### (ii) Charge partitioning
The fixed atomic charges are estimated with the Minimal Basis Iterative Stockholder partitioning scheme using HORTON (version 2.0.0). The only required input is the Gaussian FCHK file generated in **Step 1a _(i)_**. HORTON then generates the output HDF5 file containing the raw charges for each individual atom.

**input**
`SBU_freq.fchk`

**command line**
`horton-wpart.py --grid=veryfine SBU_freq.fchk horton_out.h5 mbis > horton.log`

**output**
`horton_out.h5`, `horton.log`

#### (iii) QuickFF
- **Atom type assignation**

Starting from the `SBU_freq.fchk` file, CON3F (`c3f.py`) was used to create a CHK file, from which the bonds are detected using the detect_bonds() method in the `molmod` package. The atom types are defined as such that each set of equivalent atoms is given a unique atom type. The atom types of the atoms in the termination of a cluster end with `_term`, whereas the other atom types are given a suffix identifying the SBU itself (*e.g.*, `_01-01-01` for SBU 01-01-01). The atom types are given in the `SBU_ffatypes.txt` files, which are used by the `read_atomtypes.py` script to generate the CHK file `system_ffatype.chk` that contains the atom types. Alternatively, as is done for the four SBUs emerging during synthesis and the SBUs with an extended termination, the atom types can be defined using atomic filters, which are defined in a separate `get_ffatype_filters.py` script for each of these SBU. In this case, the CHK file is generated by the `define_atomtypes.py` script.

**input**
`SBU_freq.fchk`
`SBU_ffatypes.txt` or `get_ffatype_filters.py`

**command line**
`c3f.py convert SBU_freq.fchk system.chk`
`python read_atomtypes.py system.chk SBU_ffatypes.txt > ffatypes.log`
`python define_atomtypes.py system.chk get_ffatype_filters.py > ffatypes.log`

**output**
`system.chk`, `system_ffatype.chk`

- **Conversion of atomic charges to Yaff parameter file for electrostatics**

Then, the script `qff-ei-input.py` (part of the QuickFF (version 2.2.4) package) is applied to generate a Yaff (version 1.6.0) parameter file containing the electrostatic contribution to the force field in terms of charges of atom types.  Next to the Horton HDF5 output (`horton_out.h5`), this script also requires some user-specified keyword arguments:

* `--bci`
To convert averaged atomic charges to bond charge increments
* `--gaussian`
To characterize the atomic charges as gaussian distributed charges with radii taken from Chen and Martinez.

**input**
`system_ffatype.chk`, `horton_out.h5`

**command line**
`qff-input-ei.py --bci --gaussian system_ffatype.chk horton_out.h5:/charges`

**output**
`pars_ei.txt`

- **Description of the Van der Waals interactions**

The Van der Waals interactions are described using a Buckingham potential with the MM3 parameters from Allinger *et al.* To obtain these parameters, the MM3 atom types are identified using the Tinker package as implemented in Molden (version 6.8) and written to `mm3.xyz`. Once these atom types are identified, the parameters are read and converted to the Yaff parameter file `pars_mm3.txt`using the `mm3_from_tinker.py` script.

**input**
`system_ffatype.chk`, `mm3.xyz`

**command line**
`python mm3_from_tinker.py system_ffatype.chk mm3.xyz`

**output**
`pars_mm3.txt`

- **Derivation of covalent force field terms**

Finally, a covalent force field is estimated using QuickFF using `system_ffatype.chk` (containing the equilibrium geometry and atom types) , `SBU_freq.fchk` (containing the *ab initio* hessian) and the reference force field parameter files `pars_ei.txt` and `pars_mm3.txt` (containing the electrostatic and Van der Waals contributions, respectively) as input.

To this end, the script `qff-derive-cov.py` is executed (using the QuickFF configuration defined in `config_quickff.txt`). More information on this configuration files can be found in the [online documentation of QuickFF](http://molmod.github.io/QuickFF/ug.html). QuickFF then generates as output the covalent force field (`pars_yaff.txt`), the system file (`system.chk` that can be used by Yaff) and a log file (`quickff.log`).

As specified in the `config_quickff.txt` file, QuickFF will also generate the `trajectories.pp` file. This is an intermediary output file (containing the perturbation trajectories in a pickled (binary) file). However, this file has no longterm storage value and is therefore omitted in here.

**input**
`config.txt`, `system_ffatype.chk`, `SBU_freq.fchk`, `pars_ei.txt`, `pars_mm3.txt`

**command line**
`qff-derive-cov.py -c config_quickff.txt -e pars_ei.txt -m pars_mm3.txt -p system_ffatype.chk SBU_freq.fchk > quickff.log`

**output**
`pars_yaff.txt`, `quickff.log`, `system.chk`

### Practical implementation
The procedures outlined in Step 1a _(ii)_ and Step 1a _(iii)_ are automatically performed by a single bash script, `derive_ff.sh` for convenience.

### Step 1b - Additional force field terms

The cluster force fields, and in turn the derived periodic force field, can give rise to significant deviations between the *ab initio* cluster model and the optimal force field geometry. When considering the rotation of the triazine ring in the SBUs involved in a triazine linkage, it was deemed appropriate to add an additional term. To this end, an *ab inito* rotation scan was performed to fit this additional term, replacing the original torsion term. This procedure is outlined in the `ClusterFFs/rotational_barriers/rotational_barrier_redd-coffee.ipynb` file, using a pyiron workflow (https://pyiron.org/). Clear stepwise instructions are evident from the sequential notebook headers, and, in essence, results in the following input/output:

**input**
`SBU_freq.fchk`, `SBU_freq.chk`, `system.chk`, `ffpars/pars_*.txt`

**command line**
see notebook

**output**
`fits/`, `polysix/pars_polysix.txt`, `new_ffpars/pars_*.txt`

The input `*.chk` structures serve as the reference ab initio geometry and force field geometry, where the latter also contains the relevant atom types. As output, the notebook provides illustrations of the rotational barrier fitting, showcasing the behaviour of the old force field, versus the new force field. Additionally, the new torsion term is printed separately in `polysix/pars_polysix.txt`, and the new force field is also provided in full under `new_ffpars/pars_*.txt`, where the faulty dihedral term was replaced by the newly fitted term. 

### Step 1c - UFF cluster force fields

Besides the system-specific QuickFF cluster force fields, the fully transferable UFF force field is used to derive an additional force field for the clusters. This is used for an initial optimization of the periodic structure to avoid collapse of closely placed atoms, which would be troublesome to describe with the Buckingham potential in the Van der Waals part of the QuickFF force field. As UFF uses a Lennard-Jones potential, nearly overlapping atoms can be pulled apart during an optimization.

This UFF force field is generated using the `create_uff.py` and `uff.py` scripts, which identify the UFF atom types and assign the correct parameters. The detected atom types and bond orders, from which the UFF parameters are derived, are written out to the `uff_ffatypes.txt` and `uff_bonds.txt`files. If necessary, the atom types and bond orders can be redefined in these files, which are read in in subsequent runs of the `create_uff.py` script. The parameters are written in the Yaff parameter files `pars_cov_uff.txt` and `pars_lj_uff.txt`.

**input**
`system.chk`

**command line**
`python create_uff.py -f uff_ffatypes.txt -b uff_bonds.txt -c pars_cov_uff.txt -l pars_lj_uff.txt system.chk`

**output**
`pars_cov_uff.txt`, `pars_lj_uff.txt`, `uff_ffatypes.txt`, `uff_bonds.txt`

### Step 1d - Validation

In order to validate each cluster force field, the rest values of the force field and *ab initio* optimized systems are compared. Furthermore, a basic frequency analysis is performed. A similar approach is followed for both the QuickFF and UFF force fields. First, the *ab initio* optimized systems are relaxed using the derived force fields with the `opt_nma.py`script, ensuring that only positive frequencies are obtained with a normal mode analysis (NMA) using TAMkin (version 1.2.6). This script also writes out the normal mode frequencies to TXT files.

**input**
`SBU_opt_ai.chk`, `pars_yaff.txt`, `pars_ei.txt`, `pars_mm3.txt`, `pars_cov_uff.txt`, `pars_lj_uff.txt`

**command line**
`python opt_nma.py > optimization.log`

**output**
`SBU_opt_qff.chk`, `SBU_opt_uff.chk`, `SBU_freqs_qff.txt`, `SBU_freqs_uff.txt`, `optimization.log`

- **Rest value comparison**

The rest values (bonds, bends, dihedral angles and out-of-plane distances) for both the *ab initio* and force field optimized clusters are determined and compared with each other using the `print_rvs.py` and `plot_rvs.py` scripts. The former prints all values of the internal coordinates in TXT files, which are read by the latter and used to make figures to show the agreement of the QuickFF and UFF optimized geometries with the *ab initio* relaxed clusters. These figures are plotted in Fig. S24 of the ESI of the paper.

**input**
`SBU_opt_ai.chk`, `SBU_opt_qff.chk`, `SBU_opt_uff.chk`

**command line**
`python print_rvs.py`
`python plot_rvs.py`

**output**
`SBU_rvs_qff_bond.txt`, `SBU_rvs_qff_bend.txt`, `SBU_rvs_qff_dihed.txt`, `SBU_rvs_qff_oop.txt`
`SBU_rvs_uff_bond.txt`, `SBU_rvs_uff_bend.txt`, `SBU_rvs_uff_dihed.txt`, `SBU_rvs_uff_oop.txt`
`qff_bond.pdf`, `qff_bend.pdf`, `qff_dihed.pdf`, `qff_oop.pdf`
`uff_bond.pdf`, `uff_bend.pdf`, `uff_dihed.pdf`, `uff_oop.pdf`

- **Frequency comparison**

The *ab initio* vibrational frequencies are obtained from the *ab initio* Hessian and are stored in `SBU_freqs_ai.txt`. They are compared with the force field frequencies, which are previously derived using the `opt_nma.py` script, in Fig. S25 of the ESI. This figure is created using the `plot_freqs.py` script.

**input**
`SBU_freqs_ai.txt`, `SBU_freqs_qff.txt`, `SBU_freqs_uff.txt`

**command line**
`python plot_freqs.py`

**output**
`qff_freqs.pdf`,  `uff_freqs.pdf`

## Step 2: Database generation

As discussed in the manuscript, the structures in ReDD-COFFEE are generated using an additive top-down approach, following a four-step procedure. The practical implementation of these four steps is outlined below. The first three steps tackle the initial structure generation, discussed in Step 2a below. The last step, _i.e._, the structure optimization, is described in Step 2b.

### Step 2a: Structure assembly

Five core modules are implemented to generate the initial COF structures:

- `SBU.py`
The implementation of the SBUs is described in this module, containing mainly the `SBU` class. This class supports several methods to transform the SBUs internal geometry. All `SBU`s are read in from two input files, which are provided in the `SBUs` folder: `SBU.chk` and `SBU.sbu`. `SBU.chk` contains the cluster geometry and the force field atomtypes, whereas information on the center of the SBU and the points of extension is provided in `SBU.sbu`.

- `Topology.py`
The `Topology` class contains a set of `WyckoffNode`s that are collected in a periodic unit cell. Each `WyckoffNode` consists of a number of `Node`s. The module furthermore allows to rescale the topological unit cell, and an implementation is provided for the breadth-first iteration. The topologies used in this work are stored in `topology.top` files, which are collected in the `Topology` folder

- `ParametersCombination.py`
This module contains all required algorithms to combine multiple force field parameter sets. The cluster force fields are thus combined to generate the periodic force field.

- `Construct.py`
All machinery to construct an initial periodic structure from a `Topology` and a set of `SBU`s is provided in this module.

- `Database.py`
This module supports the construction of a `Database`. Several methods are implemented to do the initial enumeration of (topology, SBUs) combinations, adopt the filters provided in the manuscript and electronic supplementary information, and provide the remaining combinations to the `Construct.py` module.

The `build_database.py`  script allows to reconstruct the initial structures of the database. First, all 5 537 951 initial combinations are enlisted and their rescaling factor, rescaling standard deviation, and largest root-mean-square deviation is calculated. These properties are stored in the `database_resc_rmsd.txt` file. This file can be restored by running:

`cat database_resc_rmsd.txt.tar.gz.* | tar xvzf -`

Once the first two filters are applied, the number of atoms and the initial unit cell volume is calculated for the remaining 403 581 combinations. Finally, the initial structure of the remaining 347 055 combinations are generated.

### Step 2b: Structure optimization

Once all initial structures are generated, they are optimized with their respective force fields. All required scripts to automize this procedure are provided in the `scripts/optimization` folder. The `job.sh` is adopted as job script, which requires the folder and name of the structure to be specified. If the LAMMPS force field turns out to be not applicable anymore for the optimization, an additional argument `only_yaff=True` can be provided to turn off the interface with LAMMPS. Furthermore, multiple optimizations can be grouped in one batch job using the input argument. This file contains the folfders and structure names of multiple structures. The jobs are parallelized by dask.

`qsub -v folder=folder,name=struct,only_yaff=False job.sh`
`qsub -v input=batch_file.txt,only_yaff=False job.sh`

The `job.sh` script provides the arguments to the `run_dask.py` script, which initiates one (or more) `StructureOptimizations`. A `StructureOptimization` is a class object that automates the structural relaxations, and is implemented in the `core.py` script. The remaining scripts, _i.e._, `utils_application.py`, `utils_log.py`, `yaff_ff.py`, `yaff_lammpsio.py`, `yaff_liblammps.py`, and `yaff_system.py` reimplement Yaff modules to avoid that output of parallel optimization jobs is written to the same file.

#### Calculation of the deformation energy

Once a structure is optimized, the deformation energy that checks if the structure is not too largely deformed can be calculated by running `python e_def.py`. The deformation energies of each of the 313 909 optimized structures are stored in the `database_edef.txt` file

## Step 3: Database analysis

### Step 3a: Diversity metrics

The diversity metrics of five COF databases are calculated. The preprocessed structures of all databases can be found in the `databases` folder. To untar the databases of Martin _et al._ and Mercado _et al._, run the following commands

`cat martin.tar.gz.* | tar xvzf -`
`cat mercado.tar.gz.* | tar xvzf -`

#### 1) Feature definition

For each structure, a set of features is defined. The pore geometry is characterized by eight structural parameters, whereas the chemical environment of the linker cores, linkages, and functional groups is represented with revised autocorrelation functions (RACs).

Zeo++ is used to calculate the structural parameters that define the pore geometry. Run the following commands for each structure in all databases to obtain the features:

`c3f.py convert struct.chk struct.cif`
`network -res struct.res -sa 1.84 1.84 3000 struct.sa -vol 1.84 1.84 3000 struct.vol struct.cif`
`cat struct.res struct.sa struct.vol > struct.geo`

The calculation of the RACs is implemented in the `racs.py` script. For each structure, a `struct.rac` file is generated specifying the number of identified linkages in the structure, as well as the RACs for each of the three chemical environments. Finally, by running `python collect_data.py`, the features of the pore geometry and the RACs for all domains are extracted and stored in the `features.csv` file. This file can be restored by running 

`cat features.csv.tar.gz.* | tar xvzf -`


#### 2) Subset selection

To extract a diverse subset from ReDD-COFFEE, the structures of our database are sorted using a maxmin algorithm. The feature vector of each material is defined by all features defined above. In each iteration of the maxmin algorithm, the structure that is furthest away from all points already selected is added to the subset. This procedure can be executed by running the `get_subset.py` script. The order of the structures is stored in the `sorted_maxmin.txt` file, where each line represents a structure.

The subset used in the paper contains 10 000 materials. These are the ones that are present in the first 10 000 lines o the `stored_maxmin.txt` file.

#### 3) Diversity metrics

The diversity metrics of each of the five COF databases for the four domains, _i.e._, the pore geometry and the chemical environments of the linker cores, linkages, and functional groups, are calculated using the `diversity_metrics.py` script. For each domain, only the respective feature set is adopted. Also the diversity metrics for the diverse subset of 10 000 structures are computed.  Besides the diversity metrics for the 10 000 structure subset, the metrics for a subset with a varying number of structures are calculated. The number of structures starts at 100 and goes up to 15 000 with steps of 100. The diversity metrics for each of these subsets of all domains are printed in the `vbd_subset_size.txt` file.

To calculate the diversity metrics, the structures of the material space are first clustered in 1 000 bins using a K-Means Clustering approach. The bin ids for all structures are defined in the `geometry.bin`, `linker.bin`, `linkage.bin`, and `functional.bin` files. The diversity metrics for all domains and all datasets are collected in the `diversity_metrics.dat` file. To plot these in a radar chart, run the `plot_vbd.py` script. The occupation of all bins, which defines the balance, is plotted using the `plot.py` script. Also the PCA and t-SNE plots provided in the electronic supplementary information is created using this script. The coordinates to create the figures in the manuscript and electronic supplementary information are stored in the `domain_pca.dat` and `domain_tsne.dat` files.

#### 4) Linkage count

Besides the RACs, also the linkages that are identified in each structure are stored in the `.rac` files. These files are read by the `collect_data.py` script, which collects all data in the `linkages.csv` file. To plot the a histogram of all groups of linkages for each database, the `linkage_analyse.py` script can be adopted. This creates the `database_linkages.pdf` figures, which are used to create Fig. 4 of the main manuscript.

### Step 3b: Textural properties

The textural properties are not only calculated for the COF databases, as is done above, but also for four databases of MOFs and zeolites. To find the structure files of these database, we kindly refer to the links provided in the electronic supplementary information. Again, the properties of the materials in these databases are calculated using Zeo++ by running the following commands.

`network -res struct.res -sa 1.84 1.84 3000 struct.sa -vol 1.84 1.84 3000 struct.vol struct.cif`
`cat struct.res struct.sa struct.vol > struct.geo`

The `collect_data.py` script reads all `.geo` files and stores the structural properties of both the COFs in the ReDD-COFFEE database and the MOFs and zeolites in the external database in the `structural.csv` file. By running the `plot.py` script, all figures of the paper are created. This are both 2D hexagonal histograms where the structures are divided according to their dimensionality (`own_XXX_XXX.pdf`), as 1D histograms where the structures are categorized by their linkage type (`hist_XXX.pdf`). Furthermore, also the 2D kernel density of all material classes (`mat_XXX_XXX.pdf`) is plotted.

### Step 3c: Adsorption properties

The GCMC calculations are performed with RASPA. To create the input files, a dedicated workflow is implemented in the `prepare_raspa.sh` script. The `input_files` folder should contain a folder for each structure in the diverse subset of 10 000 COFs, together with the optimized `.chk` file. Furthermore, the DREIDING parameters should be present in the `pars_noncov_dreiding.txt` file. These parameter files can be obtained by running the `run_dreiding.sh` job. The structure and parameter files of the methane molecule are provided in the `ch4.chk`, `pars_ch4_dreiding.txt`, and `pars_ch4_trappe_ua.txt` files.

Once all these input files are defined, the `prepare_raspa.sh` script calls the `prepare_raspa.py` script, which reads the structure and parameter files and writes all required RASPA input files as well as a job script. First, the optimized structure framework `.chk` file is converted to a `.cif` file and the pseudoatoms are defined in the `pseudo_atoms.def` file. Secondly, the host-guest interactions, which are defined by the DREIDING force field, are extracted from the `pars_noncov_dreiding.txt` and `pars_ch4_dreiding.txt` files and are specified in the `force_field_mixing_rules.def` file. The guest-guest interactions, which adopt the TraPPE-UA parameters that are given in the `pars_ch4_trappe_ua.txt` file, are written in the `force_field.def` file which overwrites the interactions defined in the mixing rules file. Finally, the `simulation.input` input file is created with detailed information on the simulation run.

Besides these input files, that depend on the studied structure, also the `ch4.def`, `run_gcmc.sh`, and `job_gcmc.sh` files are copied to each working directory. A job can simply be started by `qsub job_gcmc.sh` in each working directory. The output of the GCMC calculation is written to a `.data` file in the `Output/System_0` folder. By running `collect_data.py`, these output files can be read and the absolute adsorption and heat of adsorption of each structure are stored in the `CH4_298K_580000Pa.csv` and `CH4_298K_6500000Pa.csv` files for the low and high pressure, respectively. By running `plot.py`, all derived properties, such as the deliverable capacity, are calculated, and the figures of the manuscript are plotted. The structural properties of the structures in the subset, which are needed to make these figures, are provided in the `features_subset.csv` file.

## Benchmarks

### Benchmark 1: Validation of the periodic force fields

#### 1) PXRD patterns

In first instance, the periodic force field is validated by benchmarking its capacity to reproduce the experimental powder X-ray diffraction (PXRD) pattern. This procedure is described in the `Benchmarks/PeriodicFFValidation/PXRD_redd-coffee.ipynb` file, using a pyiron workflow (https://pyiron.org/). Similar to the previous notebook, the headers in this file delineate the sequential steps in this workflow. 

**input**
`system.chk`, `pars_cluster.txt`, `pars_uff.txt`, `material.dat`

**command line**
see notebook

**output**
`static/` , `static_bg/`, `dynamic/`, `dynamic_bg/`

Starting from the system definition, and the experimental diffraction pattern, the notebook facilitates static and dynamic force field simulations, after which the corresponding PXRD pattern is calculated. Afterward, the PXRD pattern is compared to the experimental one, and the heuristic values pertaining to their similarity/difference are calculated. Aside from the original experimental pattern, the notebook also attempts to remove the background noise, and repeats the heuristic analysis, usually leading to a better agreement.

#### 2) Single crystal structures

The force fields are further validated by checking their ability to reproduce single crystal X-ray diffraction (SCXRD) and 3D rotation electron diffraction (RED) structures. Therefore, the structures are reproduced with MD simulations. Initially, the experimental structures are optimized to obtain an initial structure for the simulations.

`python opt.py`

Subsequently, the MD simulations can be executed by running the following command:

`python md.py`

Finally, the internal coordinates and unit cell parameters are extracted from the resulting `COF_FF.h5` files, which contain the trajectory, and written to the `COF_FF_dyn.txt` files by the `analysis.py` post-processing script. These can be compared with the experimental data provided in the `COF_exp.txt` files.


### Benchmark 2: Force field arguments

To benchmark the adopted force field arguments, the real-space cutoff, reciprocal space cutoff, and scaling factor are varied from their default values to calculate the energy of nine optimized structures. The full routine can be executed by running:

`python benchmarking_pars.py`

### Benchmark 3: Zeo++ arguments

To select the number of Monte Carlo samples adopted to calculate the accessible surface area and volume, the Zeo++ calculations as described above are repeated for 21 optimized structures. The number of Monte Carlo samples is increased from 250 to 3 500 with steps of 250. This procedure is implemented in the `run.sh` script. The output of each calculation can be found in the `struct_optimized_N.sa` and `struct_optimized_N.vol` files. The analysis is performed by the `analysis.py` script.

### Benchmark 4: GCMC level of theory

Similar to the procedure outlined above, the required RASPA input files can be generated by running the `prepare_raspa.sh` script. Instead of adopting the DREIDING force field for host-guest interactions and the TraPPE-UA model for the guest-guest interactions, now a broader range of levels of theory are applied. The host-guest interactions are specified in the `pars_COF-X_LOT.txt` and the `pars_ch4_LOT.txt`, where `LOT` is the level of theory for the host-guest interactions. The guest-guest interactions are defined in the `pars_ch4_LOT.txt` files. If the model adopted for the host-guest interactions and the guest-guest interactions differs, the two `pars_ch4_LOT.txt` are not the same. The definition of the methane molecule is stored in the `ch4_all.chk` and `ch4_all.def` files. This structure contains both pseudoatoms at the atom positions (for UFF, MM3, and DREIDING), as well as in between the carbon and hydrogen atoms (for the TraPPE-EH model).

Once the working directories are initiated with the `prepare_raspa.sh` script, the jobs can be ran with `qsub job_gcmc.sh`. The isotherms and equilibration of all calculations are plotted with the `analysis.py` script. The experimental isotherms are provided in the `exp/COF-X_UptakeCH4.dat` files.

