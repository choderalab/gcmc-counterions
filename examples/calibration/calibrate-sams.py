#!/usr/bin/env python
"""
Calibrate the salt chemical potential $\mu_\mathrm{salt}$ as a function of macroscopic salt concentration c.

In this code, a self-adjusted mixture sampling (SAMS) simulation is run in which the number of salt pairs is allowed to dynamically vary.
The free energy as a function of instantaneous salt concentration is computed.

"""

import numpy as np
from numpy import random
from simtk import openmm, unit
from simtk.openmm import app

# CONSTANTS
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

from openmmtools.testsystems import TestSystem
class WaterBox(TestSystem):

    """
    Create a water box test system.
    Examples
    --------
    Create a default (TIP3P) waterbox.
    >>> waterbox = WaterBox()
    Control the cutoff.
    >>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)
    Use a different water model.
    >>> waterbox = WaterBox(model='tip4pew')
    Don't use constraints.
    >>> waterbox = WaterBox(constrained=False)
    """

    def __init__(self, box_edge=25.0*unit.angstroms, cutoff=9*unit.angstroms, model='tip3p', switch_width=1.5*unit.angstroms, constrained=True, dispersion_correction=True, nonbondedMethod=app.PME, ewaldErrorTolerance=5E-4, **kwargs):
        """
        Create a water box test system.
        Parameters
        ----------
        box_edge : simtk.unit.Quantity with units compatible with nanometers, optional, default = 2.5 nm
           Edge length for cubic box [should be greater than 2*cutoff]
        cutoff : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.9 nm
           Nonbonded cutoff
        model : str, optional, default = 'tip3p'
           The name of the water model to use ['tip3p', 'tip4p', 'tip4pew', 'tip5p', 'spce']
        switch_width : simtk.unit.Quantity with units compatible with nanometers, optional, default = 0.5 A
           Sets the width of the switch function for Lennard-Jones.
        constrained : bool, optional, default=True
           Sets whether water geometry should be constrained (rigid water implemented via SETTLE) or flexible.
        dispersion_correction : bool, optional, default=True
           Sets whether the long-range dispersion correction should be used.
        nonbondedMethod : simtk.openmm.app nonbonded method, optional, default=app.PME
           Sets the nonbonded method to use for the water box (one of app.CutoffPeriodic, app.Ewald, app.PME).
        ewaldErrorTolerance : float, optional, default=5E-4
           The Ewald or PME tolerance.  Used only if nonbondedMethod is Ewald or PME.
        Examples
        --------
        Create a default waterbox.
        >>> waterbox = WaterBox()
        >>> [system, positions] = [waterbox.system, waterbox.positions]
        Use reaction-field electrostatics instead.
        >>> waterbox = WaterBox(nonbondedMethod=app.CutoffPeriodic)
        Control the cutoff.
        >>> waterbox = WaterBox(box_edge=3.0*unit.nanometers, cutoff=1.0*unit.nanometers)
        Use a different water model.
        >>> waterbox = WaterBox(model='spce')
        Use a five-site water model.
        >>> waterbox = WaterBox(model='tip5p')
        Turn off the switch function.
        >>> waterbox = WaterBox(switch_width=None)
        Set the switch width.
        >>> waterbox = WaterBox(switch_width=0.8*unit.angstroms)
        Turn of long-range dispersion correction.
        >>> waterbox = WaterBox(dispersion_correction=False)
        """

        TestSystem.__init__(self, **kwargs)

        supported_models = ['tip3p', 'tip4pew', 'tip5p', 'spce']
        if model not in supported_models:
            raise Exception("Specified water model '%s' is not in list of supported models: %s" % (model, str(supported_models)))

        # Load forcefield for solvent model.
        ff = app.ForceField('gcmc.xml')

        # Create empty topology and coordinates.
        top = app.Topology()
        pos = unit.Quantity((), unit.angstroms)

        # Create new Modeller instance.
        m = app.Modeller(top, pos)

        # Add solvent to specified box dimensions.
        boxSize = unit.Quantity(np.ones([3]) * box_edge / box_edge.unit, box_edge.unit)
        m.addSolvent(ff, boxSize=boxSize, model=model, neutralize=False, positiveIon='Na+', negativeIon='Cl-', **kwargs)

        # Get new topology and coordinates.
        newtop = m.getTopology()
        newpos = m.getPositions()

        # Convert positions to np.
        positions = unit.Quantity(np.array(newpos / newpos.unit), newpos.unit)

        # Create OpenMM System.
        self.forcefieldOptions = { 'nonbondedMethod' : nonbondedMethod, 'nonbondedCutoff' : cutoff, 'constraints' : None, 'rigidWater' : constrained, 'removeCMMotion' : False }
        system = ff.createSystem(newtop, **(self.forcefieldOptions))

        # Set switching function and dispersion correction.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}

        forces['NonbondedForce'].setUseSwitchingFunction(False)
        if switch_width is not None:
            forces['NonbondedForce'].setUseSwitchingFunction(True)
            forces['NonbondedForce'].setSwitchingDistance(cutoff - switch_width)

        forces['NonbondedForce'].setUseDispersionCorrection(dispersion_correction)
        forces['NonbondedForce'].setEwaldErrorTolerance(ewaldErrorTolerance)

        self.ndof = 3 * system.getNumParticles() - 3 * constrained

        self.topology = newtop
        self.system = system
        self.positions = positions
        self.forcefield = ff

def compute_reduced_potential(system, positions, nmolecules, temperature, pressure, chemical_potential):
    """
    Compute the current reduced potential:

      u(x) = \beta [ U(x) + p V(x) + \mu N(x) ]

    \beta  : inverse temperature
    U(x)   : potential energy
    p      : pressure
    V(x)   : instantaneous box volume
    \mu    : chemical potential
    N(x)   : number of molecules for chemical potential

    Parameters:
    -----------
    system : simtk.openmm.System
    positions : simtk.unit.Quantity
    nmolecules : number of species corresponding to chemical potential \mu
    temperature : simtk.unit.Quantity compatible with kelvin
    pressure : simtk.unit.Quantity compatible with atmospheres
    chemical_potential : simtk.unit.Quantity compatible with kcal/mol

    """
    kT = kB * temperature
    beta = 1.0 / kT

    integrator = openmm.VerletIntegrator(1.0*unit.femtosecond)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)

    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    volume = state.getPeriodicBoxVolume()

    del context, integrator

    reduced_potential = beta * (potential + pressure*volume*unit.AVOGADRO_CONSTANT_NA + chemical_potential*nmolecules)

    return reduced_potential

def identifyWaterResidues(topology, water_residue_names=('WAT', 'HOH', 'TP4', 'TP5', 'T4E')):
    """
    Compile a list of water residues that could be converted to/from monovalent ions.

    Parameters
    ----------
    topology : simtk.openmm.app.topology
        The topology from which water residues are to be identified.
    water_residue_names : list of str
        Residues identified as water molecules.

    Returns
    -------
    water_residues : list of simtk.openmm.app.Residue
        Water residues.

    """
    water_residues = list()
    for residue in topology.residues():
        if residue.name in water_residue_names:
            water_residues.append(residue)

    print('identifyWaterResidues: %d water molecules identified.' % len(water_residues))
    return water_residues

def find_residue(topology, residue_name):
    """

    """
    import copy
    for residue in topology.residues():
        if residue.name == residue_name:
            return copy.deepcopy(residue)
    raise Exception("No residue with name '%s' was found." % residue_name)

def propose_topology(topology, mode, water_residue, anion_residue, cation_residue):
    """
    Propose a modified topology according to the specified mode.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        The topology object to be modified
    mode : str
        Select whether to add ('add-salt') or delete ('delete-salt') a salt pair.
        If 'add-salt' is selected, a random pair of water molecules are converted into a salt pair.
        If 'delete-salt' is selected, a random counterion pair are converted into water molecules.

    Returns
    -------
    new_topology : simtk.openmm.app.Topology
        The updated topology object.
    logP_proposal : float
        log( P(old|new) / P(new|old) ) for Metropolis-Hastings
    """
    import copy
    new_topology = copy.deepcopy(topology)

    # Find residues in topology.
    water_residues = [ residue for residue in new_topology.residues() if (residue.name == water_residue.name) ]
    cation_residues = [ residue for residue in new_topology.residues() if (residue.name == cation_residue.name) ]
    anion_residues = [ residue for residue in new_topology.residues() if (residue.name == anion_residue.name) ]

    # Count residues in each class.
    nwater = len(water_residues)
    ncation = len(cation_residues)
    nanion = len(anion_residues)

    # Copy residue function.
    def copy_residue(dest, src):
        dest.name = src.name
        for (destatom, srcatom) in zip(dest.atoms(), src.atoms()):
            destatom.name = srcatom.name
            destatom.element = srcatom.element

    # Select residues to transmute.
    if mode == 'add-salt':
        # Convert two water residues to cation and anion.
        replace_residues = random.choice(water_residues, size=2, replace=False)
        copy_residue(replace_residues[0], cation_residue)
        copy_residue(replace_residues[1], anion_residue)
        logP_proposal = np.log( ((1.0/(ncation+1))*(1.0/(nanion+1))) / ((1.0/nwater)*(1.0/(nwater-1))) )
    if mode == 'delete-salt':
        # Convert cation and anion to water residues.
        replace_residues = [ random.choice(cation_residues), random.choice(anion_residues) ]
        copy_residue(replace_residues[0], water_residue)
        copy_residue(replace_residues[1], water_residue)
        logP_proposal = np.log( ((1.0/nwater)*(1.0/(nwater-1))) / ((1.0/(ncation-1))*(1.0/(nanion-1))) )

    return [new_topology, logP_proposal]

# Create residue templates.
from simtk.openmm.app import element
from simtk.openmm.app.topology import Topology
topology = Topology()
chain = topology.addChain()
water_residue = topology.addResidue('HOH', chain)
a1 = topology.addAtom('O', element.oxygen, water_residue)
a2 = topology.addAtom('H1', element.hydrogen, water_residue)
a3 = topology.addAtom('H2', element.hydrogen, water_residue)
topology.addBond(a1, a2)
topology.addBond(a1, a3)
cation_residue = topology.addResidue('NA', chain)
a1 = topology.addAtom('Na', element.sodium, cation_residue)
a2 = topology.addAtom('Du', element.hydrogen, cation_residue)
a3 = topology.addAtom('Du', element.hydrogen, cation_residue)
topology.addBond(a1, a2)
topology.addBond(a1, a3)
anion_residue = topology.addResidue('CL', chain)
a1 = topology.addAtom('Cl', element.chlorine, anion_residue)
a2 = topology.addAtom('Du', element.hydrogen, anion_residue)
a3 = topology.addAtom('Du', element.hydrogen, anion_residue)
topology.addBond(a1, a2)
topology.addBond(a1, a3)
reference_topology = topology

# Create a water box
#from openmmtools.testsystems import WaterBox
waterbox = WaterBox()
[system, positions, topology] = [waterbox.system, waterbox.positions, waterbox.topology]
forcefield = waterbox.forcefield # get ForceField object
forcefield_options = waterbox.forcefieldOptions

# Parameters
temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 5.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
chemical_potential = 10.0 * unit.kilocalories_per_mole # chemical potential
nsteps = 50 # number of timesteps per iteration
niterations = 100 # number of iterations
mctrials = 10 # number of Monte Carlo trials per iteration
nsalt = 0 # current number of salt pairs
tol = 1e-6 # constraint tolerance

# Determine number of molecules
nmolecules = 0
for residue in topology.residues():
    nmolecules += 1
print('system originally has %d water molecules' % nmolecules)

# Open PDB file for writing.
from simtk.openmm.app import PDBFile
pdbfile = open('output.pdb', 'w')
PDBFile.writeHeader(topology, file=pdbfile)
PDBFile.writeModel(topology, positions, file=pdbfile, modelIndex=0)

# Simulate
for iteration in range(niterations):
    print('iteration %5d / %5d' % (iteration, niterations))

    # Create a simulation
    from openmmtools.integrators import VelocityVerletIntegrator
    integrator = VelocityVerletIntegrator(timestep)
    integrator.setConstraintTolerance(tol)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)

    # Propagate dynamics at constant counterion number.
    print('propagating dynamics for %d steps...' % nsteps)
    context.setVelocitiesToTemperature(temperature)
    context.applyConstraints(tol)
    context.applyVelocityConstraints(tol)
    integrator.step(nsteps)

    # Get positions and clean up.
    positions = context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True)
    del context, integrator

    # Update counterions.
    print('counterion Monte Carlo')
    naccepted = 0
    nrejected = 0
    for trial in range(mctrials):
        print('  mc trial %5d / %5d' % (trial, mctrials))
        # Compute initial reduced potential.
        u_initial = compute_reduced_potential(system, positions, nsalt, temperature, pressure, chemical_potential)
        # Select whether we will add or delete salt pair.
        if (nsalt==0) or ((nsalt < nmolecules) and (np.random.random() < 0.5)):
            mode = 'add-salt'
            nsalt_proposed = nsalt + 1
        else:
            mode = 'delete-salt'
            nsalt_proposed = nsalt - 1

        # Propose the modified topology and modify the system parameters in the context.
        [proposed_topology, logP_proposal] = propose_topology(topology, mode, water_residue, anion_residue, cation_residue)
        proposed_system = forcefield.createSystem(proposed_topology, **forcefield_options)
        # Compute final reduce potential.
        u_final = compute_reduced_potential(proposed_system, positions, nsalt_proposed, temperature, pressure, chemical_potential)
        # Accept or reject.
        accept = False
        logP_accept = - (u_final - u_initial) + logP_proposal
        if (logP_accept > 0) or (np.random.random() < np.exp(logP_accept)):
            accept = True
        if accept:
            naccepted += 1
            # Accept proposed topology.
            topology = proposed_topology
            system = proposed_system
        else:
            nrejected += 1

    # Write PDB frame with current topology and positions.
    PDBFile.writeModel(topology, positions, file=pdbfile, modelIndex=iteration+1)

pdbfile.close()
