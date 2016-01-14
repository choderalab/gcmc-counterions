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

def compute_reduced_potential(context, nmolecules, temperature, pressure, chemical_potential):
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
    context : simtk.openmm.Context
    nmolecules : number of species corresponding to chemical potential \mu
    temperature : simtk.unit.Quantity compatible with kelvin
    pressure : simtk.unit.Quantity compatible with atmospheres
    chemical_potential : simtk.unit.Quantity compatible with kcal/mol

    """
    kT = kB * temperature
    beta = 1.0 / kT

    state = context.getState(getEnergy=True)
    potential = state.getPotentialEnergy()
    volume = state.getPeriodicBoxVolume()

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
    """
    import copy
    new_topology = copy.deepcopy(topology)
    # Find residues in topology.
    water_residues = [ residue for residue in new_topology.residues() if (residue.name == water_residue.name) ]
    cation_residues = [ residue for residue in new_topology.residues() if (residue.name == cation_residue.name) ]
    anion_residues = [ residue for residue in new_topology.residues() if (residue.name == antion_residue.name) ]
    # Select residues to transmute.
    if mode == 'add-salt':
        # Convert two water residues to cation and anion.
        replace_residues = random.choice(water_residues, size=2, replace=False)
        copy_residue(replace_residues[0], cation_residue)
        copy_residue(replace_residues[1], anion_residue)
    if mode == 'delete-salt':
        # Convert cation and anion to water residues.
        replace_residues = [ random.choice(cation_residues), random.choice(anion_residues) ]
        copy_residue(replace_residues[0], water_residue)
        copy_residue(replace_residues[1], water_residue)

    return new_topology

def modify_system(system, topology, context=None):
    """
    Modify system (and associated context) to reflect residues defined in Topology

    """
    pass

# Create a water box
from openmmtools.testsystems import WaterBox
waterbox = WaterBox()
[system, positions, topology] = [waterbox.system, waterbox.positions, waterbox.topology]

# Parameters
temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 5.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
chemical_potential = 0.0 * unit.kilocalories_per_mole # chemical potential
nsteps = 50 # number of timesteps per iteration
niterations = 50 # number of iterations
mctrials = 10 # number of Monte Carlo trials per iteration
nsalt = 0 # current number of salt pairs
tol = 1e-6 # constraint tolerance

# Determine number of molecules
nmolecules = 0
for residue in topology.residues():
    nmolecules += 1
print('system originally has %d water molecules' % nmolecules)

# Create a simulation
from openmmtools.integrators import VelocityVerletIntegrator
integrator = VelocityVerletIntegrator(timestep)
integrator.setConstraintTolerance(tol)
context = openmm.Context(system, integrator)
context.setPositions(positions)

water_residue = find_residue(topology, 'HOH')
cation_residue = find_residue(topology, 'Na+')
anion_residue = find_residue(topology, 'Cl-')

# Open PDB file for writing.
from simtk.openmm.app import PDBFile
pdbfile = open('output.pdb', 'w')
PDBFile.writeHeader(topology, file=pdbfile)
PDBFile.writeModel(topology, positions, file=pdbfile, modelIndex=0)

# Simulate
for iteration in range(niterations):
    print('iteration %5d / %5d' % (iteration, niterations))

    # Propagate dynamics at constant counterion number.
    print('propagating dynamics for %d steps...' % nsteps)
    context.setVelocitiesToTemperature(temperature)
    context.applyConstraints(tol)
    context.applyVelocityConstraints(tol)
    integrator.step(nsteps)

    # Update counterions.
    print('counterion Monte Carlo')
    naccepted = 0
    nrejected = 0
    for trial in range(mctrials):
        print('  mc trial %5d / %5d' % (trial, mctrials))
        # Compute initial reduced potential.
        u_initial = compute_reduced_potential(context, nsalt, temperature, pressure, chemical_potential)
        # Select whether we will add or delete salt pair.
        if (nsalt==0) or ((nsalt < nmolecules) and (np.random.random() < 0.5)):
            mode = 'add-salt'
        else:
            mode = 'delete-salt'
        # Propose the modified topology and modify the system parameters in the context.
        proposed_topology = propose_topology(topology, mode, water_residue, anion_residue, cation_residue)
        modify_system(context, proposed_topology, system)
        # Compute final reduce potential.
        u_final = compute_reduced_potential(context, nsalt, temperature, pressure, chemical_potential)
        # Accept or reject.
        if accept:
            naccepted += 1
            # Accept proposed topology.
            topology = proposed_topology
        else:
            nrejected += 1
            # Revert to old system parameters.
            modify_system(context, topology, system)

    # Write PDB frame with current topology and positions.
    PDBFile.writeModel(topology, positions, file=pdbfile, modelIndex=iteration+1)

pdbfile.close()
