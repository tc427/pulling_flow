import os.path
import sys

import numpy as np

import mdtraj as md

import openmm.app as app
import openmm as omm
import openmm.unit as u

class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._init = False
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        state = simulation.context.getState(getEnergy=True, getForces=True, groups={6})
        if not self._init:
            self._out.write(f"{f.getName()}({u.kilojoules/u.mole})\tFtot({u.piconewton})\n")  
            self._init = True
            # print(f.getName(), state.getPotentialEnergy())
        # sum of the absolute values of the force
        # N.B.: we need N_A here because we have a single molecule, and forces are normally defined per mole
        Ftot = abs(state.getForces(asNumpy=True)).sum()/u.AVOGADRO_CONSTANT_NA
        self._out.write(f"{state.getPotentialEnergy().value_in_unit(u.kilojoules/u.mole)}\t{Ftot.value_in_unit(u.piconewton)}\n")
        self._out.flush()
        # forces = state.getForces().value_in_unit(u.kilojoules/u.mole/u.nanometer)
        # for f in forces:
            # self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))

pdb = md.load('villin_start.pdb')
topology = pdb.topology.to_openmm()

# pdb = PDBFile('output.pdb')
# modeller = Modeller(pdb.topology, pdb.positions)

base_name = "villin_linpos"
dat_file = "villin_linpos.h5"
traj_xtc_file = "villin_linpos.xtc"

# Create the system
forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
system = forcefield.createSystem(topology, nonbondedMethod=app.CutoffNonPeriodic)

# Create pulling potential
external = omm.CustomExternalForce('kx*(x-xz)^2')
system.addForce(external)
external.addPerParticleParameter('kx')
external.addPerParticleParameter('xz')

# Add pulling force to both ends
k_pulling = 10*u.kilojoules_per_mole/u.nanometer**2
# print(f"Pulling force: {(f_pulling/u.AVOGADRO_CONSTANT_NA).in_units_of(u.piconewton)}")
# external.addParticle(0, (-10, 0, 0)*u.kilojoules_per_mole/u.nanometer)
# external.addParticle(1012, (10, 0, 0)*u.kilojoules_per_mole/u.nanometer)
external.addParticle(0, [k_pulling, pdb.xyz[0][0][0]])
external.addParticle(1012, [k_pulling, pdb.xyz[0][1012][0]])

# Create force groups for reporting
for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

# Initialise the simulation
timestep = 2.0*u.femtoseconds
integrator = omm.LangevinIntegrator(330*u.kelvin, 1.0/u.picoseconds, timestep)
simulation = app.Simulation(topology, system, integrator)

# Initialise the particles positions and velocities 
simulation.context.setPositions(pdb.xyz[0])
simulation.context.setVelocitiesToTemperature(330*u.kelvin)

# Choose simulation length, then run
steps = 500_000_000
report_interval = 5000
N_split = 1000
sim_length = (steps*timestep).in_units_of(u.nanosecond)
print(f"{steps = }")
print(f"{sim_length = }")

if not os.path.exists(traj_xtc_file):
    simulation.reporters.append(md.reporters.HDF5Reporter(dat_file, report_interval, velocities=True))
    simulation.reporters.append(md.reporters.XTCReporter(traj_xtc_file, report_interval))
    # simulation.reporters.append(app.StateDataReporter(sys.stdout, report_interval,
    simulation.reporters.append(app.StateDataReporter(f"{base_name}_log.txt", report_interval,
                                                      step=True, time=True, potentialEnergy=True,
                                                      totalEnergy=True, temperature=True))
    simulation.reporters.append(ForceReporter(f"{base_name}_forces.txt", report_interval))

    for N_i in range(N_split):
        simulation.step(steps//N_split)
        x_pulling = np.array([-2* N_i, 0, 0])
        dx = (N_i / N_split)*30
        # print(dx, pdb.xyz[0][0][0]-dx, pdb.xyz[0][1012][0]+dx)
        external.setParticleParameters(0, 0, [k_pulling, pdb.xyz[0][0][0]-dx])
        external.setParticleParameters(0, 1012, [k_pulling, pdb.xyz[0][1012][0]+dx])
        external.updateParametersInContext(simulation.context)
