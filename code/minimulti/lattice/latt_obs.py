from minimulti.constants import Boltzmann
import numpy as np
from ase.units import second, kg, kB

Boltzmann=kB
m0 = 1.660539040e-27
m0A2s = kg * 1e-20 / second**2


def get_temperature(masses, velocities):
    return 2.0 / 3 / Boltzmann * 0.5 * np.average(
        masses * np.sum(velocities**2, axis=1)) #* m0A2s


def get_atoms_temperature(atoms):
    return 2.0 / 3 / Boltzmann * 0.5 * np.average(
        np.sum(atoms.get_momenta()**2, axis=1) / atoms.get_masses()) #* m0A2s
