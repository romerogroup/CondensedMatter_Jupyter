import numpy as np
from minimulti.spin.spin_api import SpinModel
from minimulti.utils.supercell import SupercellMaker
import copy
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.units import fs, kB


class SpinLatticeModel():
    def __init__(self,  spin_model, lattice_model, spin_lattice_coupling, atoms=None):
        self._spin_model = spin_model
        self._lattice_model = lattice_model
        self._slc = spin_lattice_coupling

        self._spin_model.add_term(self._slc, name='slc')
        self._lattice_model.add_term(self._slc, name='slc')

        #self.atoms = copy.deepcopy(self._lattice_model.ref_atoms)
        if atoms is not None:
            self.atoms=atoms
        else:
            self.atoms = copy.deepcopy(self._lattice_model.ref_atoms)
        self.atoms.set_calculator(self._lattice_model)
        self._lattice_model.atoms = self.atoms

    @property
    def spin_model(self):
        return self._spin_model

    @property
    def lattice_model(self):
        return self._lattice_model

    @property
    def slc(self):
        return self._slc

    def set_atoms(self, atoms, copy=False):
        if copy:
            self.atoms = copy.deepcopy(atoms)
        else:
            self.atoms = atoms
        self.atoms.set_calculator(self._lattice_model)
        self._lattice_model.atoms = atoms

    def set_spin_params(self, **kwargs):
        self.spin_model.set(**kwargs)

    def set_lattice_params(self, md='NVE', **kwargs):
        #self.lattice_model.set(**kwargs)
        self.lattice_temperature = kwargs['lattice_temperature']
        self.lattice_time_step = kwargs['lattice_time_step']
        self.lattice_friction = kwargs['lattice_friction']
        if md == 'Langevin':
            self._lattice_dyn = Langevin(self.atoms, self.lattice_time_step,
                                         self.lattice_temperature,
                                         self.lattice_friction, trajectory='LattHist.traj')
        elif md == 'NVE':
            self._lattice_dyn = VelocityVerlet(
                self.atoms, dt=self.lattice_time_step, trajectory='LattHist.traj')



    def make_supercell(self, sc_matrix):
        smaker = SupercellMaker(sc_matrix)
        sc_spin_model = self._spin_model.make_supercell(supercell_maker=smaker)
        sc_lattice_model = self._lattice_model.make_supercell(
            supercell_maker=smaker)
        sc_slc = self._slc.make_supercell(
            supercell_maker=smaker,
            sc_spin_model=sc_spin_model,
            sc_lattice_model=sc_lattice_model)
        return SpinLatticeModel(sc_spin_model, sc_lattice_model, sc_slc)

    #@profile
    def run(self, nstep=1):
        self._slc.S = self._spin_model.S
        self._slc.displacement = self._lattice_model.dx
        for istep in range(nstep):
            self._lattice_dyn.run(1)
            self._spin_model.run_one_step()
            self._slc.displacement = self._lattice_model.dx
            self._slc.S = self._spin_model.S

            #print(self._lattice_model.dx)
