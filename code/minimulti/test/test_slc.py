import unittest
import os
import numpy as np
from minimulti.spin_lattice_coupling import OijuTerm, TijuvTerm
from minimulti.utils.supercell import SupercellMaker


class SLCTest(unittest.TestCase):
    def setUp(self):
        self.Oiju=OijuTerm.read_netcdf(fname='./Oiju_scalarij.nc')
        self.natom=5
        self.nspin=1

    def test_make_supercell(self):
        scmaker=SupercellMaker(sc_matrix=np.eye(3)*2)
        self.sc_Oiju=self.Oiju.make_supercell(supercell_maker=scmaker)

    def test_bfield(self):
        S=np.zeros((self.nspin, 3), dtype=float)
        displacement=np.zeros((self.natom*3), dtype=float)
        bfield=np.zeros((self.natom,3), dtype=float)
        self.Oiju.eff_field(S, displacement, bfield)


    def test_force(self):
        S=np.array([[0,0,1.0]])
        Sref=np.array([[0,0,1.0]])
        forces=self.Oiju.get_forces(S,Sref)

class SLCSupercellTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.Oiju=OijuTerm.read_netcdf(fname='./Oiju_scalarij.nc')
        self.ncellx=4
        self.ncell=self.ncellx**3
        self.natom=5*self.ncell
        self.nspin=1*self.ncell
        self.scmaker=SupercellMaker(sc_matrix=np.eye(3)*self.ncellx)
        self.sc_Oiju=self.Oiju.make_supercell(supercell_maker=self.scmaker)


    def test_supercell_force(self):
        Sref=np.zeros((self.nspin, 3), dtype=float)
        Sref[:,2]=np.real(self.scmaker.phase([0.5,0.5,0.5]))
        S=np.zeros((self.nspin, 3), dtype=float)
        #S[:,2]=1.0
        #S=np.random.random((self.nspin, 3))
        S=Sref.copy()
        #S[5,1]=-1.0
        #S[5,2]=0.0
        S[1,2]*=-1
        S[2,2]*=-1
        forces=self.sc_Oiju.get_forces(S, Sref)
        #print("forces:", forces)
        print("sum of forces", np.sum(forces, axis=0))
        #print(sorted(forces.flatten()))


    def test_supercell_bfield(self):
        S=np.zeros((self.nspin, 3), dtype=float)
        displacement=np.zeros((self.natom*3), dtype=float)
        displacement[3]=0.01
        bfield=np.zeros((self.nspin,3), dtype=float)
        self.sc_Oiju.eff_field(S, displacement, bfield)
        #print(bfield)


unittest.main()
