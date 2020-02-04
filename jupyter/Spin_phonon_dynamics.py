
# coding: utf-8

# ## Spin Lattice Coupling
# 

# ### Model building
# To build the coupled-spin-phonon model, we need first standalone spin/phonon models.

# In[1]:


import numpy as np
from ase.io import read
from minimulti.ioput.ifc_parser import IFCParser
from minimulti.spin_lattice_coupling.fitting import fit_cubic_T
from minimulti.spin_lattice_coupling.slc import Order2Term

from minimulti.spin_lattice_coupling.slc import SpinLatticeCoupling

def fit_Tijuv():
    mags = ['FM', 'A', 'G', 'C']
    fname = lambda mag: 'data/%s_ifc.txt' % mag

    ref_atoms = read('./data/POSCAR')
    ifc = {}
    for mag in mags:
        parser = IFCParser(atoms=ref_atoms, fname=fname(mag))
        ifc[mag] = parser.get_total_ifc()

    ifc0, J2, J2dict = fit_cubic_T(ifc['FM'], ifc['G'], ifc['A'], ifc['C'])
    term = Order2Term(natom=5, ms=[1], parameter=J2dict)
    return term


def build_splatt_term():
    splatt_term=SpinLatticeCoupling(ms=[1], natom=5)
    Tijuv_term=fit_Tijuv()
    splatt_term.add_term(Tijuv_term,name='Tijuv')
    return splatt_term


# In[ ]:


from ase.io import read
import numpy as np
from minimulti.lattice.lattice import Lattice
from minimulti.spin.spin_api import SpinModel
from minimulti.spin_lattice_coupling.slc import SpinLatticeCoupling
from minimulti.spin_lattice_model.spin_lattice_model import SpinLatticeModel
from ase.io import write
from ase.units import kB, fs

def run_spin_lattice_dynamics(supercell_matrix=np.eye(3)*4):
    # phonon model
    ifcfile = 'data/FM_ifc.txt'
    atoms = read('data/POSCAR')   # reference atomic structure
    lattice_model = Lattice(ref_atoms=atoms)   # initialize Lattice model
    lattice_model.read_ifc_file(ifcfile)
    # spin model
    spin_model=SpinModel(fname='data/exchange_FM.xml')
    
    # spin lattice coupling term
    sp_latt_term=build_splatt_term()
    
    # put them into coupled model
    sp_latt_model= SpinLatticeModel(lattice_model=lattice_model, spin_model=spin_model, spin_lattice_coupling=sp_latt_term)
    
    # build supercell
    sp_latt_model=sp_latt_model.make_supercell(sc_matrix=supercell_matrix)
    
    # set parameters
    sp_latt_model.set_spin_params(temperature=300, time_step=1e-4, total_time=1)
    sp_latt_model.set_lattice_params(lattice_temperature=300*kB, lattice_time_step= 5*fs, lattice_friction=0.002)
    for step in range(1000):
        sp_latt_model.run(3)
        write('splatt_traj/atom%03d.xyz'%step, sp_latt_model.atoms)
    print("Finished")

run_spin_lattice_dynamics()

