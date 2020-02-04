#!/usr/bin/env python
"""
Model builder: A helper for building models.
include Perovskite t2g only and eg only model.
model from Wannier90 output.
"""
import numpy as np
from minimulti.electron.Hamiltonian import etb_model, atoms_model
from minimulti.electron.basis2 import BasisSet, gen_basis_set, atoms_to_basis
from ase.atoms import Atoms
import matplotlib.pyplot as plt
from minimulti.electron.myNN import myNeighborList
from minimulti.electron.hopping import hop_func_dxy, hop_func_t2g, hop_func_eg
from  ase.lattice.cubic import SimpleCubicFactory

t2g_orbs = ('dxy', 'dyz', 'dxz')
eg_orbs = ('dx2', 'dz2')


# Define cubic perovskite
class PerovskiteCubicFactory(SimpleCubicFactory):
    "A factory for creating perovskite (ABO3) lattices"

    xtal_name = 'cubic perovskite'
    bravais_basis = [
                     [0.,0.,0.],
                     [0.5,0.5,0.5],
                     [0.5,0.5,0.],[0.,0.5,0.5],[0.5,0.,0.5]]
    element_basis = (0,1,2,2,2)

PerovskiteCubic=PerovskiteCubicFactory()
PerovskiteCubic.__doc__="""
Usage:
eg. STO=PerovskiteCubic(['Sr','Ti','O'],latticeconstant=3.905)
"""


def perovskite_builder(
        A='Sr',
        B='Ti',
        O='O',
        lattice_constant=3.9,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        orbs='t2g',
        nspin=2,
        t=1.0,
        #delta=2.0,
        nel=0,
        Utype='Liechtenstein',
        Hubbard_dict={}, ):
    """
    helper class for transitional metal perovskite eg/ t2g only models
    """
    atoms = PerovskiteCubic(
        [A, B, O],
        latticeconstant=lattice_constant,
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    bdict={'t2g': t2g_orbs, 'eg':eg_orbs}
    model = atoms_model(atoms, basis_dict={B: bdict[orbs]}, nspin=nspin)
    mynb = myNeighborList(
        [lattice_constant / 1.4] * len(atoms),
        self_interaction=False,
        bothways=False)
    mynb.set_axis(x=directions[0], y=directions[1])
    mynb.build(atoms)
    bonds = mynb.get_all_bonds(from_species=[B], to_species=[B])
    if nspin == 2:
        spin = -1
    else:
        spin = 0
    if orbs=='t2g':
        hopping = hop_func_t2g(t, bonds, model.bset, spin=spin)
    elif orbs=='eg':
        hopping = hop_func_eg(t, bonds, model.bset, spin=spin)
    else:
        raise ValueError('only t2g | eg ')
    for h in hopping:
        t, i, j, ind_R = h
        model.set_hop(-t, i, j, ind_R=ind_R, allow_conjugate_pair=False)
    #if orbs=='t2g':
    #    model.set_onsite(np.array([1,-1,1,-1, 1,-1])*delta)
    #else:
    #    model.set_onsite(np.array([1,-1,1,-1])*delta)
    model.set_Hubbard_U(Utype=Utype, Hubbard_dict=Hubbard_dict)
    return model

