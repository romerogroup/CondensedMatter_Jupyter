from phonopy import Phonopy
from phonopy.file_IO import parse_FORCE_CONSTANTS, parse_disp_yaml, parse_FORCE_SETS
from phonopy.structure.atoms import PhonopyAtoms
from ase.atoms import Atoms
from ase.io import read
import numpy as np
from numpy.linalg import inv

from ase.dft.kpoints import *
import matplotlib.pyplot as plt

from .phonon_unfolder import phonon_unfolder
from .plotphon import plot_band_weight

def read_phonopy( sposcar='SPOSCAR', sc_mat=np.eye(3),force_constants=None,  disp_yaml=None, force_sets=None):
    if force_constants is None and (disp_yaml is None or force_sets is None):
        raise ValueError("Either FORCE_CONSTANTS or (disp.yaml&FORCE_SETS) file should be provided.")
    atoms=read(sposcar)
    #vesta_view(atoms)
    primitive_matrix=inv(sc_mat)
    bulk = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            scaled_positions=atoms.get_scaled_positions(),
            cell=atoms.get_cell())
    phonon = Phonopy(
        bulk,
        supercell_matrix=np.eye(3),
        primitive_matrix=primitive_matrix,
        #factor=factor,
        #symprec=symprec
    )

    if disp_yaml is not None:
        disp=parse_disp_yaml(filename=disp_yaml)
        phonon.set_displacement_dataset(disp)
    if force_sets is not None:
        fc=parse_FORCE_SETS(filename=force_sets)
        phonon.set_forces(fc)
    
    fc=parse_FORCE_CONSTANTS(force_constants)
    phonon.set_force_constants(fc)

    return phonon

def unf(phonon, sc_mat, qpoints, knames=None, x=None, xpts=None):
    prim=phonon.get_primitive()
    prim=Atoms(symbols=prim.get_chemical_symbols(), cell=prim.get_cell(), positions=prim.get_positions())
    #vesta_view(prim)
    sc_qpoints=np.array([np.dot(q, sc_mat) for q in qpoints])
    phonon.set_qpoints_phonon(sc_qpoints, is_eigenvectors=True)
    freqs, eigvecs = phonon.get_qpoints_phonon()
    uf=phonon_unfolder(atoms=prim, supercell_matrix=sc_mat, eigenvectors=eigvecs, qpoints=sc_qpoints, phase=False)
    weights = uf.get_weights()

    #ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*8065.6,weights[:,:].T*0.98+0.01,xticks=[knames,xpts],style='alpha')
    ax=plot_band_weight([list(x)]*freqs.shape[1],freqs.T*33.356,weights[:,:].T*0.99+0.001,xticks=[knames,xpts],style='alpha')
    return ax

def phonopy_unfold(sc_mat=np.diag([1,1,1]), unfold_sc_mat=np.diag([3,3,3]),force_constants='FORCE_CONSTANTS', sposcar='SPOSCAR', qpts=None, qnames=None, xqpts=None, Xqpts=None):
    phonon=read_phonopy(sc_mat=sc_mat, force_constants=force_constants, sposcar=sposcar)
    ax=unf(phonon, sc_mat=unfold_sc_mat, qpoints=qpts, knames=qnames,x=xqpts, xpts=Xqpts )
    return ax




from ase.build import bulk
def kpath():
    atoms = bulk('Cu', 'fcc', a=3.61)
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    GXW = [points[k] for k in 'GXWGL']
    kpts, x, X = bandpath(GXW, atoms.cell, 300)
    names = ['$\Gamma$', 'X', 'W', '$\Gamma$', 'L']
    return kpts, x, X, names


def test():
    phonon=read_phonopy(sc_mat=np.diag([1,1,1]), force_constants='FORCE_CONSTANTS', sposcar='SPOSCAR')
    qpts, x, X, names=kpath()
    ax=unf(phonon, sc_mat=np.diag([3,3,3]), qpoints=qpts, knames=names,x=x, xpts=X )
    plt.show()


if __name__=="__main__":
    test()
