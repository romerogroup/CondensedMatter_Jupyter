#!/usr/bin/env python
from pytbu.wannier90 import w90
import matplotlib.pyplot as plt
from pyDFTutils.ase_utils.kpoints import cubic_kpath
import numpy as np


def test_read_w90():
    w90reader=w90(path='data', prefix='wannier90')
    tb=w90reader.model(min_hopping_norm=0.05)
    tb.set(nel=100, sigma=0.05)
    tb.set_kmesh([6,6,6])

    print('=============eigenvalues')
    print(tb.get_eigenvalues())
    print('=============Occupations')
    print(tb.get_occupations())
    print('=============Fermi level')
    print(tb.get_fermi_level())
    print('=============Orbital occupations')
    print(tb.get_orbital_occupations()/2)

    e, dos=tb.get_dos()
    plt.plot(e, dos)
    plt.axvline(0)
    plt.show()
    return tb

def test_band():
    w90reader=w90(path='data', prefix='wannier90')
    tb=w90reader.model(min_hopping_norm=0.001)
    tb.set(nel=36, sigma=0.05)

    scmat=[[1,-1,0],[1,1,0],[0,0,2]]
    #kpts, xs, xspecial,special_path=cubic_kpath(npoints=100)
    #kpts=[np.dot(kpt, scmat) for kpt in kpts]
    #tb.set_kpoints(kpts)
    #evals= tb.get_all_eigenvalues()
    # evals(iband, ikpt)
    #for i in xrange(len(evals)):
    #    plt.plot(xs, evals[i,:])
    ax=tb.plot_projection_band(color_dict={14:'red'}, kvectors=([0,0,0], [.5,0,0], [.5,.5,0],[.5,.5,.5] ), knames=['G','X', 'M', 'R'], supercell_matrix=scmat)
    #ax=tb.plot_projection_band(orb=3, color='red', xs=xs, axis=ax)
    #ax.axhline(5.2722, linestyle='--', color='black')
    plt.ylabel('Energy (eV)')
    plt.savefig('Ni1_eg.png')
    plt.savefig('Ni1_eg.pdf')
    plt.show() 

#test_read_w90()
test_band()


