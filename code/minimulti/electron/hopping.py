#!/usr/bin/env python
"""
hoppings: help to build hopping term from NN bonds and basis_set.
"""
from math import sqrt
import numpy as np
from minimulti.electron.myNN import bond
from minimulti.electron.basis2 import Basis, BasisSet


def hop_func(t, bonds, bset, label1, label2, spin, orientations):
    """
    define hopping from bonds and basis_set

    :param t: t is just t.
    :param bonds: the NN bonds.
    :param b_set: the basis set
    :param label1 & labeel2: the label of the basis. as in basis.label. can be tuple, string, etc
    :param orientations: the list of orientations the bonds are along. Note: Do remember the "-" orientations. eg. ['x','-x','y','-y']
    :param spin: the spin None| 'UP'|'DOWN'|'BOTH'. If 'BOTH', the hopping between same spin (both up and down)


    :returns: a list tuples of (t,basis1,basis2,offset), basis1 , basis2 are the id of the basis; offset is the "R" in the tb model. t is the hopping parameter.

    """
    hoppings = []
    for b in bonds:
        a1 = b.atom1
        a2 = b.atom2
        if spin != -1:
            basis1 = Basis(site=a1, label=label1, spin=spin, index=None)
            basis2 = Basis(site=a2, label=label2, spin=spin, index=None)
            id1 = bset.get_basis_id(basis1)
            id2 = bset.get_basis_id(basis2)
            offset = b.offset
            if b.orientation in orientations:
                hoppings.append(tuple((t, id1, id2, offset)))
        else:
            hoppings = hop_func(
                t, bonds, bset, label1, label2, 0, orientations) + hop_func(
                    t, bonds, bset, label1, label2, 1, orientations)
    return hoppings


def hop_func_s(t, bonds, bset, spin=None):
    """
    building s orbital hopping.

    :param t: t
    :param bonds: bonds
    :param bset: basis set
    :param spin: None|'UP'|'DOWN'|'BOTH'
    """
    return hop_func(t, bonds, bset, 's', 's', spin,
                    ['x', '-x', 'y', '-y', 'z', '-z'])


def hop_func_dxy(t, bonds, bset, spin=None):
    """
    the shortcut of building dxy hopping

    :param t: t
    :param bonds: bonds
    :param bset: basis set
    :param spin: None|'UP'|'DOWN'|'BOTH'
    """
    return hop_func(t, bonds, bset, 'dxy', 'dxy', spin, ['x', '-x', 'y', '-y'])


def hop_func_dyz(t, bonds, bset, spin=None):
    """
    the shortcut of building dyz hopping

    :param t: t
    :param bonds: bonds
    :param bset: basis set
    :param spin: None|'UP'|'DOWN'|'BOTH'
    """
    return hop_func(t, bonds, bset, 'dyz', 'dyz', spin, ['z', '-z', 'y', '-y'])


def hop_func_dxz(t, bonds, bset, spin=None):
    """
    the shortcut of building dxz hopping

    :param t: t
    :param bonds: bonds
    :param bset: basis set
    :param spin: None|'UP'|'DOWN'|'BOTH'
    """
    return hop_func(t, bonds, bset, 'dxz', 'dxz', spin, ['z', '-z', 'x', '-x'])


def hop_func_t2g(t, bonds, bset, spin=None):
    """
    the shortcut of building the t2g hopping

    :param t: t
    :param bonds: bonds
    :param bset: basis set
    :param spin: None|'UP'|'DOWN'|'BOTH'
    """
    return hop_func_dxy(
        t, bonds, bset, spin=spin) + hop_func_dyz(
            t, bonds, bset, spin=spin) + hop_func_dxz(
                t, bonds, bset, spin=spin)


def hop_func_eg(t, bonds, bset, spin=None):
    """
    the shortcut for building the eg hopping

    along x:
     .. math::
        \\begin{pmatrix}
        3t_0/4 & -\\sqrt{3}t_0/4 \\\\
        -\\sqrt{3}t_0/4 & t_0/4
        \\end{pmatrix}

    along y:
     .. math::
        \\begin{pmatrix}
        3t_0/4 & \\sqrt{3}t_0/4 \\\\
        \\sqrt{3}t_0/4 & t_0/4
        \\end{pmatrix}

    along z:
     .. math::
        \\begin{pmatrix}
        0 & 0 \\\\
        0 & t_0
        \\end{pmatrix}
    """
    # a, b are x2-y2 and 3z2-r2.
    # a-a along x
    h_aa_x = hop_func(3.0 * t / 4, bonds, bset, 'dx2', 'dx2', spin,
                      ['x', '-x'])
    h_ab_x = hop_func(-sqrt(3.0) * t / 4, bonds, bset, 'dx2', 'dz2', spin,
                      ['x', '-x'])
    h_ba_x = hop_func(-sqrt(3.0) * t / 4, bonds, bset, 'dz2', 'dx2', spin,
                      ['x', '-x'])
    h_bb_x = hop_func(t / 4.0, bonds, bset, 'dz2', 'dz2', spin, ['x', '-x'])

    h_aa_y = hop_func(3.0 * t / 4, bonds, bset, 'dx2', 'dx2', spin,
                      ['y', '-y'])
    h_ab_y = hop_func(
        sqrt(3.0) * t / 4.0, bonds, bset, 'dx2', 'dz2', spin, ['y', '-y'])
    h_ba_y = hop_func(
        sqrt(3.0) * t / 4.0, bonds, bset, 'dz2', 'dx2', spin, ['y', '-y'])
    h_bb_y = hop_func(t / 4.0, bonds, bset, 'dz2', 'dz2', spin, ['y', '-y'])

    h_aa_z = []
    h_ab_z = []
    h_ba_z = []
    h_bb_z = hop_func(t, bonds, bset, 'dz2', 'dz2', spin, ['z', '-z'])

    h = (h_aa_x + h_ab_x + h_ba_x + h_bb_x + h_aa_y + h_ab_y + h_ba_y + h_bb_y
         + h_aa_z + h_ab_z + h_ba_z + h_bb_z)

    return h


pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
