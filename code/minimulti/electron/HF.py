# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Hartree Fock Solver
"""
import itertools
from scipy.linalg import eigh
from math import exp, log
import sys
import numpy as np
from scipy.optimize import brentq
from minimulti.electron.density import fermi, density_matrix
from minimulti.electron.basis2 import gen_basis_set

class HartreeFock(object):
    def __init__(self,
                 O=None,
                 ham0=None,
                 Upqrs=dict(),
                 nbasis=0,
                 nelectron=None,
                 width=0.2,
                 nspin=1):
        """
        :param O: overlap matrix
        :param ham0: single body hamiltonian
        :param Upqrs: the many body part of the hamiltonian. sould be a dict or a function.
        :param nbasis: number of basis,
        :param N: number of elelctrons,default 0
        :param use_sparse: True| False. whether to use sparse matrix
        :param nspin: number of spin. default 1, currently spin=2 and spinor=False is not implemented. So please use spinor mode.
        :param width: fermi function width as in  E/width
        :param spinor: spinor mode True|False
        :param restricted: Restricted Hartree Fock or Unrestricted. Restricted should be used with nspin=1, unrestricted should be used with nspin=2 and spinor=True.
        """
        # Overlap matrix
        if O is not None:
            self.O = O
        else:
            self.O = np.eye(nbasis)
        # single body Hamiltonian
        if ham0 is not None:
            self.ham0 = ham0
        else:
            self.ham0 = np.zeros((nbasis, nbasis), dtype=float)
        # Interaction U_pqrs, should be a dict {(p,q,r,s):Upqrs}
        self.nbasis = nbasis
        self.set_Upqrs(Upqrs)
        self.N = nelectron
        self.nspin = nspin
        self.width = width
        self.rho = None

    def set_ham0(self, ham0):
        """
        set ham0,should be a numpy 2-d array or scipy sparse matrix
        """
        self.ham0 = ham0

    def set_Upqrs(self, Upqrs, thr=1e-3):
        """
        set Upqrs, a dict with keys: (p,q,r,s), or a function F(p,q,r,s) or np.array
        if a function, a dict will be generated from the function.
        thr: if \| Upqrs \| <thr, it will be omitted.

        Note : why not use coo sparse matrix: because it is not quadridic matrix.
        """
        if isinstance(Upqrs, dict):
            self.Upqrs = Upqrs
        elif isinstance(Upqrs, np.ndarray):
            self.Upqrs = dict()
            indexes = itertools.product(*([range(self.nbasis)] * 4))
            for i in indexes:
                val = Upqrs[tuple(i)]
                if abs(val) > thr:
                    self.Upqrs[tuple(i)] = val
        else:
            self.Upqrs = dict()
            indexes = itertools.product(*([range(self.nbasis)] * 4))
            for i in indexes:
                val = Upqrs(*i)
                if abs(val) > thr:
                    self.Upqrs[tuple(i)] = val

    def set_rho(self, rho):
        """
        set rho.

        :param rho: the density matrix.
        """
        self.rho = rho

    def U_func(self, p, q, r, s):
        """
        the function U(p,q,r,s)
        """
        if (p, q, r, s) in self.Upqrs:
            return self.Upqrs[(p, q, r, s)]
        else:
            return 0.0

    def guess_rho(self):
        """
        guess: current implementation: rho. set to zero, which means the first iteration do not have e-e interaction.
        """
        self.rho = np.zeros((self.nbasis, self.nbasis))

    def eff_ham(self, rho, restricted=True):
        """
        Now only restricted HF is implemented,TODO: add UHF
        generate effective hamiltonian
        for restricted hf:
        Heff=ham0+Sum_{r,s}[(2*Urpsq-Urpqs)*rho_{rs}]
        for unrestricted HF: TODO

        :param rho: a n*n matrix, rho(r,s)=sum(a,<r,a|fermifunc(a))|s,a>.
        :param restricted: True| False. restricted or none restricted.
        """
        self.ham_eff = self.ham0 + HF_U(
            self.Upqrs,
            self.nbasis,
            self.rho,
            restricted=self.restricted,
            spin_indexed=False,
            density_only=False)
        return self.ham_eff

    def fermi_energy(self, xtol=1e-9):
        """
        get the fermi energy.
        """

        def func(mu):
            return (fermi(self.eigenvals, mu, self.width)).sum() * 2 / (
                self.nspin) - self.N

        try:
            # make the first guess (simple guess, assuming equal k-point weights)

            ifermi = int(self.N / 2 * self.nspin)
            elo = self.eigenvals[ifermi - 1]
            ehi = self.eigenvals[ifermi + 1]
            guess = self.eigenvals(ifermi)
            dmu = np.max((self.width, guess - elo, ehi - guess))
            mu = brentq(func, guess - dmu, guess + dmu, xtol=xtol)
        except:
            # probably a bad guess
            dmu = self.width
            mu = brentq(
                func,
                self.eigenvals[0] - dmu,
                self.eigenvals[-1] + dmu,
                xtol=xtol)
        if np.abs(func(mu)) > 1e-6:
            #print(func(mu))
            raise RuntimeError(
                'Fermi level could not be assigned reliably. Has the system fragmented?'
            )
        self.efermi = mu

    def HF_solve_all(self, max_step=100, e_tol=0.00001):
        """
        The whole process of solving the hartree fock promblem.

        :param max_step: (int, default 100) the max step to iteratively solve the problem
        :param e_tol: (int, default 1e-5) energy tolerance.
        """
        #1. initialize
        self.guess_rho()
        #2. iterative solve
        last_E = np.inf
        for i in range(max_step):
            self.eff_ham(self.rho)
            self.solve_Ham_eff()
            self.fermi_energy()
            self.calc_rho()
            #print("rho: %s\n"%self.rho)
            deltaE = self.E - last_E
            last_E = self.E
            print("Iter %s: E=%s, delta_E=%s" % (i, self.E, deltaE))
            if abs(deltaE) < e_tol:
                print("convergence reached. Stop Iteration.")
                break

        return self.E

    def solve_Ham_eff(self):
        """
        solve the generalized eigenvalue problem.
        H_eff A= \epsilon O A
        epsilon and A are the generalized eigenvals and the eigen vectors
        """
        #type=1: normalize with v.conj() b v=1
        self.eigenvals, self.eigenvecs = eigh(self.ham_eff, b=self.O, type=1)

        # order the eigen values in ascending order.
        ind = np.argsort(self.eigenvals)
        self.eigenvals = self.eigenvals[ind]
        self.eigenvecs = self.eigenvecs[:, ind]

    def calc_rho(self):
        """
        the density matrix rho(p,q)
        """
        f = fermi(self.eigenvals, self.efermi, self.width)
        #print f.shape
        self.rho = (self.eigenvecs * f).dot(self.eigenvecs.transpose())
        if self.restricted:
            E = self.eigenvals[:self.N / 2].sum() + (self.rho.transpose() *
                                                     self.ham0).sum()
        else:
            #E=(0.5*self.eigenvals[:self.N].sum()+0.5*(self.rho.transpose()*self.ham0).sum())
            E = np.vdot(self.eigenvals, f) / 2.0 + (self.rho.transpose() *
                                                    self.ham0).sum() / 2.0
        self.E = E
        #print(self.E)

    def get_energy(self):
        return self.E


class RHF(HartreeFock):
    """
    Restricted Hartree-Fock solver
    """

    def __init__(self,
                 O=None,
                 ham0=None,
                 Upqrs=dict(),
                 nbasis=0,
                 nelectron=None,
                 width=0.2):
        """
        :param O: overlap matrix
        :param ham0: single body hamiltonian
        :param Upqrs: the many body part of the hamiltonian. sould be a dict or a function.
        :param nbasis: number of basis,
        :param N: number of elelctrons,default 0
        :param use_sparse: True| False. whether to use sparse matrix
        :param nspin: number of spin. default 1, currently spin=2 and spinor=False is not implemented. So please use spinor mode.
        :param width: fermi function width as in  E/width
        :param spinor: spinor mode True|False
        :param restricted: Restricted Hartree Fock or Unrestricted. Restricted should be used with nspin=1, unrestricted should be used with nspin=2 and spinor=True.
        """
        super(RHF, self).__init__(
            O=O,
            ham0=ham0,
            Upqrs=Upqrs,
            nbasis=nbasis,
            nelectron=nelectron,
            width=width,
            nspin=1)

    def eff_ham(self, rho, restricted=True):
        """
        for restricted hf:
        Heff=ham0+Sum_{r,s}[(2*Urpsq-Urpqs)*rho_{rs}]
        :param rho: a n*n matrix, rho(r,s)=sum(a,<r,a|fermifunc(a))|s,a>.
        :param restricted: True| False. restricted or none restricted.
        """
        self.ham_eff = self.ham0 + HF_U(
            self.Upqrs,
            self.nbasis,
            self.rho,
            restricted=True,
            spin_indexed=False,
            density_only=False)
        return self.ham_eff

    def calc_rho(self):
        """
        the density matrix rho(p,q)
        """
        f = fermi(self.eigenvals, self.efermi, self.width)
        self.rho = (self.eigenvecs * f).dot(self.eigenvecs.transpose())
        E = self.eigenvals[:self.N // 2].sum() + (self.rho.transpose() *
                                                  self.ham0).sum()
        self.E = E


class UHF(HartreeFock):
    """
    Unrestricted Hartree-Fock solver
    """

    def __init__(self,
                 O=None,
                 ham0=None,
                 Upqrs=dict(),
                 nbasis=0,
                 nelectron=None,
                 width=0.2):
        """
        :param O: overlap matrix
        :param ham0: single body hamiltonian
        :param Upqrs: the many body part of the hamiltonian. sould be a dict or a function.
        :param nbasis: number of basis,
        :param N: number of elelctrons,default 0
        :param use_sparse: True| False. whether to use sparse matrix
        :param nspin: number of spin. default 1, currently spin=2 and spinor=False is not implemented. So please use spinor mode.
        :param width: fermi function width as in  E/width
        :param spinor: spinor mode True|False
        :param restricted: Restricted Hartree Fock or Unrestricted. Restricted should be used with nspin=1, unrestricted should be used with nspin=2 and spinor=True.
        """
        super(UHF, self).__init__(
            O=O,
            ham0=ham0,
            Upqrs=Upqrs,
            nbasis=nbasis,
            nelectron=nelectron,
            width=width,
            nspin=2)

    def eff_ham(self, rho, restricted=True):
        """
        for restricted hf:
        Heff=ham0+Sum_{r,s}[(2*Urpsq-Urpqs)*rho_{rs}]
        :param rho: a n*n matrix, rho(r,s)=sum(a,<r,a|fermifunc(a))|s,a>.
        :param restricted: True| False. restricted or none restricted.
        """
        self.ham_eff = self.ham0 + HF_U(
            self.Upqrs,
            self.nbasis,
            self.rho,
            restricted=False,
            spin_indexed=True,
            density_only=False)
        return self.ham_eff

    def calc_rho(self):
        """
        the density matrix rho(p,q)
        """
        f = fermi(self.eigenvals, self.efermi, self.width)
        self.rho = (self.eigenvecs * f).dot(self.eigenvecs.transpose())
        E = 0.5 * self.eigenvals[:self.N].sum() + 0.5 * (self.rho.transpose() *
                                                         self.ham0).sum()
        self.E = E


def func_to_dict(U_func, nbasis, thr=0.001):
    """
    Ufunc to dict
    """
    Upqrs = dict()
    indexes = itertools.product(*([range(nbasis)] * 4))
    for i in indexes:
        val = U_func(*i)
        if abs(val) > thr:
            Upqrs[tuple(i)] = val
    return Upqrs


def func_to_array(U_func, nbasis, thr=0.001):
    """
    U() to U[]
    """
    Upqrs = []
    indexes = itertools.product(*([range(nbasis)] * 4))
    for i in indexes:
        val = U_func(*i)
        if abs(val) > thr:
            Upqrs.append(i + [val])
    return np.asarray(Upqrs)


def HF_U(U_func,
         nbasis,
         rho,
         spinor=True,
         restricted=False,
         spin_indexed=True,
         density_only=True):
    """
    use unrestricted Hartree Fock approximation to reduce Urpsq to quadratic from U_eff(p,q)
     .. math::
        HU(p,q)=\\sum_{rs}[U_{rpsq} (\\rho_{\sigma}(r,s)+\\rho_{\\bar{\\sigma}}(r,s)-U_{rpqs} \\rho_{\\sigma}(r,s)]

    :param U_func: a dict U[(p,q,r,s)] which represents the function Upqrs(p,q,r,s). In UHF, the spin indices are included in the indices p,q,r,s.
    :param nbasis: the number of basis.
    :param rho: rho(p,q), if unrestricted and not spin_indexed, a list of two rho matrix rho_up and rho_down should be specified.
    :param density_only: (bool) use density instead of density matrix. That is :math:`\\rho_{mn} =\\rho_{mn} \\delta_{m,n}` . This is the normal LDA+U method.
    :Returns:
     if restricted or spin_indexed: a matrix Hu. Hu(p,q)
     if unrestricted and not spin_indexed: two matrix Hu_up and Hu_down.
    """
    Upqrs = U_func
    if density_only:
        if spin_indexed:
            rho = np.diag(rho.diagonal())
        else:
            rho_up = np.diag(rho[0].diagonal())
            rho_dn = np.diag(rho[1].diagonal())
            rho = (rho_up, rho_dn)
    if restricted:
        HU = np.zeros((nbasis, nbasis))
        for ind in Upqrs:
            r, p, s, q = ind
            HU[p, q] += 2.0 * Upqrs[(r, p, s, q)] * rho[r, s]
            r, p, q, s = ind
            HU[p, q] -= Upqrs[(r, p, q, s)] * rho[r, s]
    elif spin_indexed:  ## This will be used for Unrestricted HF, since this is more general.
        HU = np.zeros((nbasis, nbasis))
        for ind,u in Upqrs.items():
            #u = Upqrs[ind]
            r, p, s, q = ind
            HU[p, q] += u * rho[r, s]
            r, p, q, s = ind
            HU[p, q] -= u * rho[r, s]
    else:  ## unrestricted & not spin_indexed. This is not going to be used in this code (perhaps).
        HU_up = np.zeros((nbasis, nbasis), dtype=complex)
        HU_down = np.zeros((nbasis, nbasis), dtype=complex)
        rho_up, rho_down = rho
        for ind, u in Upqrs.items():
            r, p, s, q = ind
            h1 = u * (rho_up[r, s] + rho_down[r, s])
            HU_up[p, q] += h1
            HU_down[p, q] += h1
            r, p, q, s = ind
            HU_up[p, q] -= u * (rho_up[r, s])
            HU_down[p, q] -= u * (rho_down[r, s])
            HU = HU_up, HU_down
    return HU


# IF use cython version. A little faster.
#HF_U=cyfunc.HF_U


def nospin_to_spin(O, ham0, Upqrs, nbasis):
    """
    the order: (b1_up,b2_up,b3_up,...bn_up, b1_dn,b2_dn,...,bn_dn)
    O --> O_spin
    ham0 --> ham0_spin
    Upqrs --> Upqrs_spin

    :param O:
    :param ham0:
    :param Upqrs:
    :param nbasis: number of basis
    """
    #print Upqrs.shape
    #bas = list(range(nbasis))
    #bas_spin = [(i, 'UP') for i in bas] + [(i, 'DOWN') for i in bas]
    bas = gen_basis_set(nsites=1, nlabels=nbasis, nspin=1)
    bas_spin = bas.add_spin_index()
    print(bas)
    print(bas_spin)
    O = np.array(O)
    #zero_array = np.zeros((nbasis, nbasis))
    #O_spin = np.hstack((np.vstack((O, zero_array)), np.vstack(
    #    (zero_array, O))))
    O_spin = np.zeros((nbasis * 2, nbasis * 2), dtype=float)
    O_spin[::2, ::2] = O
    O_spin[1::2, 1::2] = O

    ham0_spin = np.zeros((nbasis * 2, nbasis * 2), dtype=float)
    ham0_spin[::2, ::2] = ham0
    ham0_spin[1::2, 1::2] = ham0

    Upqrs_spin = dict()
    for ind in itertools.product(bas_spin, bas_spin, bas_spin, bas_spin):
        p, q, r, s = ind
        if p.spin == r.spin and q.spin == s.spin:
            U = Upqrs[(p.index // 2, q.index // 2, r.index // 2, s.index // 2)]
        else:
            U = 0.0
        if U > 1e-5:
            Upqrs_spin[p.index, q.index, r.index, s.index] = U

    return O_spin, ham0_spin, Upqrs_spin


def nospin_to_spin_spinblock(O, ham0, Upqrs, nbasis):
    """
    the order: (b1_up,b2_up,b3_up,...bn_up, b1_dn,b2_dn,...,bn_dn)
    O --> O_spin
    ham0 --> ham0_spin
    Upqrs --> Upqrs_spin

    :param O:
    :param ham0:
    :param Upqrs:
    :param nbasis: number of basis
    """
    #print Upqrs.shape
    bas = list(range(nbasis))
    bas_spin = [(i, 'UP') for i in bas] + [(i, 'DOWN') for i in bas]
    #print bas_spin
    O = np.array(O)
    zero_array = np.zeros((nbasis, nbasis))
    O_spin = np.hstack((np.vstack((O, zero_array)), np.vstack((zero_array,
                                                               O))))
    ham0_spin = np.hstack((np.vstack((ham0, zero_array)), np.vstack(
        (zero_array, ham0))))
    Upqrs_spin = dict()
    for ind in itertools.product(*([range(nbasis * 2)] * 4)):
        p, q, r, s = ind
        if bas_spin[p][1]==bas_spin[r][1] and \
            bas_spin[q][1]==bas_spin[s][1]:
            #print bas_spin[p][0],bas_spin[q][0],bas_spin[r][0],bas_spin[s][0]
            U = Upqrs[bas_spin[p][0], bas_spin[q][0], bas_spin[r][0], bas_spin[
                s][0]]

        else:
            U = 0.0
        if U > 1e-5:
            Upqrs_spin[p, q, r, s] = U

    return O_spin, ham0_spin, Upqrs_spin

if __name__ == '__main__':
    main()
