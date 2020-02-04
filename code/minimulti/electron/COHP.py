#!/usr/bin/env python
import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.units import Bohr
from minimulti.electron.plot import plot_band_weight
import pickle


class COHP():
    """
    analyze the tight binding model.
    """

    def __init__(self,
                 tbmodel=None,
                 kpts=None,
                 ham=None,
                 kweights=None,
                 evals=None,
                 evecs=None):
        self.tbmodel = tbmodel
        self.kpts = kpts
        self.kweights = kweights
        self.ham = ham
        self.evals = evals
        self.evecs = evecs
        if kpts is not None and kweights is None:
            nkpts = len(kpts)
            self.kweights = np.ones(nkpts, dtype='float') / nkpts
        if self.evals is not None and self.evecs is not None:
            self.norbs = self.evals.shape[0]
        elif self.tbmodel is not None:
            self.norbs = self.tbmodel.get_num_orbitals()
        else:
            raise ValueError(
                "One of tbmodle and [evals, evecs] should not be None.")

    def calc_cohp_k(self, ham_k, evec_kj):
        """
        calculate COHP for a wavefunction at a point.
        Parameters:
        --------------
        ham_k: The hamiltonian at k. (ndim*ndim), where ndim is the number of wannier functions.
        evec_kj: the jth eigenvector at k point

        Return:
        --------------
        a matrix huv, u&v are the indices of wannier functions.
        """
        cohp_kj = np.outer(np.conj(evec_kj), evec_kj) * ham_k
        #cohp_kj = np.outer(evec_kj, evec_kj) * ham_k
        return np.real(cohp_kj)

    def calc_cohp_allk(self, kpts=None, iblock=None, jblock=None):
        """
        calculate all COHPs.
        """
        if kpts is not None:
            self.kpts = kpts
        nkpts = len(self.kpts)
        if iblock is None:
            iblock = range(self.norbs)
        if jblock is None:
            jblock = range(self.norbs)

        self.cohp = np.zeros((nkpts, self.norbs, len(iblock), len(jblock)))

        if self.evals is None or self.evecs is None:
            self.evals, self.evecs, self.ham = self.tbmodel.solve_all(
                k_list=kpts, eig_vectors=True, total_ham=True)
            for ik, k in enumerate(self.kpts):
                ham_k = self.ham[ik]
                #ham_total, evals_k, evecs_k = self.tbmodel._sol_ham(ham_k, eig_vectors=True, total_ham=True)
                ## for each kpt,band there is a cohp matrix.
                evals_k, evecs_k = self.evals[:, ik], self.evecs[:, ik, :]
                for iband in range(self.norbs):
                    ## note that evec[i,:] is the ith eigenvector
                    evec = evecs_k[iband, :]
                    self.cohp[ik, iband] = self.calc_cohp_k(ham_k, evec)
        else:
            for ik, k in enumerate(self.kpts):
                ham_k = self.ham[ik]
                evals_k, evecs_k = self.evals[:, ik], self.evecs[:, ik, :]
                #self.evals[:,ik] = evals_k
                ## for each kpt,band there is a cohp matrix.
                for iband in range(self.norbs):
                    ## note that evec[i,:] is the ith eigenvector
                    evec = evecs_k[iband, :]
                    self.cohp[ik, iband] = self.calc_cohp_k(ham_k, evec)
        return self.cohp

    def save(self, fname):
        obj = {
            'cohp': self.cohp,
            'evals': self.evals,
            'kpts': self.kpts,
            'kweights': self.kweights
        }
        with open(fname, 'wb') as myfile:
            pickle.dump(obj, myfile)

    def get_cohp_etotal(self, efermi=None, iblock=None, jblock=None):
        cohpij = self.get_cohp_block_pair(iblock, jblock)
        #for ikpt, kpt in enumerate(self.kpts):
        #    for iband in range(self.norbs):
        #        if self.evals[iband, ikpt]<=efermi:
        #            ret+=cohpij[ikpt, iband]
        return np.dot(
            self.kweights,
            np.sum(
                np.where(self.evals.T <= efermi, cohpij,
                         np.zeros_like(cohpij)),
                axis=1))

    def get_cohp_pair(self, i, j):
        return self.cohp[:, :, i, j]

    def get_cohp_block_pair(self, iblock, jblock):
        if iblock is None:
            iblock = range(self.norbs)
        if jblock is None:
            jblock = range(self.norbs)
        iblock = np.array(iblock, dtype=int)
        jblock = np.array(jblock, dtype=int)
        iiblock = np.array(
            list(set(iblock) & set(jblock)),
            dtype=int)  # for removing diagonal terms.
        I, J = np.meshgrid(iblock, jblock)
        # print(self.cohp.shape)
        #return np.sum(self.cohp[:, :, I, J], axis=(2,3))
        return np.einsum('ijkl->ij', self.cohp[:, :, I, J]) - np.einsum(
            'ijk->ij', self.cohp[:, :, iiblock, iiblock])

    def get_cohp_all_pair(self):
        return self.get_cohp_block_pair(range(self.norbs), range(self.norbs))

    def get_cohp_density(self, kpts=None, kweights=None, emin=-20, emax=20):
        """
        cohp(E)= sum_k cohp(k) (\delta(Ek-E))
        """
        if kpts is None:
            kpts = self.kpts
        if kweights is None:
            kweights = self.kweights

    def get_COHP_energy(self):
        """
        COHP as function of energy.
        """
        # raise NotImplementedError('COHP density has not been implemented yet.')
        pass

    def plot_COHP_fatband(self,
                          kpts,
                          k_x,
                          X,
                          xnames=None,
                          width=5,
                          iblock=None,
                          jblock=None,
                          show=False,
                          efermi=None,
                          axis=None,
                          **kwargs):
        self.kpts = kpts
        self.calc_cohp_allk(kpts=kpts)
        if iblock is None:
            wks = self.get_cohp_all_pair()
        else:
            wks = self.get_cohp_block_pair(iblock, jblock)
        wks = np.moveaxis(wks, 0, -1)
        kslist = [k_x] * self.norbs
        ekslist = self.evals
        axis = plot_band_weight(
            kslist,
            ekslist,
            wkslist=wks,
            efermi=efermi,
            yrange=None,
            style='color',
            color='blue',
            width=width,
            axis=axis,
            **kwargs)
        axis.set_ylabel('Energy (eV)')
        axis.set_xlabel('k-point')
        axis.set_xlim(k_x[0], k_x[-1])
        axis.set_xticks(X)
        if xnames is not None:
            axis.set_xticklabels(xnames)
        for x in X:
            axis.axvline(x, linewidth=0.6, color='gray')

        #if show:
        #    plt.show()
        return axis


def load_cohp_from_pickle(fname):
    with open(fname, 'rb') as myfile:
        cohp = pickle.load(fname)
    return cohp
