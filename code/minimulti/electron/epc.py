import numpy as np
import matplotlib.pyplot as plt
from minimulti.electron.plot import plot_band_weight
from minimulti.math import Pert


def epc_shift(kpts, evecs, dham, evals=None, onsite=True, order=1):
    onsite_energies = np.diag(dham[(0, 0, 0)])
    if not onsite:
        np.fill_diagonal(dham[(0, 0, 0)], 0.0)
    nband, nkpt, norb = evecs.shape
    ret = np.zeros((nband, nkpt), dtype=float)
    for ik, k in enumerate(kpts):
        Hk = np.zeros((norb, norb), dtype=complex)
        for R, HR in dham.items():
            phase = np.exp(2.0j * np.pi * np.dot(k, R))
            Hk += HR * phase
        Hk += Hk.T.conjugate()
        #for ib in range(nband):
        #    ret[ib, ik] = np.real(
        #        np.vdot(evecs[ib, ik, :], np.dot(Hk, evecs[ib, ik, :])))
        evecs_ik = evecs[:, ik, :].T
        if order == 1:
            #ret[:, ik] = np.diag(evecs_ik.T.conj().dot(Hk).dot(evecs_ik))
            ret[:, ik] = Pert.Epert1(evecs_ik, Hk)
        elif order == 2 and evals is not None:
            ret[:, ik] = Pert.Epert2(evals[:, ik], evecs_ik, Hk, norb)
        else:
            raise ValueError(
                "Error in calculating epe shift, for order should be 1 or 2, and for order2, evals should be given."
            )

    if not onsite:
        np.fill_diagonal(dham[(0, 0, 0)], onsite_energies)
    return ret


class EPC(object):
    def __init__(self, epc_dict=None, norb=None):
        if epc_dict is not None:
            self._epc = epc_dict
            for key, val in epc_dict.items():
                self._norb = val.shape[0]
                break

        elif norb is not None:
            self._epc = {}
            self._norb = norb

    def to_spin_polarized(self):
        norb = self._norb * 2
        epc = dict()
        for R, val in self._epc.items():
            epc[R] = np.zeros((norb, norb), dtype=float)
            epc[R][::2, ::2] = val
            epc[R][1::2, 1::2] = val
        return EPC(epc)

    @property
    def epc(self):
        return self._epc

    @property
    def norb(self):
        return self._norb

    def add_term(self, R, i, j, val):
        if R not in self._epc:
            self._epc[R] = np.zeros((self._norb, self._norb), dtype=float)
        self._epc[R][i, j] = val

    def add_term_R(self, R, mat):
        self._epc[R] = mat

    def get_band_shift(self, kpts, evecs, evals=None, order=1, onsite=True):
        return epc_shift(
            kpts, evecs, self._epc, evals=evals, order=order, onsite=onsite)

    def plot_epc_fatband(self,
                         kpts,
                         evals,
                         evecs,
                         k_x,
                         X,
                         order=1,
                         onsite=True,
                         xnames=None,
                         width=5,
                         show=False,
                         efermi=None,
                         axis=None,
                         **kwargs):

        wks = self.get_band_shift(
            kpts, evecs, evals=evals, order=order, onsite=onsite)
        kslist = [k_x] * self._norb
        ekslist = evals
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
        for i in range(self._norb):
            axis.plot(k_x, evals[i, :], color='gray', linewidth=0.3)
        axis.set_ylabel('Energy (eV)')
        axis.set_xlim(k_x[0], k_x[-1])
        axis.set_xticks(X)
        if xnames is not None:
            axis.set_xticklabels(xnames)
        for x in X:
            axis.axvline(x, linewidth=0.6, color='gray')
        if show:
            plt.show()
        return axis
