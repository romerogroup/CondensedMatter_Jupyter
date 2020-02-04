import sys
from math import log, exp
import numpy as np
#import numba
from numba import vectorize, float64, jit

MAX_EXP_ARGUMENT = log(sys.float_info.max)


@vectorize([float64(float64, float64, float64)])
def fermi(e, mu, width):
    """
    the fermi function.
     .. math::
        f=\\frac{1}{\exp((e-\mu)/width)+1}

    :param e,mu,width: e,\mu,width
    """

    x = (e - mu) / width
    if x < MAX_EXP_ARGUMENT:
        return 1 / (1.0 + exp(x))
    else:
        return 0.0


#fermi = np.vectorize(fermi_s)


def density_matrix(eigenvals, eigenvecs, efermi, width, spin_split=True):
    """
    calculate the density matrix. (not in k space)
     .. math::
        \\rho(p,q)=\\sum_\\alpha A_{p,\\alpha} f(\\alpha-\\mu) A^*_{q,\\alpha}
    and the energy (if calc_E).
     .. math::
        E=\\frac{1}{2} (\\sum_\\alpha\\epsilon_\\alpha f(\\epsilon_\\alpha-\\mu) + \\sum_{qq} H_{pq}^0 \\rho_{qp})


    :param eigenvals: (np.ndarray ) energy levels. indexes (band)
    :param eigenvecs: (np.ndarray ) energy eigenvectors. indexes:(orb,band)
    :param efermi: fermi energy
    :param width: smearing width
    :param spin_split (bool): if true ,return (rho_up,rho_dn)
    :param calc_E: (bool) whether to calculate the Hartree Fock energy.
    :param H0: (ndarray) H0. For calculating the energy.
    :returns:
     rho(p,q)  indexes are [state,state], Note here, spin index are included in p.
    """
    f = fermi(eigenvals, efermi, width)
    rho = (eigenvecs * f).dot(eigenvecs.conj().transpose())
    if not spin_split:
        return rho
    else:
        return rho[::2, ::2], rho[1::2, 1::2]


@jit(nopython=True, fastmath=True, cache=True)
def rhok(occ_k, evec_k, kweight, rho):
    nband, nstate = evec_k.shape
    for i in range(nband):
        for j in range(nstate):
            tmp_ij = kweight * occ_k[i] * evec_k[i, j].conjugate()
            for k in range(nstate):
                rho[j, k] += tmp_ij * (evec_k[i, k])


@jit(nopython=True, fastmath=True, cache=True)
def rhok_diag(occ_k, evec_k, kweight, rho):
    nband, nstate = evec_k.shape
    for i in range(nband):
        tmp = kweight * occ_k[i]
        for j in range(nstate):
            rho[j, j] += tmp * evec_k[i, j] * (evec_k[i, j].conjugate())


def density_matrix_kspace(
        eigenvecs,
        occupations,
        kweight,
        split_spin=True,
        diag_only=True,
        rho_k=None,
):
    """
    calculate the density matrix with multiple k.
     .. math::
        \\rho= \sum_k \\rho_k weight(k)

    :param eigenvecs: the eigenvec matrix. indexes are [band,kpt,orb,spin] or [band,kpt,orb]
    :param occupations: the occupation matrix. indexes are [band,kpt] the same as the eigenvals. Not the same as the eigenvecs.
    :param kweight: (ndarray) used if eigenvalues
    :param split_spin (bool)
    """
    eshape = list(eigenvecs.shape)
    nband, nk, norb = eshape[:3]
    if len(eshape) == 3:
        nspin = 1
    elif len(eshape) == 4:
        nspin = 2
    nstate = norb * nspin

    rho = np.zeros((nstate, nstate), dtype=complex)
    for k in range(nk):
        # evec_k -> [orb,band]
        if nspin == 2:
            evec_k = eigenvecs[:, k, :, :].reshape(nband, nstate)
        elif nspin == 1:  # no spin index
            evec_k = eigenvecs[:, k, :]
        occ_k = occupations[:, k]

        # rho_k is always saved in its full form.
        if rho_k is not None:
            if diag_only:
                rhok(occ_k, evec_k, kweight[k], rho_k[k])
                rhok_diag(occ_k, evec_k, kweight[k], rho)
            else:
                rhok(occ_k, evec_k, kweight[k], rho_k[k])
                rho += rho_k[k]
        else:
            if diag_only:
                rhok_diag(occ_k, evec_k, kweight[k], rho)
            else:
                rhok(occ_k, evec_k, kweight[k], rho)
    if split_spin:
        return np.real(rho[::2, ::2]), np.real(rho[1::2, 1::2])
    else:
        return np.real(rho)


class DensityMatrix(object):
    def __init__(self, kpts=None, kweights=None):
        self.kpts = kpts
        self.kweights = None
        self.diag_only = False

    @property
    def nspin(self):
        return self._nspin

    @property
    def gamma_only(self):
        return (len(self.kpts) == 1 and np.allclose(self.kpts[0], [0, 0, 0]))

    def density_matrix_kspace(
            self,
            eigenvecs,
            occupations,
            kweight,
            split_spin=True,
            diag_only=True,
            rho_k=None,
    ):
        pass
