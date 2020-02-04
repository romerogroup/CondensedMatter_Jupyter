#!/usr/bin/env python
"""
The Hubbard term. onsite U.
"""
from itertools import product
from minimulti.electron.HF import HF_U
import numpy as np
from minimulti.electron.basis2 import Basis, Basis_Set
import scipy.sparse as sp


def Delta(x, y):
    """
    delta function
    """
    return 1 if x == y else 0


def onsite_U_spin_indexed(bset, p, q, r, s, U, J):
    """
    Hubbard Urpsq, the spin index is included in the rpsq index.
    :param bset: basis set.
    :param r,p,s,q: basis indices
    :param U: the U parameter.
    :returns:
      A function of U(p,q,r,s).
    """
    sp, sq, sr, ss = (bset[i].site for i in (p, q, r, s))  #site
    lp, lq, lr, ls = (
        bset[i].label for i in (p, q, r, s))  # label (often orbital)
    spin_p, spin_q, spin_r, spin_s = (bset[i].spin for i in (p, q, r, s))
    if sp == sq == sr == ss:
        if spin_p == spin_r and spin_q == spin_s and lp == lr and lq == ls:
            if spin_p == spin_r == spin_q == spin_s and lp == lr == lq == ls:
                return 0
            else:
                return U
        elif spin_p == spin_r == spin_q == spin_s and lp == ls and lq == lr and lp != lq:
            return J
        else:
            return 0.0
    else:
        return 0.0


def LDAU_Durarev(U,
                 bset,
                 rho,
                 spinor=True,
                 restricted=False,
                 spin_indexed=True,
                 density_only=False):
    """
    Durarev's LDA +U method. The effective potential.
     .. math::
        V^U_{jl}=(\\bar{U}-\\bar{J})(\\frac{1}{2}\\delta_{j,l}-\\rho_{j,l}^\\sigma)

        E^U=\\frac{\\bar{U}-\\bar{J}}{2} \\sum_{l,j,\\sigma}\\rho_{lj}^\\sigma\\rho_{jl}^\\sigma

    :param U: dict of :math:`\\bar{U}-\\bar{J}`, eg. {'Fe':4.5,'Ni':3} ; or a number, same U for all.
    :param bset: the basis set
    :param rho: the density matrix
    :param spinor: use spinor mode or not
    :param restricted: UHF/RHF . No use. always UHF
    :param spin_indexed: if spin_indexed, rho is a single matrix, else rho is two matrices rho_up & rho_dn.
    :param density_only: should be True.

    :Returns: if spin_indexed, HU is a matrix. else two matrices HU_up , HU_dn
    """
    nbasis = len(bset)
    sites = [bset[i].site for i in range(nbasis)]
    elems = [bset[i].element for i in range(nbasis)]
    if isinstance(U, dict):
        Us = [U[elem] for elem in elems]
    else:
        Us = np.ones(nbasis) * U
    spins = [bset[i].spin for i in range(nbasis)]
    if spin_indexed:
        HU = np.zeros((nbasis, nbasis))
        for j in range(nbasis):
            for l in range(nbasis):
                if sites[l] == sites[j] and spins[l] == spins[j]:
                    HU[j, l] += -Us[j] * rho[j, l]
                if j == l:
                    HU[j, l] += 0.5 * Us[j]
    else:
        HU_up = np.zeros((nbasis, nbasis))
        HU_dn = np.zeros((nbasis, nbasis))
        rho_up, rho_dn = rho
        for j in range(nbasis):
            for l in range(nbasis):
                if sites[l] == sites[j]:
                    HU_up[j, l] -= Us[j] * rho_up[j, l]
                    HU_dn[j, l] -= Us[j] * rho_dn[j, l]
                if j == l:
                    HU_up[j, l] += 0.5 * Us[j]
                    HU_dn[j, l] += 0.5 * Us[j]
        HU = (HU_up, HU_dn)

    return HU


class Hubbard_U_Term(object):
    """
    General class for Hubbard U Term.
    """

    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_shift=False):
        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.DC = DC
        self.DC_shift = DC_shift
        self.nbasis = len(self.bset)
        self.Udict = self.basis_to_Udict(bset, Hubbard_dict)

        self.Us = []
        self.Js = []
        self.DC_shifts = []
        if isinstance(Hubbard_dict, dict):
            elems = self.bset.get_chemical_symbols()
            for elem in elems:
                if elem in Hubbard_dict:
                    self.Us.append(Hubbard_dict[elem]['U'])
                    self.Js.append(Hubbard_dict[elem]['J'])
                    if 'DC_shift' in Hubbard_dict[elem]:
                        self.DC_shifts.append(Hubbard_dict[elem]['DC_shift'])
                    else:
                        self.DC_shifts.append(0)
                else:
                    self.Us.append(0.0)
                    self.Js.append(0.0)
                    self.DC_shifts.append(0.0)
        self.Us = np.array(self.Us)
        self.Js = np.array(self.Js)
        self.DC_shifts = np.array(self.DC_shifts)

    def basis_to_Udict(self, bset, Hubbard_U_dict):
        pass

    def potential(self, rho):
        """
        calculate effective H_U. Heff=H_0+H_U
        """
        self.H_U = HF_U(
            self.Udict,
            self.nbasis,
            rho,
            restricted=False,
            spin_indexed=True,
            density_only=False)
        if self.DC:
            np.fill_diagonal(
                self.H_U,
                np.diag(self.H_U) + self.DC_potential(rho))
        if self.DC_shift:
            np.fill_diagonal(self.H_U, np.diag(self.H_U) + self.DC_shifts)
        return self.H_U

    def DC_potential(self, rho):
        """
        """
        n = np.zeros((len(self.bset.atoms), 2), dtype=float)
        for b in self.bset:
            n[b.site, b.spin] += rho[b.index, b.index]
        n_tot = np.sum(n, axis=1)
        self.DC = np.zeros(len(self.bset))
        for b in self.bset:
            self.DC[b.index] = -self.Us[b.index] * (
                n_tot[b.site] - 0.5) + self.Js[b.index] * (
                    n[b.site, b.spin] - 0.5)
        return self.DC

    def energy(self, rho):
        """
        calculate 1/2 H_U rho.
        Note that E_total = 1/2 E_band + 1/2 H_0 rho
        E_band = H_eff rho = (H_0+H_U)rho
        -> E_total = H_0 rho + 1/2 H_U rho.
        """
        return (rho.transpose() * self.H_U).sum() / 2.0


class Hubbard_U_Dudarev(object):
    """
    General class for Hubbard U Term.
    """

    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=False,
                 DC_shift=False):
        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.nbasis = len(self.bset)
        self.Us = self.basis_to_Udict(bset, Hubbard_dict)

    @staticmethod
    def basis_to_Udict(bset, Hubbard_dict):
        nbasis = len(bset)
        elems = bset.get_chemical_symbols()
        Us = []
        if isinstance(Hubbard_dict, dict):
            for elem in elems:
                if elem in Hubbard_dict:
                    Us.append(Hubbard_dict[elem]['U'] -
                              Hubbard_dict[elem]['J'])
                else:
                    Us.append(0.0)
        else:
            Us = np.ones(nbasis) * Hubbard_dict
        return Us

    def potential(self, rho, spin_indexed=True):
        """
        calculate effective H_U. Heff=H_0+H_U
        """
        bset = self.bset
        spins = [b.spin for b in bset]
        sites = bset.get_sites()
        nbasis = len(bset)
        Us = self.Us
        if spin_indexed:
            HU = np.zeros((nbasis, nbasis))
            for j in range(nbasis):
                for l in range(nbasis):
                    if sites[l] == sites[j] and spins[l] == spins[j]:  # non diagonal
                        #if j==l: # diagonal
                        HU[j, l] += -Us[j] * rho[j, l]
                    if j == l:
                        HU[j, l] += 0.5 * Us[j]
        else:
            HU_up = np.zeros((nbasis, nbasis))
            HU_dn = np.zeros((nbasis, nbasis))
            rho_up, rho_dn = rho
            for j in range(nbasis):
                for l in range(nbasis):
                    if sites[l] == sites[j]:
                        HU_up[j, l] -= Us[j] * rho_up[j, l]
                        HU_dn[j, l] -= Us[j] * rho_dn[j, l]
                    if j == l:
                        HU_up[j, l] += 0.5 * Us[j]
                        HU_dn[j, l] += 0.5 * Us[j]
            HU = (HU_up, HU_dn)
        self.HU = HU
        #if self.DC_shift:
        #    np.fill_diagonal(self.H_U, np.diag(self.H_U) + self.DC_shifts)
        return self.HU

    def energy(self, rho):
        """
        calculate 1/2 H_U rho.
        Note that E_total = 1/2 E_band + 1/2 H_0 rho
        E_band = H_eff rho = (H_0+H_U)rho
        -> E_total = H_0 rho + 1/2 H_U rho.
        """
        # Hu= (1/2-rho) -> E= U(1/2-rho)rho/2. which is wrong. The correction should be 1/4*U*rho
        return (rho.transpose() * self.HU).sum() / 2.0


A_mat = np.array(
    [[0.00, -0.52, -0.52, 0.52, 0.52],
     [-0.52, 0.00, 0.86, -0.17, -0.17],
     [-0.52, 0.86, 0.00, -0.17, -0.17],
     [0.52, -0.17, -0.17, 0.00, -0.17],
     [0.52, -0.17, -0.17, -0.17, 0.00]])

B_mat = np.array([[1.14, -0.63, -0.63, 0.06,
                   0.06], [-0.63, 1.14, 0.29, -0.40,
                           -0.40], [-0.63, 0.29, 1.14, -0.40,
                                    -0.40], [0.06, -0.40, -0.40, 1.14, -0.40],
                  [0.06, -0.40, -0.40, -0.40, 1.14]])

label_dict = {
    'd3z^2-r^2': 0,
    'd3z2-r2': 0,
    'dz^2': 0,
    'dz2': 0,
    'dx^2-y^2': 1,
    'dx2-y2': 1,
    'dx^2': 1,
    'dx2': 1,
    'dxy': 2,
    'dyz': 3,
    'dxz': 4
}


class Hubbard_U_Liechtenstein(Hubbard_U_Term):
    """
    General class for Hubbard U Term.
    """

    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_shift=False,
                 DC_type='FLL-ns'):

        super(Hubbard_U_Liechtenstein,
              self).__init__(bset, Hubbard_dict, DC, DC_shift)
        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.nbasis = len(self.bset)
        self.DC_type = DC_type

        self.UJ_diag = np.zeros(self.nbasis)
        self.UJmat = sp.lil_matrix((self.nbasis, self.nbasis))

        symbols = bset.atoms.get_chemical_symbols()
        for site, site_bset in bset.group_by_site():
            if symbols[site] in Hubbard_dict:
                U = Hubbard_dict[symbols[site]]['U']
                J = Hubbard_dict[symbols[site]]['J']
                #print("U,J:", U, J)
                site_bset = list(site_bset)
                for bp in site_bset:
                    self.UJ_diag[bp.index] = 0.5 * (U - J)
                for bp, bq in product(list(site_bset), list(site_bset)):
                    #print(bp, bq)
                    orb_p = label_dict[bp.label]
                    orb_q = label_dict[bq.label]
                    if bp.spin == bq.spin:
                        self.UJmat[bp.index,
                                   bq.index] += J * A_mat[orb_p, orb_q]
                        if orb_p == orb_q:
                            #print("Herre2")
                            self.UJmat[bp.index, bq.index] -= (U - J)
                    else:
                        self.UJmat[bp.index,
                                   bq.index] += J * B_mat[orb_p, orb_q]
        self.UJmat = sp.csc_matrix(self.UJmat)
        #print("UJdiag: ", self.UJ_diag)
        #print("UJmat:", self.UJmat)

    def potential(self, rho, spin_indexed=True):
        """
        calculate effective H_U. Heff=H_0+H_U
        """
        #self.HU = np.diag(self.UJ_diag + self.UJmat.dot(rho.diagonal()))
        #self.energy=0.0
        self.HU = np.diag(self.UJ_diag + self.UJmat.dot(rho.diagonal()) +
                          self.DC_potential(rho))
        #np.fill_diagonal(
        #        self.HU,
        #        np.diag(self.HU) + self.DC_potential(rho))

        #if self.DC_shift:
        #    np.fill_diagonal(self.H_U, np.diag(self.H_U) + self.DC_shifts)
        return self.HU

    def energy(self, rho):
        """
        calculate 1/2 H_U rho.
        Note that E_total = 1/2 E_band + 1/2 H_0 rho
        E_band = H_eff rho = (H_0+H_U)rho
        -> E_total = H_0 rho + 1/2 H_U rho.
        """
        return -(rho.transpose() * self.HU).sum() / 2.0 + 0.5 * np.dot(
           self.UJ_diag, rho.diagonal())

    def DC_potential(self, rho):
        """
        """
        n = np.zeros((len(self.bset.atoms), 2), dtype=float)
        for b in self.bset:
            n[b.site, b.spin] += rho[b.index, b.index]
        n_tot = np.sum(n, axis=1)
        self.DC = np.zeros(len(self.bset))
        if self.DC_type == 'FLL-ns':
            for b in self.bset:
                # Difference between nospin and spin
                self.DC[b.index] = -self.Js[b.index] * (
                    n[b.site, b.spin] - 0.5) + self.Js[b.index] * (
                        n_tot[b.site] / 2 - 0.5)
        elif self.DC_type == 'FLL-s':
            pass
        else:
            raise ("Only FLL-ns or FLL-s is allowed for Liechtenstein U.")
        return self.DC

    def V_and_E(self, rho):
        V=self.potential(rho)
        E=self.energy(rho)
        return V, E


class Hubbard_U_Kanamori(Hubbard_U_Term):
    #class Hubbard_U_Kanamori(object):
    """
    General class for Hubbard U Term.
    """

    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 dim=5,
                 DC_shift=False,
                 DC_type='FLL-ns'):
        super(Hubbard_U_Kanamori, self).__init__(bset, Hubbard_dict,
                                                 DC, DC_shift)
        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.DC = DC
        self.DC_type = DC_type
        self.DC_shift = DC_shift
        self.nbasis = len(self.bset)

        #self.UJ_diag = np.zeros(self.nbasis)
        self.UJmat = sp.lil_matrix((self.nbasis, self.nbasis))
        symbols = bset.atoms.get_chemical_symbols()
        for site, site_bset in bset.group_by_site():
            if symbols[site] in Hubbard_dict:
                U = Hubbard_dict[symbols[site]]['U']
                J = Hubbard_dict[symbols[site]]['J']
                #print("U,J:", U, J)
                site_bset = list(site_bset)
                for bp, bq in product(list(site_bset), list(site_bset)):
                    #print(bp, bq)
                    orb_p = label_dict[bp.label]
                    orb_q = label_dict[bq.label]
                    if bp.spin == bq.spin:
                        if bp.label != bq.label:
                            self.UJmat[bp.index, bq.index] += (1.0 * U - 3 * J)
                    else:
                        if bp.label == bq.label:
                            self.UJmat[bp.index, bq.index] += 1.0 * U
                        else:
                            self.UJmat[bp.index, bq.index] += (1.0 * U - 2 * J)
        self.UJmat = sp.csc_matrix(self.UJmat)
        self.dim = dim
        #print("UJmat:", self.UJmat)

    def potential(self, rho, spin_indexed=True):
        """
        calculate effective H_U. Heff=H_0+H_U
        """

        #self.HU = np.diag(self.UJmat.dot(rho.diagonal()) + self.DC_potential(rho))
        self.HU = np.diag(self.UJmat.dot(rho.diagonal()))
        if self.DC:
            np.fill_diagonal(
                self.HU,
                np.diag(self.HU) + self.DC_potential(rho))
        #if self.DC_shift:
        #    np.fill_diagonal(self.HU, np.diag(self.HU) + self.DC_shifts)
        return self.HU

    def DC_potential(self, rho):
        """
        """
        n = np.zeros((len(self.bset.atoms), 2), dtype=float)
        for b in self.bset:
            n[b.site, b.spin] += rho[b.index, b.index]
        n_tot = np.sum(n, axis=1)
        self.DC = np.zeros(len(self.bset))
        dim = self.dim
        for b in self.bset:
            # see PhysRevB.90.125114
            #self.DC[b.index] = -(self.Us[b.index]-8.0/5*self.Js[b.index]) * (
            #    n_tot[b.site] - 0.5) + self.Js[b.index] * 7.0/5* (n[b.site, b.spin] -0.5)
            # see
            #self.DC[b.index] = -self.Us[b.index] * (
            #    n_tot[b.site] - 0.5) + self.Js[b.index] * 5.0/3* (n_tot[b.site] -0.5)
            # Held
            if self.DC_type == 'Held':
                # see https://triqs.ipht.cnrs.fr/applications/dft_tools/_modules/dft/sumk_dft.html#SumkDFT
                self.DC[b.index] = -(self.Us[b.index] - (5.0 * dim - 5.0) /
                                     (2 * dim - 1) * self.Js[b.index]) * (
                                         n_tot[b.site] - 0.5)
            elif self.DC_type == 'FLL-s':
                # FLL-spin
                self.DC[b.index] = -(self.Us[b.index] - (4.0 * dim - 4.0) /
                                     (2 * dim - 1.0) * self.Js[b.index]) * (
                                         n_tot[b.site] -
                                         0.5) + self.Js[b.index] * (
                                             n[b.site, b.spin] - 0.5)
            elif self.DC_type == "FLL-ns":
                # FLL-nospin
                self.DC[b.index] = -(self.Us[b.index] - (4.0 * dim - 4.0) /
                                     (2 * dim - 1.0) * self.Js[b.index]) * (
                                         n_tot[b.site] -
                                         0.5) + self.Js[b.index] * (
                                             n_tot[b.site] / 2 - 0.5)
            else:
                raise ValueError("Invalid DC_type.")
        return self.DC

    def energy(self, rho):
        """
        calculate 1/2 H_U rho.
        Note that E_total = 1/2 E_band + 1/2 H_0 rho
        E_band = H_eff rho = (H_0+H_U)rho
        -> E_total = H_0 rho + 1/2 H_U rho.
        """
        return (rho.transpose() * self.HU).sum() / 2.0


class Hubbard_U_SUN(Hubbard_U_Term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=False,
                 DC_shift=False):
        """
        The SU(N) onsite Hubbard U term.
        """
        super(Hubbard_U_SUN, self).__init__(bset, Hubbard_dict,
                                            DC, DC_shift)

        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.nbasis = len(bset)
        self.Udict = self.basis_to_Udict(
            bset=self.bset, Hubbard_dict=self.Hubbard_dict)

    @staticmethod
    def basis_to_Udict(bset, Hubbard_dict):
        ret = dict()
        symbols = bset.atoms.get_chemical_symbols()
        for site, site_bset in bset.group_by_site():
            ids = [x.index for x in site_bset]
            indexes = product(ids, ids, ids, ids)
            for ind in indexes:
                p, q, r, s = ind
                if symbols[site] in Hubbard_dict:
                    U = Hubbard_dict[symbols[site]]['U']
                    J = Hubbard_dict[symbols[site]]['J']
                    val = onsite_U_spin_indexed(bset, p, q, r, s, U, J)
                    if abs(val) > 1e-8:
                        ret[(p, q, r, s)] = val
        return ret


def onsite_U_spin_not_indexed(bset, u):
    """
    Hubbard Urpsq, the spin index is included in the rpsq index.

    :param bset: basis set.
    :param r,p,s,q: basis indices
    :param U: the U parameter.

    :returns:
      A function of U(p,q,r,s).
    """

    ids = [x.id for x in bset.get_basis()]
    indexes = product(ids, ids, ids, ids)
    ret = dict()
    for index in indexes:
        id_p, id_q, id_r, id_s = index  #site
        sp, sq, sr, ss = [bset[i].site for i in index]
        if sp == sq == sr == ss:
            ret[(id_p, id_q, id_r, id_s)] = u
    return ret
