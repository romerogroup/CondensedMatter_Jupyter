from itertools import product
import numpy as np
from minimulti.electron.basis2 import Basis, BasisSet
import scipy.sparse as sp


class Hubbard_term(object):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL',
                 DC_shift=False):
        self.bset = bset
        self.Hubbard_dict = Hubbard_dict
        self.DC = DC
        self.DC_type = DC_type
        self.DC_shift = DC_shift

        self.UJ_site = None
        self.UJ_diag = None
        self.UJ_mat = None

    def get_UJmat(self):
        pass

    def V_and_E(self, rho):
        pass

    def _VDC_and_EDC(self, rho):
        return 0.0, 0.0


class Hubbard_U_Dudarev(Hubbard_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL',
                 DC_shift=False):
        super(Hubbard_U_Dudarev, self).__init__(bset, Hubbard_dict, DC,
                                                DC_type, DC_shift)
        self._prepare()

    def _prepare(self):
        symbols = self.bset.atoms.get_chemical_symbols()
        self.UJ_site = np.zeros(self.bset.get_nsites())
        self.UJ_diag = np.zeros(len(self.bset))
        for site, site_bset in self.bset.group_by_site():
            if symbols[site] in self.Hubbard_dict:
                U = self.Hubbard_dict[symbols[site]]['U']
                J = self.Hubbard_dict[symbols[site]]['J']
                self.UJ_site[site] = U - J
                site_bset = list(site_bset)
                for bp in site_bset:
                    self.UJ_diag[bp.index] = U - J

    def V_and_E(self, rho):
        """
        E=1/2 (U-J) n (1-n)
        V= (U-J) * (1/2 - n)
        """
        #V = np.dot(np.diag(self.UJ_diag), (0.5 * np.eye(len(self.bset)) - rho))
        nb = len(self.bset)
        V = np.zeros((nb, nb))
        for i in range(nb):
            for j in range(nb):
                if self.bset[i].site == self.bset[j].site:
                    V[i, j] -= self.UJ_diag[i] * rho[i, j]
                if i == j:
                    V[i, j] += self.UJ_diag[i] * 0.5
        E = 0.5 * (np.dot(np.diag(self.UJ_diag), rho - rho * rho)).trace()

        self.E_Uband = 0
        self.E_int = 0
        self.E_DC = 0
        return V, E

    def DC_potential(self, rho):
        """
        No need for double counting because it is already included.
        """
        return 0.0


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


class Hubbard_U_Liechtenstein(Hubbard_term):
    """
    General class for Hubbard U Term.
    """

    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_shift=False,
                 DC_type='FLL-ns'):

        super(Hubbard_U_Liechtenstein, self).__init__(bset, Hubbard_dict, DC,
                                                      DC_type, DC_shift)
        self.nbasis = len(self.bset)
        self._prepare()

    def _prepare(self):
        self.UJ_diag = np.zeros(self.nbasis)
        self.UJmat = sp.lil_matrix((self.nbasis, self.nbasis))

        # matrix for Lechtenstein.
        A_mat = np.array([[0, -0.52, -0.52, 0.52,
                           0.52], [-0.52, 0, 0.86, -0.17,
                                   -0.17], [-0.52, 0.86, 0, -0.17, -0.17],
                          [0.52, -0.17, -0.17, 0,
                           -0.17], [0.52, -0.17, -0.17, -0.17, 0]])

        B_mat = np.array([[1.14, -0.63, -0.63, 0.06,
                           0.06], [-0.63, 1.14, 0.29, -0.40,
                                   -0.40], [-0.63, 0.29, 1.14, -0.40, -0.40],
                          [0.06, -0.40, -0.40, 1.14,
                           -0.40], [0.06, -0.40, -0.40, -0.40, 1.14]])

        if self.DC_type == 'FLL-ns':
            A_mat = A_mat - 0.5
            B_mat = B_mat + 0.5
        elif self.DC_type == "FLL-s":
            pass
        else:
            raise ValueError("DC_type can only be FLL-ns or FLL-s")

        symbols = self.bset.atoms.get_chemical_symbols()

        for site, site_bset in self.bset.group_by_site():
            if symbols[site] in self.Hubbard_dict:
                U = self.Hubbard_dict[symbols[site]]['U']
                J = self.Hubbard_dict[symbols[site]]['J']
                site_bset = list(site_bset)
                for bp in site_bset:
                    self.UJ_diag[bp.index] = (U - J)
                for bp, bq in product(list(site_bset), list(site_bset)):
                    orb_p = label_dict[bp.label]
                    orb_q = label_dict[bq.label]
                    if bp.spin == bq.spin:
                        self.UJmat[bp.index,
                                   bq.index] += J * A_mat[orb_p, orb_q]
                        if orb_p == orb_q:
                            self.UJmat[bp.index, bq.index] -= (U - J)
                    else:
                        self.UJmat[bp.index,
                                   bq.index] += J * B_mat[orb_p, orb_q]
        self.UJmat = sp.csc_matrix(self.UJmat)

    def V_and_E(self, rho):
        rdiag = rho.diagonal()
        V = np.diag(0.5 * self.UJ_diag + self.UJmat.dot(rdiag))
        #V = np.diag(0.5*self.UJ_diag) + self.UJmat * rho + self.DC_potential(rho)
        #V = np.diag(0.5 * self.UJ_diag + self.UJmat.dot(rdiag))
        E = 0.5 * np.dot(self.UJ_diag, rdiag) + 0.5 * np.dot(
            self.UJmat.dot(rdiag), rdiag)
        return V, E

    def DC_potential(self, rho):
        """
        already included in V_and_E
        """
        return 0


class Hubbard_U_Kanamori(Hubbard_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        super(Hubbard_U_Kanamori, self).__init__(bset, Hubbard_dict, DC,
                                                 DC_type, DC_shift)
        self._prepare()

    def _prepare(self):
        self.nbasis = len(self.bset)
        #self.UJ_diag = np.zeros(self.nbasis)
        self.UJmat = sp.lil_matrix((self.nbasis, self.nbasis))
        symbols = self.bset.atoms.get_chemical_symbols()
        self.U_site = {}
        self.J_site = {}
        self.dim_site = np.zeros(self.bset.get_nsites(), dtype=int)
        for site, site_bset in self.bset.group_by_site():
            site_bset = list(site_bset)
            if symbols[site] in self.Hubbard_dict:
                U = self.Hubbard_dict[symbols[site]]['U']
                J = self.Hubbard_dict[symbols[site]]['J']
                if 'D' in self.Hubbard_dict[symbols[site]]:
                    self.dim_site[site] = self.Hubbard_dict[symbols[site]]['D']
                else:
                    self.dim_site[site] = len(site_bset)
                self.U_site[site] = U
                self.J_site[site] = J
                #print("U,J:", U, J)
                site_bset = list(site_bset)
                for bp, bq in product(list(site_bset), list(site_bset)):
                    #print(bp, bq)
                    if bp.spin == bq.spin:
                        if bp.label != bq.label:
                            self.UJmat[bp.index, bq.index] += (1.0 * U - 3 * J)
                    else:
                        if bp.label == bq.label:
                            self.UJmat[bp.index, bq.index] += 1.0 * U
                        else:
                            self.UJmat[bp.index, bq.index] += (1.0 * U - 2 * J)
        self.UJmat = sp.csc_matrix(self.UJmat)
        #print("UJmat:", self.UJmat)

    def V_and_E(self, rho):
        rdiag = rho.diagonal()
        Vdiag = self.UJmat.dot(rdiag)
        V = np.diag(Vdiag)
        E = 0.5 * np.dot(V, rdiag)

        if self.DC:
            V_DC, E_DC = self.VDC_and_EDC(rho)
            np.fill_diagonal(V, np.diag(V) + V_DC)
            E = E + E_DC
        return V, E

    def VDC_and_EDC(self, rho):
        """
        """
        n = np.zeros((len(self.bset.atoms), 2), dtype=float)
        for b in self.bset:
            n[b.site, b.spin] += rho[b.index, b.index]
        n_tot = np.sum(n, axis=1)
        V_DC = np.zeros(len(self.bset))
        E_DC = 0.0

        if self.DC_type == 'Held':
            # see PhysRevB.90.125114
            #self.DC[b.index] = -(self.Us[b.index]-8.0/5*self.Js[b.index]) * (
            #    n_tot[b.site] - 0.5) + self.Js[b.index] * 7.0/5* (n[b.site, b.spin] -0.5)
            # see
            #self.DC[b.index] = -self.Us[b.index] * (
            #    n_tot[b.site] - 0.5) + self.Js[b.index] * 5.0/3* (n_tot[b.site] -0.5)
            # Held
            for b in self.bset:
                dim = self.dim_site[b.site]
                # see https://triqs.ipht.cnrs.fr/applications/dft_tools/_modules/dft/sumk_dft.html#SumkDFT
                Uavg = (self.U_site[b.site] - (5.0 * dim - 5.0) /
                        (2 * dim - 1) * self.J_site[b.site])
                Javg = 0.0
                V_DC[b.index] = -Uavg * (n_tot[b.site] - 0.5)
            for site in self.U_site:
                dim = self.dim_site[site]
                U = self.U_site[site]
                J = self.J_site[site]
                Uavg = (U - (5.0 * dim - 5.0) / (2 * dim - 1) * J)
                Javg = 0.0
                E_DC = E_DC - 0.5 * Uavg * n_tot[b.site] * (
                    n_tot[b.site] - 1
                )  # + 0.5 * Javg* (n[b.site, 0] * (n[b.site,0]-1.0) + n[b.site, 1]* (n[b.site,1]-1.0))
        elif self.DC_type == 'FLL-s':
            # FLL-spin
            for b in self.bset:
                dim = self.dim_site[b.site]
                U = self.U_site[b.site]
                J = self.J_site[b.site]
                #Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Uavg = U - (2 * dim - 2.0) / dim * J
                Javg = J * (dim + 2.0) / dim  #Uavg - (U - 3 * J)
                V_DC[b.index] = -Uavg * (n_tot[b.site] - 0.5) + Javg * (
                    n[b.site, b.spin] - 0.5)
            for site in self.U_site:
                dim = self.dim_site[site]
                U = self.U_site[site]
                J = self.J_site[site]
                Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Javg = Uavg - (U - 3 * J)
                E_DC = E_DC - 0.5 * Uavg * n_tot[b.site] * (
                    n_tot[b.site] - 1) + 0.5 * Javg * (
                        n[b.site, 0] * (n[b.site, 0] - 1.0) + n[b.site, 1] *
                        (n[b.site, 1] - 1.0))
                #self.E_DC+= -0.5* (self.Us[b.index] - (4.0 * dim - 4.0) /(2 * dim - 1.0) * self.Js[b.index])  *  n_tot[b.site]*(n_tot[b.site]-1.0) * self.Js[b.index] * n[b.site,b.spin]* (n[b.site, b.spin]-1.0)
        elif self.DC_type == "FLL-ns":
            # FLL-nospin
            for b in self.bset:
                dim = self.dim_site[b.site]
                U = self.U_site[b.site]
                J = self.J_site[b.site]
                #Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Uavg = U - (2 * dim - 2.0) / dim * J
                Javg = J * (dim + 2.0) / dim  #Uavg - (U - 3 * J)
                V_DC[b.index] = -Uavg * (n_tot[b.site] - 0.5) + Javg * (
                    n[b.site, b.spin] - 0.5)
            for site in self.U_site:
                dim = self.dim_site[site]
                U = self.U_site[site]
                J = self.J_site[site]

                Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Javg = Uavg - (U - 3 * J)
                E_DC = E_DC - 0.5 * Uavg * n_tot[b.site] * (
                    n_tot[b.site] - 1) + 0.5 * Javg * n_tot[b.site] * (
                        n_tot[b.site] / 2 - 1.0)
        elif self.DC_type == 'NoDC':
            pass

        else:
            raise ValueError("Invalid DC_type.")
        return V_DC, E_DC
