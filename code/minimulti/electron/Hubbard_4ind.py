"""
The Hubbard term. onsite U.
"""
from itertools import product
from minimulti.electron.HF import HF_U
import numpy as np
import scipy.sparse as sp
from minimulti.electron.spherical_harmonic import V_Slater
from functools import lru_cache
import copy


@lru_cache(maxsize=8)
def V_kanamori(dim, U, J):
    Uprime = U - 2.0 * J
    U_matrix = np.zeros((dim, dim, dim, dim), dtype=float)
    m_range = range(dim)
    for m, mp in product(m_range, m_range):
        if m == mp:
            U_matrix[m, m, mp, mp] = U
        else:
            U_matrix[m, m, mp, mp] = Uprime
            U_matrix[m, mp, mp, m] = J
            U_matrix[m, mp, m, mp] = J
    return U_matrix


@lru_cache(maxsize=8)
def V_Dudarev(dim, U, J):
    Uprime = U 
    U_matrix = np.zeros((dim, dim, dim, dim), dtype=float)
    m_range = range(dim)
    for m, mp in product(m_range, m_range):
        if m == mp:
            U_matrix[m, m, mp, mp] = U
        else:
            U_matrix[m, m, mp, mp] = Uprime
            U_matrix[m, mp, mp, m] = J
            U_matrix[m, mp, m, mp] = J
    return U_matrix


def U_nonspin_to_spin(Un, apply_pauli=False):
    # Note that here V(i up, i up) is U!!
    dim = Un.shape[0] * 2
    Us = np.zeros((dim, dim, dim, dim), dtype=Un.dtype)
    Us[::2, ::2, ::2, ::2] = Un[:, :]
    Us[1::2, 1::2, ::2, ::2] = Un[:, :]
    Us[::2, ::2, 1::2, 1::2] = Un[:, :]
    Us[1::2, 1::2, 1::2, 1::2] = Un[:, :]
    if apply_pauli:
        for i in range(dim):
            Us[i, i, i, i] = 0.0
    return Us


def get_average_uj(v2e):
    m_range = range(v2e.shape[0])
    u_avg = 0
    j_avg = 0
    isum_u = 0
    isum_j = 0
    for i, j in product(m_range, m_range):
        u_avg += v2e[i, i, j, j]
        isum_u += 1
        if i != j:
            j_avg += v2e[i, i, j, j] - v2e[i, j, j, i]
            isum_j += 1
    u_avg /= isum_u
    if isum_j > 0:
        j_avg = u_avg - j_avg/isum_j
    return u_avg, j_avg


class Hubbard_matrix_term(object):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        self.bset = bset
        self.Hubbard_dict = copy.copy(Hubbard_dict)
        for key, val in Hubbard_dict.items():
            if 'L' not in val:
                val['L'] = 2
            self.Hubbard_dict[key] = val
        self.DC = DC
        self.DC_type = DC_type
        self.DC_shift = DC_shift

        self._group_bset()

    def _group_bset(self):
        # correlated subspace: bset with same site and l.
        self._corr_group = {}  # site, l : (U, J, dim, index_bset)

        for site_l, b in self.bset.group_by_site_and_l():
            b=list(b)
            site, l = site_l
            symbol = self.bset.atoms.get_chemical_symbols()[site]
            if symbol in self.Hubbard_dict:
                tmp = self.Hubbard_dict[symbol]
                if 'L' in tmp and l == tmp['L']:
                    U = tmp['U']
                    J = tmp['J']
                    ids = np.array([bb.index for bb in b])
                    Umat = self._U_site(site, l, U, J, ids)
                    self._corr_group[site_l] = (U, J, ids, Umat)

    def _U_site(self, site, l, U, J, ids):
        U_nospin = V_Slater(l, U, J)
        return U_nonspin_to_spin(U_nospin)

    def V_and_E(self, rho):
        """
        E=E_band - V_int rho + 0.5* V_int rho -V_DC rho + 0.5 * V_DC rho -E_DC
        =E_band -0.5* Vee * rho - E_DC
        """
        nbasis = len(self.bset)
        Vee = np.zeros((nbasis, nbasis), dtype=complex)
        E_eig = 0.0
        E_int = 0.0
        E_DC = 0.0
        E = 0.0
        for g, UJ_ind in self._corr_group.items():
            site, l = g
            U, J, ids, Umat = UJ_ind
            ind = np.ix_(ids, ids)
            V_int_site, E_int_site = self._Int_site(Umat, rho[ind])
            Vee[ind] += V_int_site
            E_int += E_int_site

            V_DC_site, E_DC_site = self._DC_site(U, J, Umat, rho[ind])
            Vee[ind] += V_DC_site
            E_DC += E_int_site + E_DC_site

            E_eig += -np.sum(Vee[ind] * rho[ind])

        E = E_eig + E_int + E_DC
        return Vee, E

    def _Int_site(self, Umat, rho):
        Vint= np.einsum('ijkl, kl -> ij', Umat, rho) \
            - np.einsum('ikjl, kl -> ij', Umat, rho)
        Eint = 0.5 * np.sum(Vint * rho.T)
        return Vint, Eint

    def _DC_site(self, U, J, Umat, rho):
        n_tot = np.trace(rho)
        if self.DC_type == 'FLL-s':
            n_up = np.sum(rho.diagonal()[::2])
            n_dn = np.sum(rho.diagonal()[1::2])
        elif self.DC_type == 'FLL-ns':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
        else:
            raise ValueError('DC_ype should be FLL-s or FLL-ns')

        V_DC_diag = np.zeros(rho.shape[0], dtype=np.complex)
        V_DC_diag[:] = -U * (n_tot - 0.5)
        V_DC_diag[::2] += J * (n_up - 0.5)
        V_DC_diag[1::2] += J * (n_dn - 0.5)
        V_DC = np.zeros_like(rho, dtype=np.complex)
        V_DC[np.diag_indices(rho.shape[0])] = V_DC_diag[:]

        # Note the minus is already here.
        E_DC = -0.5 * U * n_tot * (n_tot - 1.0) + 0.5 * J * (
            n_up * (n_up - 1.0) + n_dn * (n_dn - 1.0))
        return V_DC, E_DC


class Lichtenstein_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        super(Lichtenstein_term, self).__init__(bset, Hubbard_dict, DC,
                                                DC_type, DC_shift)

    def _U_site(self, site, l, U, J, ids):
        U_nospin=V_Slater(l, U, J, to_eg_t2g=True )
        U=U_nonspin_to_spin(U_nospin)
        Uavg, Javg=get_average_uj(U_nospin)
        #print("Uavg: ", Uavg)
        #print("Javg: ", Javg)
        return U



class Lichtenstein2_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        super(Lichtenstein2_term, self).__init__(bset, Hubbard_dict, DC,
                                                DC_type, DC_shift)

    def _V_and_E_site(self, U, J, rho):
        rd=rho.diagonal()
        rd_up=rd[::2]
        rd_dn=rd[1::2]
        n_tot = np.trace(rho)
        n_up = np.sum(rho.diagonal()[::2])
        n_dn = np.sum(rho.diagonal()[1::2])

        # A_mat: For same spin channel
        A_mat = np.array([[0, -0.52, -0.52, 0.52, 0.52],
                          [-0.52, 0, 0.86, -0.17,-0.17],
                          [-0.52, 0.86, 0, -0.17,-0.17],
                          [0.52, -0.17, -0.17, 0, -0.17],
                          [0.52, -0.17, -0.17, -0.17, 0]])
        # B_mat: For opposite spin channel
        B_mat = np.array([[1.14, -0.63, -0.63, 0.06,0.06],
                          [-0.63, 1.14, 0.29, -0.40,-0.40],
                          [-0.63, 0.29, 1.14, -0.40, -0.40],
                          [0.06, -0.40, -0.40, 1.14,-0.40],
                          [0.06, -0.40, -0.40, -0.40, 1.14]])
        if self.DC_type == 'FLL-ns':
            A_mat = A_mat #- 0.5
            B_mat = B_mat #+ 0.5
        elif self.DC_type == "FLL-s":
            pass
        else:
            raise ValueError("DC_type can only be FLL-ns or FLL-s")

        V=(U-J)*(0.5*np.eye(10, dtype=float)-rho[:,:])

        Vd=np.zeros(10)
        Vd[::2]+=J*(np.dot(A_mat, rd_up)+np.dot(B_mat, rd_dn)) -0.5*J* (n_up-n_dn)
        Vd[1::2]+=J*(np.dot(A_mat, rd_dn)+np.dot(B_mat, rd_up))+0.5*J*(n_up-n_dn)

        V+=np.diag(Vd)
        #print("Rho")
        #print(rho)
        #print("V")
        #print(V)

        #E=0.5*(U-J)*np.sum(rd-rd**2)+  \
        E=0.5*(U-J)*np.trace(rho-np.dot(rho, rho))+  \
            + 0.5*J*np.dot(rd_up,np.dot(A_mat, rd_up))  \
            + 0.5*J*np.dot(rd_dn,np.dot(A_mat, rd_dn))  \
            + 0.5*J*np.dot(rd_up,np.dot(B_mat, rd_dn))  \
            + 0.5*J*np.dot(rd_dn,np.dot(B_mat, rd_up))  \
            - 0.25*J*(n_up-n_dn)**2
        return V, E

    def V_and_E(self, rho):
        """
        E=E_band - V_int rho + 0.5* V_int rho -V_DC rho + 0.5 * V_DC rho -E_DC
        =E_band -0.5* Vee * rho - E_DC
        """
        nbasis = len(self.bset)
        Vee = np.zeros((nbasis, nbasis), dtype=complex)
        E_eig = 0.0
        E_int = 0.0
        E = 0.0
        for g, UJ_ind in self._corr_group.items():
            site, l = g
            U, J, ids, Umat = UJ_ind
            ind = np.ix_(ids, ids)
            Vee_site, E_site = self._V_and_E_site(U, J, rho[ind])
            Vee[ind] += Vee_site
            E_int+=E_site
            E_eig += -np.sum(Vee_site * rho[ind].transpose())
        E = E_eig + E_int
        #print("Warning: using wrong energy expression , just for debuging")
        #E= 0.5*E_eig
        return Vee, E




class Kanamori_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        super(Kanamori_term, self).__init__(bset, Hubbard_dict, DC, DC_type,
                                            DC_shift)

    def _U_site(self, site, l, U, J, ids):
        dim = len(
            ids
        ) // 2  # note that spin index is included in basis, but not in V_kanamori
        U_nospin = V_kanamori(dim, U, J)
        return U_nonspin_to_spin(U_nospin)

    def _DC_site(self, U, J, Umat, rho):
        dim = rho.shape[0] / 2
        n_tot = np.trace(rho)
        if self.DC_type == 'FLL-s':
            n_up = np.sum(rho.diagonal()[::2])
            n_dn = np.sum(rho.diagonal()[1::2])
        elif self.DC_type == 'FLL-ns':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
            Javg = Uavg - (U - 3 * J)
        elif self.DC_type == 'Held':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            Uavg = (U - (5.0 * dim - 5.0) / (2 * dim - 1) * J)
            Javg = 0.0
        else:
            raise ValueError('DC_ype should be FLL-s or FLL-ns')

        V_DC_diag = np.zeros(rho.shape[0], dtype=np.complex)
        V_DC_diag[:] = -Uavg * (n_tot - 0.5)
        V_DC_diag[::2] += Javg * (n_up - 0.5)
        V_DC_diag[1::2] += Javg * (n_dn - 0.5)
        V_DC = np.zeros_like(rho, dtype=np.complex)
        V_DC[np.diag_indices(rho.shape[0])] = V_DC_diag[:]
        # Note the minus is already here.
        E_DC = -0.5 * Uavg * n_tot * (n_tot - 1.0) + 0.5 * Javg * (
            n_up * (n_up - 1.0) + n_dn * (n_dn - 1.0))
        return V_DC, E_DC


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
        E = 0.5 * (np.dot(np.diag(self.UJ_diag),
                          rho - np.dot(rho, rho))).trace()
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
                Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Javg = Uavg - (U - 3 * J)
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
                Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
                Javg = Uavg - (U - 3 * J)
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

        else:
            raise ValueError("Invalid DC_type.")
        return V_DC, E_DC
