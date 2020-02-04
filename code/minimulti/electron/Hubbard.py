"""
The Hubbard term. onsite U.
"""
from itertools import product
import numpy as np
from minimulti.electron.U_matrix import (U_matrix, reduce_4index_to_2index,
                                         U_matrix_kanamori, U_matrix_dudarev)
from functools import lru_cache
import copy
from numba import njit


def nospin_to_spin(U4):
    n = U4.shape[0] * 2
    Uspin = np.zeros((n, n, n, n), dtype=U4.dtype)
    # Uijkl * n_ik - Uijlk*n_ik
    Uspin[::2, ::2, ::2, ::2] = U4
    Uspin[1::2, ::2, 1::2, ::2] = U4
    Uspin[::2, 1::2, ::2, 1::2] = U4
    Uspin[1::2, 1::2, 1::2, 1::2] = U4
    return Uspin


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
        j_avg = u_avg - j_avg / isum_j
    return u_avg, j_avg


@njit(fastmath=True, cache=False)
def hartree(U, rho):
    n = rho.shape[0]
    V = np.zeros_like(rho)
    for r in range(n):
        for p in range(n):
            for s in range(n):
                for q in range(n):
                    V[p, q] += U[r, p, s, q] * rho[r, s]
    return V


@njit(fastmath=True, cache=False)
def fock(U, rho):
    n = rho.shape[0]
    V = np.zeros_like(rho)
    for r in range(n):
        for p in range(n):
            for q in range(n):
                for s in range(n):
                    V[p, q] -= U[r, p, q, s] * rho[r, s]
    return V


class Hubbard_matrix_term(object):
    def __init__(
            self,
            bset,
            Hubbard_dict,
            DC=True,
            DC_type='FLL-ns',
            DC_shift=False,
            use_4index=False,
    ):
        self.bset = bset
        self.Hubbard_dict = copy.copy(Hubbard_dict)
        for key, val in Hubbard_dict.items():
            if 'L' not in val:
                val['L'] = 2
            self.Hubbard_dict[key] = val
        self.DC = DC
        self.DC_type = DC_type
        self.DC_shift = DC_shift
        self._use_4index = use_4index

        self._group_bset()

    def _group_bset(self):
        # correlated subspace: bset with same site and l.
        self._corr_group = {}  # site, l : (U, J, dim, index_bset)

        for site_l, b in self.bset.group_by_site_and_l():
            b = list(b)
            site, l = site_l
            symbol = self.bset.atoms.get_chemical_symbols()[site]
            if symbol in self.Hubbard_dict:
                tmp = self.Hubbard_dict[symbol]
                if 'L' in tmp and l == tmp['L']:
                    U = tmp['U']
                    J = tmp['J']
                    ids = np.array([bb.index for bb in b])
                    U4ind, U2, U2prime = self._U_site(site, l, U, J, ids)
                    self._corr_group[site_l] = (U, J, ids, U4ind, U2, U2prime)


    def _U_site(self, site, l, U, J, ids):
        #U_nospin = V_Slater(l, U, J)
        #return U_nonspin_to_spin(U_nospin)
        U4 = U_matrix(l, U_int=U, J_hund=J, basis='spherical')
        U4 = np.real(U4)
        if self._use_4index:
            U4ind = U4  #nospin_to_spin(U4)
            U2 = U2prime = None
        else:
            U4ind = None
            U2, U2prime = reduce_4index_to_2index(U4)
        return U4ind, U2, U2prime

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
            U, J, ids, U4ind, U2, U2prime = UJ_ind
            ind = np.ix_(ids, ids)

            V_int_site, E_int_site = self._Int_site(
                rho[ind], U4=U4ind, U2=U2, U2prime=U2prime)
            Vee[ind] += V_int_site
            E_int += E_int_site
  
            V_DC_site, E_DC_site = self._DC_site(U, J, U2, U2prime, rho[ind])
            Vee[ind] += V_DC_site
            E_DC += E_DC_site

            E_eig += -np.sum(Vee[ind] * rho[ind])
        E = E_eig + E_int + E_DC
        self.E_Uband = E_eig
        self.E_int = E_int
        self.E_DC = E_DC
        return Vee, E

    def _Int_site(self, rho, U4=None, U2=None, U2prime=None):
        """
        calculate Vint and Eint (single site)
        """
        if self._use_4index:
            return self._Int_site_4index(U4, rho)
        else:
            return self._Int_site_2index(U2, U2prime, rho)

    def _Int_site_4index(self, U4, rho):
        """
        calculate Vint and Eint from U4 and rho (single site)
        """
        #Vint= np.einsum('ijkl, jl -> ik', U4, rho) - np.einsum('ijlk, jl -> ik', U4, rho)
        n = rho.shape[0]
        #Vint=np.zeros((n,n), dtype=float)
        Vint = np.zeros_like(rho)

        # hartree
        #Vint[::2, ::2]=np.einsum('ijkl, ik->jl', U4, rho[::2, ::2]+rho[1::2, 1::2])
        #Vint[1::2, 1::2]=np.einsum('ijkl, ik->jl', U4, rho[::2, ::2]+rho[1::2, 1::2])
        Vint[::2, ::2] += hartree(U4, rho[::2, ::2] + rho[1::2, 1::2])
        Vint[1::2, 1::2] += hartree(U4, rho[::2, ::2] + rho[1::2, 1::2])

        # fock
        #Vint[::2,::2]+=-np.einsum('ijlk, ik->jl', U4, rho[::2,::2])
        #Vint[1::2,1::2]+=-np.einsum('ijlk, ik->jl', U4, rho[1::2,1::2])
        Vint[::2, ::2] += fock(U4, rho[::2, ::2])
        Vint[1::2, 1::2] += fock(U4, rho[1::2, 1::2])
        #Eint = 0.5 * np.sum(Vint * rho.conjugate())
        Eint = 0.5 * np.trace(Vint.dot(rho))

        return Vint, Eint

    def _Int_site_2index(self, U2, U2prime, rho):
        n = rho.shape[0]
        rdiag = rho.diagonal()
        Vdiag = np.zeros(n, dtype=float)
        Vdiag[::2] = np.dot(U2, rdiag[::2]) + np.dot(U2prime, rdiag[1::2])
        Vdiag[1::2] = np.dot(U2, rdiag[1::2]) + np.dot(U2prime, rdiag[::2])
        Eint = 0.5 * np.dot(Vdiag, rdiag)
        return np.diag(Vdiag), Eint

    def _DC_site(self, U, J, U2, U2prime, rho):
        """
        calculate double couting V and E.
        """
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
    def __init__(
            self,
            bset,
            Hubbard_dict,
            DC=True,
            DC_type='FLL-ns',
            DC_shift=False,
            use_4index=True,
    ):
        super(Lichtenstein_term, self).__init__(bset, Hubbard_dict, DC,
                                                DC_type, DC_shift, use_4index)

    def _U_site(self, site, l, U, J, ids):
        #U_nospin = V_Slater(l, U, J)
        #return U_nonspin_to_spin(U_nospin)
        U4 = U_matrix(l, U_int=U, J_hund=J, basis='cubic')
        if self._use_4index:
            #U4ind=nospin_to_spin(U4)
            if (np.abs(U4 - np.real(U4)) > 0.01).any():
                print("Waring: Shouldn't cast complex to real")
            U4ind = np.real(U4)
            U2 = U2prime = None
        else:
            U4ind = None
            U2, U2prime = reduce_4index_to_2index(U4)
        #print(U2, U2prime)
        return U4ind, U2, U2prime


class Lichtenstein2_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False,
                 use_4index=False):
        super(Lichtenstein2_term, self).__init__(bset, Hubbard_dict, DC,
                                                 DC_type, DC_shift, use_4index)

    def _V_and_E_site(self, U, J, rho):
        rd = rho.diagonal()
        rd_up = rd[::2]
        rd_dn = rd[1::2]
        n_tot = np.trace(rho)
        n_up = np.sum(rho.diagonal()[::2])
        n_dn = np.sum(rho.diagonal()[1::2])

        # A_mat: For same spin channel
        A_mat = np.array([[0, -0.52, -0.52, 0.52,
                           0.52], [-0.52, 0, 0.86, -0.17,
                                   -0.17], [-0.52, 0.86, 0, -0.17, -0.17],
                          [0.52, -0.17, -0.17, 0,
                           -0.17], [0.52, -0.17, -0.17, -0.17, 0]])
        # B_mat: For opposite spin channel
        B_mat = np.array([[1.14, -0.63, -0.63, 0.06,
                           0.06], [-0.63, 1.14, 0.29, -0.40,
                                   -0.40], [-0.63, 0.29, 1.14, -0.40, -0.40],
                          [0.06, -0.40, -0.40, 1.14,
                           -0.40], [0.06, -0.40, -0.40, -0.40, 1.14]])
        if self.DC_type == 'FLL-ns':
            A_mat = A_mat  #- 0.5
            B_mat = B_mat  #+ 0.5
        elif self.DC_type == "FLL-s":
            pass
        else:
            raise ValueError("DC_type can only be FLL-ns or FLL-s")

        V = (U - J) * (0.5 * np.eye(10, dtype=float) - rho[:, :])

        Vd = np.zeros(10)
        Vd[::2] += J * (np.dot(A_mat, rd_up) + np.dot(
            B_mat, rd_dn)) - 0.5 * J * (n_up - n_dn)
        Vd[1::2] += J * (np.dot(A_mat, rd_dn) + np.dot(
            B_mat, rd_up)) + 0.5 * J * (n_up - n_dn)

        V += np.diag(Vd)
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
            E_int += E_site
            E_eig += -np.sum(Vee_site * rho[ind].transpose())
        E = E_eig + E_int
        #print("Warning: using wrong energy expression , just for debuging")
        #E= 0.5*E_eig
        return Vee, E


class Kanamori_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 use_4index=False,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        use_4index = False
        super(Kanamori_term, self).__init__(bset, Hubbard_dict, DC, DC_type,
                                            DC_shift, use_4index)

    def _U_site(self, site, l, U, J, ids):
        # note that spin index is included in basis, but not in V_kanamori
        dim = len(ids) // 2
        U4ind = None
        U2, U2prime = U_matrix_kanamori(dim, U, J)
        return U4ind, U2, U2prime

    def _DC_site(self, U, J, U2, U2prime, rho):
        dim = rho.shape[0] / 2
        n_tot = np.trace(rho)
        if self.DC_type == 'FLL-s':
            n_up = np.sum(rho.diagonal()[::2])
            n_dn = np.sum(rho.diagonal()[1::2])
            Uavg = U - (2.0 * dim - 2) / dim * J
            Javg = Uavg - (U - 3 * J)
        elif self.DC_type == 'FLL-ns':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            #Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
            Uavg = U - (2.0 * dim - 2) / dim * J
            Javg = Uavg - (U - 3 * J)
        elif self.DC_type == 'Held':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            Uavg = (U - (5.0 * dim - 5.0) / (2 * dim - 1) * J)
            Javg = 0.0
        elif self.DC_type == 'NoDC':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            Uavg = 0.0
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


class Dudarev_term(Hubbard_matrix_term):
    def __init__(self,
                 bset,
                 Hubbard_dict,
                 use_4index=False,
                 DC=True,
                 DC_type='FLL-ns',
                 DC_shift=False):
        use_4index = False
        super(Dudarev_term, self).__init__(bset, Hubbard_dict, DC, DC_type,
                                           DC_shift, use_4index)

    def _U_site(self, site, l, U, J, ids):
        # note that spin index is included in basis, but not in V_kanamori
        dim = len(ids) // 2

        U4ind = None
        U2, U2prime = U_matrix_dudarev(dim, U, J)
        return U4ind, U2, U2prime

    def _DC_site(self, U, J, U2, U2prime, rho):
        dim = rho.shape[0] / 2
        n_tot = np.trace(rho)
        if self.DC_type == 'FLL-s':
            n_up = np.sum(rho.diagonal()[::2])
            n_dn = np.sum(rho.diagonal()[1::2])
            Uavg = U
            Javg = J
        elif self.DC_type == 'FLL-ns':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            #Uavg = (U - (4.0 * dim - 4.0) / (2 * dim - 1) * J)
            Uavg = U
            Javg = J
        elif self.DC_type == 'NoDC':
            n_up = 0.5 * n_tot
            n_dn = 0.5 * n_tot
            Uavg = 0.0
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
