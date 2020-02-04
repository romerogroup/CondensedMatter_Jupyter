"""
This module fit the U and J values of a Hubbard model to a correlated DFT-tightbinding model (e.g. DFT+U/Hybrid functional so that the Gamma onsite-energies.
"""
import numpy as np
from scipy.optimize import minimize
from minimulti.electron.wannier90 import wannier_to_model
import matplotlib.pyplot as plt


class UJFitter():
    def __init__(self,
                 NM_model,
                 M_model,
                 nel,
                 elem,
                 kmesh=[6, 6, 6],
                 Utype='Kanamori',
                 dim=5):
        self.NM = NM_model
        self.M = M_model
        self.nel = nel
        self.elem = elem
        self.Utype = Utype
        self.dim = dim

        self.NM.set(nel=self.nel)
        self.M.set(nel=self.nel)

        self.kmesh = kmesh
        self.NM.set_kmesh(kmesh)
        self.M.set_kmesh(kmesh)

    def fit(self):
        # calculate rho from M_model
        self.M.scf_solve()
        rhoM = self.M._rho

        # gamma_ham
        print("gen ham NM")
        ham_NM = self.NM._gen_ham(k_input=[0, 0, 0])
        ham_M = self.M._gen_ham(k_input=[0, 0, 0])

        # on
        def target_func(x0):
            U, J = x0
            self.NM.set_Hubbard_U(
                Utype=self.Utype,
                Hubbard_dict={self.elem: {
                    'U': U,
                    "J": J
                }},
                dim=self.dim)
            dV, E_U = self.NM._U_term.V_and_E(rhoM)
            ham_NM_U = ham_NM + dV
            avg_diagNM = np.average(np.diag(ham_NM_U))
            avg_diagM = np.average(np.diag(ham_M))
            return np.sum((ham_NM_U - ham_M - np.diag(
                [avg_diagNM - avg_diagM] * self.NM._norb))**2)

        res = minimize(target_func, x0=[3.0, 1.0])
        if res.success:
            self.U, self.J = res.x
            return res.x
        else:
            raise Exception("U, J not found")

    def plot_bands(self,
                   supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                   kvectors=[[0, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0], [0, 0, 0],
                             [0.5, 0.5, 0.5]],
                   knames=['$\Gamma$', 'X', 'M', "$\Gamma$", 'R']):
        self.M.set(nel=self.nel)
        self.M.set_initial_spin([0.0] * 9 + [0, 1, 1, 0, 1])
        self.M.set_kmesh(self.kmesh)
        self.M.scf_solve()
        rho = self.M._rho
        self.M.plot_band(
            supercell_matrix=supercell_matrix,
            kvectors=kvectors,
            knames=knames)
        plt.savefig('M_model.png')
        plt.show()

        self.NM.set(nel=self.nel)
        self.NM.set_initial_spin(self.spinat)
        self.NM.set_kmesh(self.kmesh)
        self.NM.set_Hubbard_U(
            Utype=self.Utype,
            Hubbard_dict={self.elem: {
                'U': self.U,
                "J": self.J
            }},
            dim=self.dim)

        self.NM.set_density(rho)
        self.NM.scf_solve()
        self.NM.plot_band(
            supercell_matrix=supercell_matrix,
            kvectors=kvectors,
            knames=knames)
        plt.savefig('NM_model.png')
        plt.show()
