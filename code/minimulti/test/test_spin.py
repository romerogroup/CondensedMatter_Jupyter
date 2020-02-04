import unittest
import os
import numpy as np
from minimulti.spin.builder import cubic_3d_hamiltonian
from minimulti.constants import mu_B, gyromagnetic_ratio
from minimulti.spin.mover import SpinMover
from minimulti.spin.hist_file import read_hist
import matplotlib.pyplot as plt


class SpinTest(unittest.TestCase):
    def setUp(self):
        self.N = 8

    def test_exchange_Heff(self):
        ham = cubic_3d_hamiltonian(Jx=3e-21, Jy=3e-21, Jz=3e-21)
        sc_ham = ham.make_supercell(np.eye(3) * 1)
        S = np.array([[0, 0, 1.0]])
        Heff=np.zeros_like(S)
        sc_ham.get_effective_field(S=S, Heff=Heff )
        np.testing.assert_allclose(Heff , [[0.0, 0.0, 3.6e-20]])

    def test_cubic(self):
        ham = cubic_3d_hamiltonian(Jx=3e-21, Jy=3e-21, Jz=3e-21)
        sc_ham = ham.make_supercell(np.eye(3) * self.N)
        mover = SpinMover(hamiltonian=sc_ham, hist_fname='Spinhist.nc', write_hist=True)
        mover.set(
            time_step=0.005,
            temperature=200.0,
            total_time=1,
            save_all_spin=False)
        mover.run(write_step=10)

    def test_read_hist(self):
        result = read_hist('Spinhist.nc')
        plt.plot(result["time"], result["total_spin"] / self.N**3)
        plt.plot(result["time"],
                 np.linalg.norm(result["total_spin"] / self.N**3, axis=1))
        plt.show()

    def test_read_hist_onespin(self):
        result = read_hist('Spinhist.nc')
        plt.plot(result["time"], result["spin"][:, 0, :])
        plt.show()

    def tearDownClass():
        os.remove('Spinhist.nc')


if __name__ == '__main__':
    unittest.main()
