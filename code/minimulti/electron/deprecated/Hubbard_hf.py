#!/usr/bin/env python
"""
Hubard model HF solver.
"""
from myTB import mytb
from utils import fermi
from pythtb import tb_model
from hf import density_matrix_kspace, func_to_dict, HF_U
import numpy as np
from ase.calculators.interface import Calculator, DFTCalculator
from ase.dft.dos import DOS
from ase.dft.kpoints import monkhorst_pack
#from tetrahedronDos import tetrahedronDosClass
from occupations import Occupations
import copy
from Hubbard import LDAU_Durarev
from scipy.optimize import root
from mixing import Mixer,Pulay_Mixer
#from box.pulay import PulayMixer, DummyMixer
#from box.misc import AndersonMixer

class hubbard_hf(object):
    """
    hubbard model , hartree fock solver. Unrestriced. use spinor mode.
    """
    def __init__(self,
                 dim_k,
                 dim_r,
                 lat,
                 orb,
                 basis_set=None,
                 per=None,
                 nspin=2,
                 nel=None,
                 width=0.2,
                 verbose=True):
        #mytb.__init__(self,dim_k,dim_r,lat,orb,per=per,nspin=nspin,nel=nel,width=width,verbose=True)
        self._basis_set = basis_set
        self._dim_k = dim_k
        self._dim_r = dim_r
        self._norb = len(orb)
        self._tbmodel = tb_model(dim_k, dim_r, lat, orb, per=per, nspin=nspin)
        self._tbmodel0 = None
        self._eigenvals = None
        self._eigenvecs = None
        self._nspin = nspin
        self._nstate = self._norb * self._nspin
        # if fix_spin: _efermi is a tuple (ef_up,ef_dn)
        self._efermi = None
        self._occupations = None
        self._kpts = None
        # _kweight is a  array [w1,w2,....].
        self._kweights = None
        self._nel = nel
        self._U = None
        self._width = width
        self._verbose = verbose
        self._eps = 0.001
        self._onsite_en = np.zeros(len(orb))
        self._nbar = np.ndarray([len(orb), 2])
        #self._fix_spin=fix_spin
        # U function index r,p,q,s include spin index or not.
        self._U_spin_indexed = None
        self._old_occupations = None
        self._occupations = None
        # self._rho: density matrix _nstate*_nstate
        self._rho = None  #np.zeros((self._norb*self._nspin ,self._norb*self._nspin))
        self._total_energy = None
        self._niter = 0

    def set_kpoints(self, kpts):
        """
        set the kpoints to calculate. each kpoint can be a
        """
        if len(kpts[0]) == self._dim_k:
            self._kpts = kpts
            self._kweights = np.array(
                [1.0 / len(self._kpts)] * len(self._kpts))
        elif len(kpts[0]) == self._dim_k + 1:
            self._kpts = kpts[:, :-1]
            self._kweights = kpts[:, -1]

    def get_number_of_bands(self):
        """
        number of bands.
        """
        return self._tbmodel.get_num_orbitals()

    def get_eigenvalues(self, kpt=0, spin=None, refresh=False):
        """
        Ak_spin. Calculate the eigenvalues and eigen vectors. the eigenvalues are returned.
        """
        if self._eigenvals is None or refresh:
            self._eigenvals, self._eigenvecs = self._tbmodel.solve_all(
                k_list=self._kpts, eig_vectors=True)

        if spin is None:
            return self._eigenvals[:, kpt]
        else:
            ## seperate the spin up/ down
            ## project the evec to spin up/down basis
            ## TODO: if hamiltonian is diagonal in spin, get sepearate eval & evecs
            eval_up = []  #np.zeros(self._norb)
            eval_dn = []  #np.zeros(self._norb)
            for ib, eval in enumerate(self._eigenvals[:, kpt]):
                vec_up = self._eigenvecs[ib, kpt, :, 0]
                vec_dn = self._eigenvecs[ib, kpt, :, 1]
                #if np.abs(np.abs(vec_up)).sum()>np.abs(np.abs(vec_dn)).sum():
                if np.linalg.norm(vec_up) > np.linalg.norm(vec_dn):
                    eval_up.append(eval)
                else:
                    eval_dn.append(eval)
            eval_up = np.array(eval_up)
            eval_dn = np.array(eval_dn)

            if spin == 0 or spin == 'UP':
                return eval_up
            if spin == 1 or spin == 'DOWN':
                return eval_dn

    def get_fermi_level(self):
        """
        return the fermi level.
        """
        if self._efermi == None:
            print("Warning: Efermi not calculated yet. Using 0 instead.")
            return 0.0
        else:
            return self._efermi

    def get_bz_k_points(self):
        """
        kpoints
        """
        return self._kpts

    def get_ibz_k_points(self):
        """
        irreducible k points
        """
        raise NotImplementedError

    def get_k_point_weights(self):
        """
        weight of kpoints
        """
        return self._kweights

    def get_number_of_spins(self):
        """
        number of spins
        """
        return self._nspin

    def get_dos(self, width=0.15, method='gaussian', npts=501, spin=None):
        """
        density of states.

        :param width: smearing width
        :param method: 'gaussian'| 'tetra'
        :param npts: number of DOS energies.

        :returns:
          energies, dos. two ndarray.
        TODO: implement spin resolved DOS.
        """
        if method == 'tetra':
            dos = tetrahedronDosClass(self, width, npts=npts)
        else:
            dos = DOS(self, width, window=None, npts=npts)
        return dos.get_energies(), dos.get_dos(spin=spin)

    def get_projection(self):
        """
        project of eigenvalue to the basis.
         .. math::
            \\rho_i(\\epsilon)=\\sum_n<i|\\psi_n><\\psi_n|i>\\delta(\\epsilon-\\epsilon_n)
        """
        pass

    def set_density_matrix(self, order='random', scale=0.001):
        """
        set initial rho.

        :param nel: number of electrons
        :param order: 'random'|'PM'|'AFM'|'FM'
        """
        if not self._U_spin_indexed:
            if order == 'PM':
                self._rho = self.get_density_matrix()
            if order == 'FM':
                rho_up = np.eye(
                    self._norb) * self._nel / self._norb + np.random.normal(
                        loc=-scale, scale=scale, size=(self._norb, self._norb))
                rho_dn = np.eye(self._norb) * 0 + np.random.normal(
                    loc=scale, scale=scale, size=(self._norb, self._norb))
                self._rho = (rho_up, rho_dn)
            if order == 'random':
                self._rho = self.get_density_matrix()
                rho_up = self._rho[0] + np.random.normal(
                    loc=0, scale=scale, size=(self._norb, self._norb))
                rho_dn = self._rho[1] + np.random.normal(
                    loc=0, scale=scale, size=(self._norb, self._norb))
                self._rho = (rho_up, rho_dn)

    def get_density_matrix(self):
        """
        calculate the density matrix
        """
        if self._verbose:
            print("Calc density matrix")

        #1 if occupations are not calculated, calculate it.
        if self._occupations is None:
            self.get_occupations(self._nel, self._width, refresh=True)

        if self._U_spin_indexed:
            self._rho = density_matrix_kspace(
                self._eigenvecs,
                self._occupations,
                self._kweights,
                split_spin=False)
        else:
            # rho-> (rho_up,rho_dn)
            self._rho = density_matrix_kspace(
                self._eigenvecs,
                self._occupations,
                self._kweights,
                split_spin=True)
        if self._verbose:
            print("End calc density matrix")
        return self._rho

    def get_occupations(self, nel, width=None, refresh=False):
        """
        calculate occupations of each eigenvalue.
        the the shape of the occupation is the same as self._eigenvals.
        [eig_k1,eigk2,...], each eig_k is a column with the length=nbands.

        if nspin=2 and fix_spin, there are two fermi energies. NOTE: this conflicts with the DOS caluculation. FIXME.

        :param nel: number of electrons. if fix_spin, the nel is a tuple of (nel_up,nel_dn)

        :Returns:
          self._occupations (np.ndarray) index:[band,kpt,orb,spin] if nspin==2 else [band,kpt,orb] same as eigenvec
        """
        if self._verbose:
            print("Calc occupation")

        self._nel = nel
        self.get_eigenvalues(refresh=refresh)
        #print self._kweights
        #print self._eigenvals

        occ = Occupations(nel, self._width, self._kweights, nspin=self._nspin)
        self._occupations = occ.occupy(self._eigenvals)
        self._efermi = occ.get_mu()
        if self._verbose:
            print("End Calc occupation")

    def get_orbital_magnetic_moments(self):
        if self._U_spin_indexed:
            rho_ii = np.real(self._rho.diagonal())
            return rho_ii[::2] - rho_ii[1::2]
        else:
            return np.real(self._rho[0].diagonal() - self._rho[1].diagonal())

    def get_orbital_occupations(self):
        """
        Qui!
        """
        if self._U_spin_indexed:
            return np.real(self._rho.diagonal())
        else:
            return np.real(
                np.vstack((self._rho[0].diagonal(), self._rho[1].diagonal()))
                .transpose().flatten())

    def set_initial_nbar(self):
        """
        TODO: also density matrix?
        """
        pass

    def set_U(self, U):
        """
        set the U value of elements.

        :param U: dict or number. U is a dict, eg. {'Fe':2} or a number. If U is a number, all the elements have the same U value.
        """
        self._U = U

    def set_U_func(self, U, spin_indexed=False):
        """
        TODO: choice1, U_func(p,q,r,s)
        choice2, U site
        """
        self._U_spin_indexed = spin_indexed
        if spin_indexed:
            nbasis = self._norb
        else:
            nbasis = self._norb * 2
        if not isinstance(U, dict):
            self._U = func_to_dict(U, nbasis)
        else:
            self._U = U

    def save_H0(self):
        """
        save the current tight bindig model so that the H0 is saved.
        """
        self._tbmodel0 = copy.copy(self._tbmodel)

    def set_magnetic_field(self, M):
        """
        set the magnetic field M.
        """
        if not hasattr(M, '__iter__'):
            M = [M] * self._norb
        elif len(M) != self._norb:
            raise Exception(
                "M should be a number, or a list of the size self._norb")
        for i in range(self._norb):
            self._tbmodel.set_onsite(
                np.array([M[i], 0, 0, -M[i]]).reshape((2, 2)), i, mode='add')

    def set_Heff(self, hubbard_type=2):
        """
        Heff=H0+HU

        :param hubbard_type: (optional.) 1: Lishestein. 2: Durarev
        """
        if self._verbose:
            print("set Heff")

        #1 If rho is not calculated, calculate rho
        if self._rho is None:
            self.get_density_matrix()

        if hubbard_type == 1:
            if self._U_spin_indexed:
                Hu = HF_U(
                    self._U,
                    self._nstate,
                    self._rho,
                    spinor=True,
                    restricted=False,
                    spin_indexed=self._U_spin_indexed)
                raise NotImplementedError(
                    "spin_indexed Currently not implemented")
            else:
                Hu = HF_U(
                    self._U,
                    self._norb,
                    self._rho,
                    spinor=True,
                    restricted=False,
                    spin_indexed=self._U_spin_indexed)
                Hu_up, Hu_dn = Hu
        elif hubbard_type == 2:
            Hu = LDAU_Durarev(
                self._U,
                self._basis_set,
                self._rho,
                spin_indexed=self._U_spin_indexed)

            if self._U_spin_indexed:
                raise NotImplementedError(
                    "spin_indexed Currently not implemented")
            else:
                Hu_up, Hu_dn = Hu

        self._tbmodel = copy.deepcopy(self._tbmodel0)
        for i in range(self._norb):
            for j in range(self._norb):
                if i != j:
                    self._tbmodel.set_hop(
                        np.array([Hu_up[i, j], 0, 0, Hu_dn[i, j]]).reshape(
                            (2, 2)),
                        i,
                        j,
                        ind_R=[0] * self._dim_r,
                        mode='add',
                        allow_conjugate_pair=True)
                else:
                    #print Hu_up[i,i],Hu_dn[i,i]
                    self._tbmodel.set_onsite(
                        np.array([Hu_up[i, j], 0, 0, Hu_dn[i, j]]).reshape(
                            (2, 2)),
                        i,
                        mode='add')
        if self._verbose:
            print("End set Heff")

    def solve_all(self, kpts=None):
        """
        solve the eigen value problem of self._tbmodel
        """
        if self._verbose:
            print("solve eig , iter: %s" % self._niter)
            self._niter += 1

        if kpts is not None:
            self._kpts = kpts
        self._eigenvals, self._eigenvecs = self._tbmodel.solve_all(
            k_list=self._kpts, eig_vectors=True)
        if self._verbose:
            print("end solve eig")

        return self._eigenvals

    def get_energy(self):
        """
        calculate the total energy, not implemented yet.
        """
        if self._eigenvals is None:
            self.get_eigenvalues()
        if self._efermi is None:
            self.get_occupations(self._nel, width=self._width)
            mu = self.get_fermi_level()
        e = 0

        #mu=self._efermi
        for ik, kpt in enumerate(self._kpts):
            eig_k = self._eigenvals[:, ik]
            e += np.inner(eig_k,
                          fermi(eig_k, mu, self._width,
                                nspin=self._nspin)) * self._kweights[ik]
        self._total_energy = e
        # sum of the occupation
        return self._total_energy
        #raise NotImplementedError("total energy not yet implemented")

    def scf_solve_broyden(self,
                          max_step=200,
                          e_tol=1e-5,
                          rho_tol=1e-5,
                          method='broyden1'):
        """
        solve the problem with broyden's method.
        """
        #1 initialize
        if self._rho is None:
            self.get_density_matrix()

        rho_guess = np.hstack((self._rho[0].flatten(), self._rho[1].flatten()))

        def target_func(flat_rho):
            self.set_Heff()
            self.solve_all()
            self.get_occupations(self._nel, self._width)
            self.get_density_matrix()
            nr = np.hstack((self._rho[0].flatten(), self._rho[1].flatten()))
            return nr  #np.linalg.norm(nr-flat_rho)

        def callback_func(rho, f):
            l = len(rho) / 2
            self._rho = (rho[:l].reshape((self._norb, self._norb)),
                         rho[l:].reshape((self._norb, self._norb)))

        root(
            target_func,
            rho_guess,
            tol=rho_tol,
            callback=callback_func,
            method='Krylov',
            options={'iter': max_step,
                     'verbose': True})

    def scf_solve(self, max_step=200, e_tol=1e-5, rho_tol=1e-5):
        """
        solve the problem self consistently. The loop will end if max steps is reached or both the energy and rho tolenrance is reached.

        :param max_step: max step to run.
        :param e_tol: total energy convergence tolerance.
        :param rho_tol: density matrix convergence tolerance.

        """
        #1 initialize
        if self._rho is None:
            self.get_density_matrix()

        #2 iterative solve
        last_E = np.inf
        last_rho = self._rho

        mixer = PulayMixer(
            mixing_constant=0.35, convergence=rho_tol, chop=None, memory=6)

        for i in range(max_step):
            print("Iter %s: " % i)
            self.set_Heff()
            self.solve_all()
            self.get_occupations(self._nel, self._width)
            self.get_density_matrix()

            self._total_energy = 2
            deltaE = self._total_energy - last_E
            last_E = self._total_energy

            delta_rho = np.abs(np.asarray(last_rho) -
                               np.asarray(self._rho)).max()
            #beta=0.6
            #self._rho=np.asarray(last_rho)*(1-beta) + np.asarray(self._rho)*beta
            last_rho_in = np.array(last_rho)
            rho_in = np.array(self._rho)
            shape = rho_in.shape
            conv, rho_out = self._rho = mixer(last_rho_in.flatten(),
                                              rho_in.flatten())
            self._rho = rho_out.reshape(shape)
            print self._rho

            last_rho = self._rho
            #self._rho=self._rho
            print("delta_rho= %s" % delta_rho)
            print(abs(self._rho[0].diagonal()))
            print(abs(self._rho[1].diagonal()))
            #if abs(deltaE)<e_tol and abs(delta_rho)<rho_tol:
            if conv:
                print("Convergence reached. Stop Iteration.")
                break

        return self._total_energy
