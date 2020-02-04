"""
Define TB / TB+U model.
"""
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.linalg import eigh
from scipy.interpolate import interp1d
from scipy.sparse import coo_matrix
import logging
import pickle
import copy
from collections import defaultdict

from ase.dft.kpoints import monkhorst_pack
from ase.dft.dos import DOS
from ase.atoms import Atoms
from ase.dft.kpoints import bandpath
from minimulti.electron.pdos import PDOS, WDOS
from minimulti.electron.pythtb import tb_model
from minimulti.electron.occupations import Occupations, GaussOccupations
from minimulti.electron.density import density_matrix_kspace
from minimulti.electron.box.pulay import PulayMixer
from minimulti.electron.utils import fermi
from minimulti.electron.basis2 import atoms_to_basis
from minimulti.electron.Hubbard import (
    #Hubbard_U_Dudarev, Hubbard_U_Liechtenstein, Hubbard_U_Kanamori,
    Kanamori_term,
    Lichtenstein_term,
    Dudarev_term,
    Lichtenstein2_term)
#from minimulti.electron.Hubbard_old import Hubbard_U_Liechtenstein
from minimulti.electron.constraint import SiteChargeConstraint
from minimulti.utils.supercell import SupercellMaker
import matplotlib.pyplot as plt
from minimulti.electron.plot import plot_band_weight
from minimulti.electron.pythtb import _nicefy_eig
from minimulti.electron.COHP import COHP
from minimulti.unfolding.unfolder import Unfolder

from minimulti.electron.susceptibility import calc_chi0_list


class mytbmodel(tb_model):
    # This is fully compatible with tb_model, only a few functions
    # are overided so it is faster
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        super(mytbmodel, self).__init__(
            dim_k=dim_k, dim_r=dim_r, lat=np.array(lat), orb=orb, per=per, nspin=nspin)

    def get_onsite(self):
        return self._site_energies

    def get_hoppings(self):
        return self._hoppings


class etb_model(mytbmodel):
    def __init__(self, dim_k=3, dim_r=3, lat=None, orb=None, per=None,
                 nspin=1):
        # not nspin is set to 1, spin is included in orb.
        mytbmodel.__init__(
            self, dim_k=dim_k, dim_r=dim_r, lat=np.array(lat), orb=orb, per=per, nspin=1)

        self._efermi = None
        self._nel = None
        self._smearing = None
        self._sigma = None
        self._eigenvals = None
        self._eigenvecs = None

        self._spin_diag = False

        self._occupations = None
        self._kpts = None
        self._kweights = None
        self._verbose = False
        self._eps = 0.000001

        self._iter = 0

        self._width = 0.1
        self._smearing = 'gaussian'

        if orb is not None:
            self._norb = len(orb)
        self._actual_nspin = nspin

        # density matrix
        self._rho = np.zeros((self._norb, self._norb))
        self._has_rho = False
        # has Effective U term or not. If not, a tight binding model.
        self._has_ham_U = False
        # Hamiltonian in k space. [nkpts, norb, norb]
        self._ham0_k = None
        # effective potential of Hubbard Term. Note that H_U is not k-dependent.
        # [norb, norb]
        self._ham_U = np.zeros((self._norb, self._norb), dtype=float)
        self._U_energy = 0.0
        self._Uband_energy = 0.0
        self._int_energy = 0.0
        self._DC_energy = 0.0
        self._spinat = np.zeros(self._norb // 2)
        self._nel = 0
        self._tol_energy = 1e-10
        self._tol_rho = 1e-10
        self._mixing = 0.4
        self._sigma = 0.15

        self._diag_only = False

        self._TBnoU_energy = 0.0

        self._results = {}
        self._U_term = None

        self._zeeman_field = None

        self._constraint_term = None
        self._constraint_V = None
        self._constraint_E = 0.0

        self._converged=False

    def to_spin_polarized(self):
        orb = np.repeat(self._orb, 2)
        model = etb_model(
            self,
            dim_k=self._dim_k,
            dim_r=_self._dim_r,
            lat=copy.copy(self._lat),
            orb=orb,
            per=None,
            nspin=1)

    def set(
            self,
            nel=None,
            smearing='gaussian',
            sigma=0.1,
            mixing=0.3,
            nmix=5,
            tol_rho=1e-9,
            tol_energy=1e-9,
            workdir='./',
            prefix='prefix',
            save_density=False,
            load_density=False,
            save_wfk=False,
            load_wfk=False,
    ):
        """
        set parameters
        """
        self._nel = nel
        self._smearing = smearing
        self._width = sigma
        self._mixing = mixing
        self._nmix = nmix
        self._tol_rho = tol_rho
        self._tol_energy = tol_energy
        self._workdir = workdir

        if not os.path.exists(self._workdir):
            os.makedirs(self._workdir)
        self._prefix = prefix
        self._save_density = save_density
        self._load_density = load_density
        self._density_file = os.path.join(workdir, "%s_den.npy" % prefix)
        self._prepare_logging_file()

    def _prepare_logging_file(self):
        self.log_file = os.path.join(self._workdir, 'log')
        logging.basicConfig(filename=self.log_file, level='DEBUG')

    def set_density(self, rho):
        self._rho = rho
        self._has_rho = True

    def save_density(self, fname=None, rhok_fname=None):
        if rhok_fname is None:
            if fname is None:
                fname = self._density_file
            np.save(fname, self._rho, allow_pickle=False)
        else:
            nk = len(self._kpts)
            rhok = np.zeros((nk, self._norb, self._norb), dtype='complex')
            self.get_density_matrix(rhok=rhok)
            np.save(fname, self._rho, allow_pickle=False)
            np.save(rhok_fname, rhok, allow_pickle=False)

    def load_density(self, fname=None):
        if fname is None:
            fname = self._density_file
        self._rho = np.load(fname, allow_pickle=False)
        self._has_rho = True

    def set_kpoints(self, kpts, kweights=None):
        """
        set the kpoints to calculate. each kpoint can be a
        """
        self._kpts = kpts
        if kweights is None:
            self._kweights = np.array(
                [1.0 / len(self._kpts)] * len(self._kpts))
        else:
            self._kweights = kweights

    def set_kmesh(self, kmesh):
        """
        set kpoints by giving a mesh.
        """
        self.set_kpoints(monkhorst_pack(kmesh))

    def get_cell(self):
        return self._lat

    def get_number_of_bands(self):
        """
        number of bands. count spin up and spin down  as 2.
        """
        return self._norb

    def get_number_of_spins(self):
        """
        number of spins.
        """
        return self._actual_nspin

    def get_fermi_level(self):
        """
        get fermi energy.
        """
        if self._efermi == None:
            return 0.0
        else:
            return self._efermi

    def get_bz_k_points(self):
        return self._kpts

    def get_ibz_k_points(self):
        raise NotImplementedError

    def get_k_point_weights(self):
        return self._kweights

    def get_all_eigenvalues(self, spin=None, refresh=False):
        if self._eigenvals is None or refresh:
            self._eigenvals, self._eigenvecs = self.solve_all(
                k_list=self._kpts, eig_vectors=True)
        if spin is None:
            return self._eigenvals

    def split_eigenvalue_spin(self, kpts, eigenvalues, eigenvectors):
        """
        return eigen values for spin up &down
        for each spin,
        indexes are : iband, ikpt.
        """
        #eval_up=eigenvalues[::2,:]
        #eval_dn=eigenvalues[1::2,:]
        evals_up = []
        evals_dn = []
        for ikpt, kpt in enumerate(kpts):
            eval_up = []  #np.zeros(self._norb)
            eval_dn = []  #np.zeros(self._norb)
            evs = enumerate(eigenvalues[:, ikpt])
            ds = []
            for ib, evalue in evs:
                vec_up = eigenvectors[ib, ikpt, ::2]
                vec_dn = eigenvectors[ib, ikpt, 1::2]
                #if np.abs(np.abs(vec_up)).sum()>np.abs(np.abs(vec_dn)).sum():
                d = np.linalg.norm(vec_up) - np.linalg.norm(vec_dn)
                ds.append(d)
                if d > 1e-2:
                    eval_up.append(evalue)
                elif abs(d) <= 1e-2:
                    nib, nevalue = next(evs)
                    if abs(evalue - nevalue) < 1e-3:
                        eval_up.append(evalue)
                        eval_dn.append(nevalue)
                    else:
                        print("something wrong")
                else:
                    eval_dn.append(evalue)
            evals_up.append(eval_up)
            evals_dn.append(eval_dn)
        if len(evals_up) != len(evals_dn):
            evals_up = eigenvalues[::2, :]
            evals_dn = eigenvalues[1::2, :]
        return np.array(evals_up).T, np.array(evals_dn).T

    def get_eigenvalues(self, kpt=0, spin=None, refresh=False):
        """
        Ak_spin. Calculate the eigenvalues and eigen vectors. the eigenvalues are returned.
        self._eigenvals are returned.
        """
        if self._eigenvals is None or refresh:
            self._eigenvals, self._eigenvecs = self.solve_all(
                k_list=self._kpts, eig_vectors=True)
        if spin is None or self._actual_nspin == 1:
            return self._eigenvals[:, kpt]
        else:
            ## seperate the spin up/ down
            ## project the evec to spin up/down basis
            eval_up = []  #np.zeros(self._norb)
            eval_dn = []  #np.zeros(self._norb)

            evs = enumerate(self._eigenvals[:, kpt])
            for ib, evalue in evs:
                vec_up = self._eigenvecs[ib, kpt, ::2]
                vec_dn = self._eigenvecs[ib, kpt, 1::2]
                d = np.linalg.norm(vec_up) - np.linalg.norm(vec_dn)
                if d > 1e-5:
                    eval_up.append(evalue)
                elif abs(d) <= 1e-5:
                    nib, nevalue = next(evs)
                    if abs(evalue - nevalue) < 1e-5:
                        eval_up.append(evalue)
                        eval_dn.append(evalue)
                else:
                    eval_dn.append(evalue)
            eval_up = np.array(eval_up)
            eval_dn = np.array(eval_dn)

            if len(eval_up) != len(eval_dn):
                eval_up = self._eigenvals[::2, kpt]
                eval_dn = self._eigenvals[1::2, kpt]

            if spin == 0 or spin == 'UP':
                #return self._eigenvals[::2,kpt]
                return eval_up
            if spin == 1 or spin == 'DOWN':
                return eval_dn
                #return self._eigenvals[1::2,kpt]

    def get_dos(self, width=None, smearing='gaussian', npts=501, spin=None):
        """
        density of states.
        :param width: smearing width
        :param method: 'gaussian'| 'tetra'
        :param npts: number of DOS energies.
        :returns:
          energies, dos. two ndarray.
        TODO: implement spin resolved DOS.
        """
        if width is None:
            width = self._width
        if smearing == 'tetra':
            raise NotImplementedError('tetrahedron DOS not implemented')
            #dos = tetrahedronDosClass(self, width, npts=npts)
        else:
            dos = DOS(self, width, window=None, npts=npts)
        return (dos.get_energies(), dos.get_dos(spin=spin))

    def get_pdos(self, emin, emax, nedos, sigma=None, fname=None):
        if sigma is None:
            sigma = self._sigma

        self.pdos = PDOS(
            kpts=self._kpts,
            kweights=self._kweights,
            sigma=sigma,
            evals=self._eigenvals,
            evecs=self._eigenvecs,
            emin=emin,
            emax=emax,
            nedos=nedos)
        if fname is not None:
            np.savetxt(
                fname,
                np.vstack((self.pdos.get_energy() - self._efermi,
                           self.pdos.get_pdos())).T,
                header='efermi=%s' % self._efermi)
        return self.pdos

    #@profile
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

        occ = Occupations(
            nel, self._width, self._kweights, nspin=self._actual_nspin)
        self._occupations = occ.occupy(self._eigenvals)
        self._efermi = occ.get_mu()
        if self._verbose:
            print("End Calc occupation")

    def get_density_matrix(self, diag_only=None, rhok=None):
        """
        calculate the density matrix
        """
        if self._verbose:
            print("Calc density matrix")

        if diag_only is None:
            diag_only = self._diag_only

        #1 if occupations are not calculated, calculate it.
        if self._occupations is None:
            self.get_occupations(self._nel, self._width, refresh=True)

        #if self._U_spin_indexed:
        self._rho = density_matrix_kspace(
            self._eigenvecs,
            self._occupations,
            self._kweights,
            split_spin=False,
            diag_only=diag_only,
            rho_k=rhok)
        if self._verbose:
            print("End calc density matrix")
        return self._rho

    def get_orbital_magnetic_moments(self):
        """
        get magnetic moments of orbitals.
        """
        rho_ii = np.real(self._rho.diagonal())
        return rho_ii[::2] - rho_ii[1::2]

    def get_orbital_occupations(self):
        """
        get occupations of orbitals
        """
        return np.real(self._rho.diagonal())

    def set_density_matrix(self, rho, add=False):
        """
        set density matrix.
        """
        if not add:
            self._rho = rho
        if add:
            self._rho += rho

    def set_initial_spin(self, spinat):
        """
        params:
            spinat: a list of initial spin, length should be the same as number of orbitals(same orbital/different spin count as 1).
        set initial spin polarization, per orbital.
        The rho is aranged like (orb1, up), (orb1, dn), (orb2, up)...
        So this is done by change the diagonal of non-spin polarized rho.
        """
        if len(spinat) != self._norb // 2:
            raise ValueError(
                "length of spinat should be the same as number of orbitals")
        self._spinat = spinat

    def set_zeeman_field(self, zeeman_field):
        self._zeeman_field = np.array(zeeman_field, dtype=float)

    def set_spinat_rho(self):
        drho = np.zeros(self._norb)
        drho[::2] = np.array(self._spinat) / 2.0
        drho[1::2] = -np.array(self._spinat) / 2.0
        self._rho += np.diag(drho)

    def set_U_term(self, U_term):
        """
        set Hubbard U term. The U term should have two functions:
        get_potential(rho)
        and get_energy(rho).
        """
        self._U_term = U_term
        self._has_ham_U = True

    def set_constraint(self, V, charge_dict):
        term = SiteChargeConstraint(
            bset=self.bset, V=V, charge_dict=charge_dict)
        self._constraint_term = term

    def calc_Heff(self):
        """
        calculate effective H_U.
        """
        if not self._has_rho:
            self.get_density_matrix()
        self._ham_U, self._U_energy = self._U_term.V_and_E(self._rho)

        self._Uband_energy = -np.sum(self._ham_U * self._rho)
        #self._Uband_energy = self._U_term.E_Uband
        self._int_energy = self._U_term.E_int
        self._DC_energy = self._U_term.E_DC

        if self._constraint_term is not None:
            self._constraint_V, self._constraint_E = self._constraint_term.V_and_E(
                self._rho)

    def calc_energy_from_rho(self, rhok, rho):
        #band_energy
        tb_energy = 0.0
        for hami, rhoki in zip(self._ham0_k, rhok):
            tb_energy += np.sum(hami * rhoki)
        ham_U, U_energy = self._U_term.V_and_E(rho)
        #U_band_energy = -np.sum(ham_U * rho)
        int_energy = self._U_term.E_int
        DC_energy = self._U_term.E_DC
        return tb_energy + int_energy + DC_energy

    def _gen_ham0_k(self, convention=2):
        """
        generate hamiltonian for k point.
        """
        self._ham0_k = np.zeros(
            shape=(len(self._kpts), self._norb, self._norb), dtype=complex)
        for ik, k in enumerate(self._kpts):
            self._ham0_k[ik, :, :] = self._gen_ham(
                k_input=k, convention=convention)

    def _sol_ham(self, ham, eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        if self._nspin == 1:
            ham_use = ham
        elif self._nspin == 2:
            ham_use = ham.reshape((2 * self._norb, 2 * self._norb))
        # check that matrix is hermitian
        if np.max(ham_use - ham_use.T.conj()) > 1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        #solve matrix
        if eig_vectors == False:  # only find eigenvalues
            #eval=np.linalg.eigvalsh(ham_use)
            eval_up = np.linalg.eigvalsh(ham_use[::2, ::2])
            eval_dn = np.linalg.eigvalsh(ham_use[1::2, 1::2])
            eval = np.zeros(self._norb)
            eval[::2] = eval_up
            eval[1::2] = eval_dn
            # sort eigenvalues and convert to real numbers
            eval = _nicefy_eig(eval)
            return np.array(eval, dtype=float)
        else:  # find eigenvalues and eigenvectors
            #(eval,eig)=eigh(ham_use, turbo=True)
            #(eval,eig)=np.linalg.eigh(ham_use )
            paral_spin = False
            if paral_spin:
                with ProcessPoolExecutor(max_workers=2) as executor:
                    future1 = executor.submit(eigh, ham_use[::2, ::2])
                    future2 = executor.submit(eigh, ham_use[1::2, 1::2])
                eval_up, eig_up = future1.result(
                )  #np.linalg.eigh(ham_use[::2,::2])
                eval_dn, eig_dn = future2.result(
                )  #np.linalg.eigh(ham_use[1::2,1::2])
            else:
                eval_up, eig_up = eigh(ham_use[::2, ::2])
                eval_dn, eig_dn = eigh(ham_use[1::2, 1::2])

            eval = np.zeros(self._norb)
            eval[::2] = eval_up
            eval[1::2] = eval_dn
            eig = np.zeros((self._norb, self._norb), dtype=complex)
            eig[::2, ::2] = eig_up
            eig[1::2, 1::2] = eig_dn

            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig = eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval, eig) = _nicefy_eig(eval, eig)
            # reshape eigenvectors if doing a spinfull calculation
            if self._nspin == 2:
                eig = eig.reshape((self._nsta, self._norb, 2))
            return (eval, eig)

    def solve_all(self,
                  k_list=None,
                  eig_vectors=True,
                  total_ham=False,
                  convention=2,
                  zeeman=True):
        """
        If kpts is None, kpts is default, there is no need to refresh ham0_k, else calculate ham for each k point and solve. note _ham0_k is not refreshed.
        """
        if k_list is None:
            has_ham0 = True
            k_list = self._kpts
            if self._ham0_k is None:
                self._gen_ham0_k()
        else:
            has_ham0 = False

        ham_total = []

        nkp = len(k_list)  # number of k points
        # first initialize matrices for all return data
        #    indices are [band,kpoint]
        ret_eval = np.zeros((self._nsta, nkp), dtype=float)
        if self._nspin == 1:
            ret_evec = np.zeros((self._nsta, nkp, self._norb), dtype=complex)
        elif self._nspin == 2:
            ret_evec = np.zeros(
                (self._nsta, nkp, self._norb, 2), dtype=complex)
        for i, kpt in enumerate(k_list):
            if has_ham0:
                ham0 = self._ham0_k[i, :, :]
            else:
                ham0 = self._gen_ham(k_input=kpt, convention=convention)
            h_total_k = ham0 + self._ham_U
            if self._constraint_V is not None:
                h_total_k += self._constraint_V
            if zeeman and (self._zeeman_field is not None):
                t = np.array([self._zeeman_field,
                              -self._zeeman_field]).T.flatten()
                np.fill_diagonal(h_total_k, np.diag(h_total_k) + t)
            if total_ham:
                ham_total.append(h_total_k)
            if not eig_vectors:
                evalue = self._sol_ham(h_total_k, eig_vectors=eig_vectors)
                ret_eval[:, i] = evalue[:]
            else:
                (evalue, evec) = self._sol_ham(
                    h_total_k, eig_vectors=eig_vectors)
                ret_eval[:, i] = evalue[:]
                if self._nspin == 1:
                    ret_evec[:, i, :] = evec[:, :]
                elif self._nspin == 2:
                    ret_evec[:, i, :, :] = evec[:, :, :]
        # return stuff
        if not eig_vectors:
            # indices of eval are [band,kpoint]
            self._eigenvals = ret_eval
            return ret_eval
        else:
            if not total_ham:
                # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
                self._eigenvals = ret_eval
                self._eigenvecs = ret_evec
                return (ret_eval, ret_evec)
            else:
                self._eigenvals = ret_eval
                self._eigenvecs = ret_evec
                return (ret_eval, ret_evec, np.array(ham_total))

    def calc_COHP(self, iblock=None, jblock=None, save=False):
        evals, evecs, ham = self.solve_all(eig_vectors=True, total_ham=True)
        cohp = COHP(
            tbmodel=self,
            kpts=self._kpts,
            ham=ham,
            kweights=self._kweights,
            evals=evals,
            evecs=evecs)
        cohp.calc_cohp_allk(iblock=iblock, jblock=jblock)
        if save:
            cohp.save(fname=os.path.join(self._workdir, 'COHP.pickle'))
        return cohp

    def get_ham0_energy(self):
        e0 = 0.0
        for ik, kpt in enumerate(self._kpts):
            evecs = self._eigenvecs[:, ik, :].T  # Note ib, ik , iorb.
            values = np.diag(evecs.T.conj().dot(self._ham0_k[ik]).dot(evecs))
            e0 += self._kweights[ik] * np.sum(
                values * self._occupations[:, ik])
        return e0

    def get_rho(self):
        return self._rho

    def get_charges(self):
        return self._rho.diagonal()

    def get_magnetic_moments(self):
        diag_rho = self._rho.diagonal()
        return diag_rho[::2] - diag_rho[1::2]

    def get_band_energy(self):
        if self._eigenvals is None:
            self.get_eigenvalues()
        if self._efermi is None:
            self.get_occupations(self._nel, width=self._width)
        mu = self.get_fermi_level()
        e = 0

        #mu=self._efermi
        for ik, kpt in enumerate(self._kpts):
            eig_k = np.real(self._eigenvals[:, ik])
            e += np.inner(
                eig_k, fermi(eig_k, mu, self._width,
                             nspin=self._actual_nspin)) * self._kweights[ik]
        self._band_energy = e
        # sum of the occupation
        return self._band_energy

    def get_U_energy(self):
        """
        get energy from U term.
        """
        return np.real(self._U_energy)

    def get_energy(self):
        self._energy = self.get_band_energy() + np.real(
            self._U_energy) + self._constraint_E
        return self._energy

    def print_energy_terms(self):
        print("Total Energy: %12.5e" % self._energy)
        # print(
        #     "Total Energy = Band Energy(H_eff rho) - Double Counting (DC) Energy (1/2 H_U rho)"
        # )
        # print("Band Energy: %12.5e" % self._band_energy)
        # print("DC Energy: %12.5e" % self._U_energy)
        # print(
        #     "Total Energy = Single electron Energy(H_0*rho) + Hubbard Energy(1/2 H_U rho)"
        # )
        # print("Single electron Energy: %12.5e" %
        #       (self._band_energy - 2 * self._U_energy))
        # print("DC Energy: %12.5e" % self._U_energy)

    def randomize_rho(self):
        r = np.random.rand(self._norb)
        self._rho = self._rho + np.diag((r - np.sum(r)) * 12.0)

    def scf_solve(
            self,
            max_step=500,
            tol_energy=None,
            tol_rho=None,
            print_iter_info=True,
            convention=2,
    ):
        if self._U_term is None:
            self.scf_solve_noU(convention=convention)
        else:
            self.scf_solve_U(
                max_step=max_step,
                tol_energy=tol_energy,
                tol_rho=tol_rho,
                print_iter_info=print_iter_info,
                convention=convention)

    def scf_solve_noU(self, convention=2):
        self.solve_all(eig_vectors=True, convention=convention)
        self.get_occupations(self._nel, self._width, refresh=False)
        self.get_density_matrix()
        self._TBnoU_energy = self.get_energy()
        #self.randomize_rho()
        self._has_rho = True
        self._total_energy = self._TBnoU_energy
        if self._save_density:
            self.save_density()

    def scf_solve_U(
            self,
            max_step=500,
            tol_energy=None,
            tol_rho=None,
            print_iter_info=True,
            convention=2,
    ):
        """
        solve the problem self consistently. The loop will end if max steps is reached or both the energy and rho tolenrance is reached.

        :param max_step: max step to run.
        :param e_tol: total energy convergence tolerance.
        :param rho_tol: density matrix convergence tolerance.
        """
        if tol_energy is None:
            tol_energy = self._tol_energy
        if tol_rho is None:
            tol_rho = self._tol_rho

        #1 initialize

        # initialize by reading density file
        if self._load_density and os.path.exists(self._density_file):
            print("Load density from file: %s" % self._density_file)
            self._rho = np.load(self._density_file)
            self._has_rho = True
        elif not self._has_rho:
            self.solve_all(
                eig_vectors=True, convention=convention, zeeman=False)
            self.get_occupations(self._nel, self._width, refresh=False)
            self.get_density_matrix()
            self._TBnoU_energy = self.get_energy()
            #self.randomize_rho()
            self.set_spinat_rho()
            self._has_rho = True
        else:
            print("Using input rho")
            tmp = copy.copy(self._rho)
            self.solve_all(
                eig_vectors=True, convention=convention, zeeman=False)
            self.get_occupations(self._nel, self._width, refresh=False)
            self.get_density_matrix()
            self._TBnoU_energy = self.get_energy()
            self._rho = tmp
            self._has_rho = True

        #2 iterative solve
        last_E = np.inf
        last_rho = self._rho

        mixer = PulayMixer(
            mixing_constant=self._mixing,
            convergence=tol_rho,
            chop=None,
            memory=self._nmix)

        if print_iter_info:
            print("Begining Scf calculations")
            print("=====================================================")
            print("Iteration     Energy       Delta_E     Delta_rho")
        for i in range(max_step):
            self._iter = i
            self.calc_Heff()
            self.solve_all(eig_vectors=True, zeeman=True)
            self.get_occupations(self._nel, self._width, refresh=False)
            if self._iter < 3:
                self.get_density_matrix(diag_only=True)
            else:
                self.get_density_matrix()

            self._total_energy = self.get_energy()
            deltaE = self._total_energy - last_E
            last_E = self._total_energy

            delta_rho = np.abs(np.asarray(last_rho) -
                               np.asarray(self._rho)).max()
            #beta=0.6
            #self._rho=np.asarray(last_rho)*(1-beta) + np.asarray(self._rho)*beta
            last_rho_in = np.array(last_rho)
            rho_in = np.array(self._rho)
            shape = rho_in.shape
            conv_rho, rho_out = mixer(last_rho_in.flatten(), rho_in.flatten())
            self._rho = rho_out.reshape(shape)

            if print_iter_info:
                print("Iter %4d: %12.5e %12.5e %12.5e" %
                      (i, np.real(self._total_energy), np.real(deltaE),
                       np.real(delta_rho)))
            #print("rho: %s"%self._rho)

            last_rho = self._rho
            conv_e = False
            if abs(deltaE) < tol_energy:
                conv_e = True
            if conv_rho:  # or conv_e:
                self._converged=True
                if print_iter_info:
                    print("Convergence reached. Stop Iteration.")
                break
        if print_iter_info:
            self.print_energy_terms()
            print("Writting density files to %s." % self._density_file)
        if self._save_density:
            self.save_density()

        #self.save_result()
        return self._total_energy

    def save_result(self, pfname=None):
        if pfname is None:
            pfname = os.path.join(self._workdir, 'result.pickle')
        self._results['Converged'] =self._converged
        self._results['TBnoU_energy'] = self._TBnoU_energy
        self._results['band_energy'] = self._band_energy
        self._results['U_energy'] = self._U_energy
        self._results['Uband_energy'] = self._Uband_energy
        self._results['int_energy'] = self._int_energy
        self._results['DC_energy'] = self._DC_energy
        self._results['total_energy'] = self._total_energy
        self._results[
            'correction_energy'] = self._total_energy - self._TBnoU_energy
        self._results['efermi'] = self._efermi
        self._results['kpts'] = self._kpts
        self._results['kweights'] = self._kweights
        self._results['nel'] = self._nel
        self._results['basis_set'] = self.bset
        self._results['orb_occupations'] = self.get_charges()
        self._results['orb_magnetic_moments'] = self.get_magnetic_moments()
        self._results['site_charges'] = self.get_site_charges()
        self._results[
            'site_magnetic_moments'] = self.get_site_magnetic_moments()
        if self._has_ham_U:
            self._results['ham_U'] = self._ham_U

        logging.info("Parameters:")
        logging.info("=========================================")

        logging.info("Basis set:")
        #for b in self.bset:
        #    logging.info("%s" % (b, ))

        logging.info("Kpoints:")
        for k in self._kpts:
            logging.info("%s" % k)

        logging.info("Kweights:")
        for k in self._kweights:
            logging.info("%s" % k)

        logging.info("Results of self-consistent calculation:")
        logging.info("=========================================")
        for key in [
                'band_energy', 'U_energy', 'total_energy', 'efermi', 'nel'
        ]:
            logging.info("%s : %s" % (key, self._results[key]))
        #for key in [
        #        'orb_occupations', 'orb_magnetic_moments', 'site_charges',
        #        'site_magnetic_moments'
        #]:
        #    logging.info(
        #        "%s: %s" % (key, ' '.join("%s" % x
        #                                  for x in self._results[key])))
        pfpath, _ = os.path.split(pfname)
        if not os.path.exists(pfpath):
            os.makedirs(pfpath)
        with open(pfname, 'wb') as myfile:
            pickle.dump(self._results, myfile)

    def plot_dos(self, split_spin=True, ax=None):
        efermi = self.get_fermi_level()
        if ax is None:
            fig, ax = plt.subplots()
        if split_spin and self._actual_nspin == 2:
            e, dos_up = self.get_dos(spin=0)
            e, dos_dn = self.get_dos(spin=1)
            ax.plot(e, dos_up)
            ax.plot(e, -dos_dn)
            ax.axhline(color='black')
        else:
            e, dos_up = self.get_dos(spin=0)
            ax.plot(e, dos_up)
        return ax

    def calc_chi0_list(self,
                       qlist,
                       omega=0.0,
                       kmesh=None,
                       supercell_matrix=None):
        return calc_chi0_list(
            self,
            qlist,
            omega=omega,
            kmesh=kmesh,
            supercell_matrix=supercell_matrix)

    def unfold(self, kpts, sc_matrix, evals=None, evecs=None):
        if evals is None and evecs is None:
            evals, evecs = self.solve_all(
                k_list=kpts, eig_vectors=True, convention=1)
        positions = self._orb
        # tbmodel: evecs[iband, ikpt, iorb]
        # unfolder: [ikpt, iorb, iband]
        self.unf = Unfolder(
            cell=self.atoms.cell,
            basis=[b.label + str(b.spin)
                   for b in self.bset],  # self.bset.get_labels(),
            positions=positions,
            supercell_matrix=sc_matrix,
            eigenvectors=np.swapaxes(np.swapaxes(evecs, 0, 1), 1, 2),
            qpoints=kpts)
        return evals, self.unf.get_weights()

    def plot_unfolded_band(
            self,
            supercell_matrix,
            kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                               [0, 0, 0], [.5, .5, .5]]),
            knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            npoints=200,
            ax=None,
    ):
        """
        plot the projection of the band to the basis
        """
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.atoms.cell, npoints)
        kslist = [x] * len(self._orb)
        evals, wkslist = self.unfold(kpts, supercell_matrix)  #.T * 0.98 + 0.01
        wkslist = wkslist.T * 0.98 + 0.01
        ekslist = evals
        ax = plot_band_weight(
            kslist,
            ekslist,
            wkslist=wkslist,
            efermi=None,
            yrange=None,
            output=None,
            style='alpha',
            color='blue',
            axis=ax,
            width=20,
            xticks=None)
        for i in range(len(self._orb)):
            ax.plot(x, evals[i, :], color='gray', alpha=1, linewidth=0.1)

        ax.axhline(self.get_fermi_level(), linestyle='--', color='gray')
        ax.set_xlabel('k-point')
        ax.set_ylabel('Energy (eV)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def plot_band(self,
                  kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                                     [0, 0, 0], [.5, .5, .5]]),
                  knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                  supercell_matrix=None,
                  split_spin=True,
                  npoints=100,
                  ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        evalues, evecs = self.solve_all(k_list=kpts, eig_vectors=True)
        if split_spin and self._actual_nspin == 2:
            evalues_up, evalues_dn = self.split_eigenvalue_spin(
                kpts, evalues, evecs)
            for i in range(self._norb // 2):
                try:
                    ax.plot(
                        x,
                        evalues_up[i, :],
                        color='red',
                        alpha=0.5,
                        label='up')
                    ax.plot(
                        x,
                        evalues_dn[i, :],
                        color='blue',
                        alpha=0.5,
                        label='down')
                except:
                    ax.plot(
                        x,
                        evalues[2 * i, :],
                        color='red',
                        alpha=0.7,
                        label='up')
                    ax.plot(
                        x,
                        evalues[2 * i + 1, :],
                        color='blue',
                        alpha=0.7,
                        label='down')

        else:
            for i in range(self._norb):
                ax.plot(x, evalues[i, :], color='blue', alpha=1)

        plt.axhline(self.get_fermi_level(), linestyle='--', color='gray')
        ax.set_xlabel('k-point')
        ax.set_ylabel('Energy (eV)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def plot_pdos(self, colordict=None, labeldict=None, ax=None):
        emin = np.min(self._eigenvals) - 1.0
        emax = np.max(self._eigenvals) + 1.0
        pdos = self.get_pdos(emin, emax, nedos=200, sigma=0.05)
        energy = pdos.get_energy()
        doses = pdos.get_pdos()
        if ax is None:
            fig, ax = plt.subplots()
        if colordict is None:
            for i in range(self._norb):
                if labeldict is None:
                    b = self.bset[i]
                    label = "%s_%s_%s" % (b.site, b.spin, b.label)
                else:
                    label = labeldict[i]
                b = self.bset[i]
                ax.plot(doses[i], pdos.get_energy(), label=label)
        else:
            for i, color in colordict.items():
                if labeldict is None:
                    b = self.bset[i]
                    label = "%s_%s_%s" % (b.site, b.spin, b.label)
                else:
                    label = labeldict[i]
                b = self.bset[i]
                ax.plot(doses[i], pdos.get_energy(), label=label)

        # # get data from first line of the plot
        # for i in range(len(ax.lines)):
        #     newx = ax.lines[0].get_ydata()
        #     newy = ax.lines[0].get_xdata()
        #     # set new x- and y- data for the line
        #     ax.lines[0].set_xdata(newx)
        #     ax.lines[0].set_ydata(newy)
        #     plt.draw()
        return ax

    def get_projection(self, orb, spin=0, eigenvecs=None):
        """
        get the projection to nth orb.

        :param orb: the index of the orbital.
        :param spin: if spin polarized, 0 or 1

        :returns: eigenvecs[iband,ikpt]
        """
        if eigenvecs is None:
            eigenvecs = self._eigenvecs
        if self._nspin == 2:
            return eigenvecs[:, :, orb, spin] * eigenvecs[:, :, orb,
                                                          spin].conjugate()
        else:
            return eigenvecs[:, :, orb] * eigenvecs[:, :, orb].conjugate()

    def plot_projection_band(
            self,
            spin=0,
            color_dict={},
            kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                               [0, 0, 0], [.5, .5, .5]]),
            knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            supercell_matrix=None,
            split_spin=True,
            npoints=200,
            ax=None,
    ):
        """
        plot the projection of the band to the basis
        """
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        evalues, evecs = self.solve_all(k_list=kpts, eig_vectors=True)
        #fig,a = plt.subplots()
        kslist = [x] * self._norb
        efermi = self.get_fermi_level()
        for orb, color in color_dict.items():
            ekslist = evalues
            wkslist = np.abs(
                self.get_projection(orb, spin=spin, eigenvecs=evecs))
            ax = plot_band_weight(
                kslist,
                ekslist,
                wkslist=wkslist,
                efermi=None,
                yrange=None,
                output=None,
                style='alpha',
                color=color,
                axis=ax,
                width=20,
                xticks=None)
        for i in range(self._norb):
            ax.plot(x, evalues[i, :], color='gray', alpha=1, linewidth=0.1)

        ax.axhline(self.get_fermi_level(), linestyle='--', color='gray')
        ax.set_xlabel('k-point')
        ax.set_ylabel('Energy (eV)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

    def plot_COHP_fatband(self,
                          kvectors=np.array([[0, 0, 0], [0.5, 0,
                                                         0], [0.5, 0.5, 0],
                                             [0, 0, 0], [.5, .5, .5]]),
                          knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                          supercell_matrix=None,
                          npoints=200,
                          width=5,
                          iblock=None,
                          jblock=None,
                          show=False,
                          efermi=None,
                          axis=None):
        band_cohp = COHP(tbmodel=self)
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        if efermi is None:
            efermi = self.get_fermi_level()
        ax = band_cohp.plot_COHP_fatband(
            kpts=kpts,
            k_x=x,
            X=X,
            width=width,
            iblock=iblock,
            jblock=jblock,
            xnames=knames,
            show=show,
            efermi=efermi,
            axis=axis)
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')

        return ax

    def plot_epc_fatband(self,
                         epc,
                         onsite=True,
                         order=1,
                         kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                            [0.5, 0.5, 0], [0, 0,
                                                            0], [.5, .5, .5]]),
                         knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                         supercell_matrix=None,
                         npoints=200,
                         width=5,
                         iblock=None,
                         jblock=None,
                         show=False,
                         efermi=None,
                         axis=None):
        if supercell_matrix is not None:
            kvectors = [np.dot(k, supercell_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.get_cell(), npoints)
        if efermi is None:
            efermi = self.get_fermi_level()
        evals, evecs = self.solve_all(
            k_list=kpts, eig_vectors=True, convention=2)
        evals = evals - efermi
        ax = epc.plot_epc_fatband(
            kpts,
            evals,
            evecs,
            order=order,
            onsite=onsite,
            k_x=x,
            X=X,
            xnames=knames,
            width=width,
            show=show,
            efermi=0.0,
            axis=axis)
        return ax

    #def plot_epc_dos(self, onsite=True, axis=None):
    def get_epc_dos(self,
                    epc,
                    onsite=True,
                    emin=-10,
                    emax=8,
                    order=1,
                    nedos=500,
                    sigma=0.1):
        evals = self._eigenvals
        weights = epc.get_band_shift(
            self._kpts,
            self._eigenvecs,
            evals=evals,
            onsite=onsite,
            order=order)
        d = WDOS(self._kpts, self._kweights, sigma, evals, weights, emin, emax,
                 nedos)
        return d.get_energy(), d.get_wdos(), d.get_idos()

    def plot_epc_dos(
            self,
            epc,
            onsite=True,
            emin=-10,
            emax=8,
            nedos=500,
            sigma=0.1,
            order=1,
            idos=True,
            color='red',
            axis=None,
            save_fname=None,
    ):
        e, dos, idos = self.get_epc_dos(
            epc,
            onsite=onsite,
            order=order,
            emin=emin,
            emax=emax,
            nedos=nedos,
            sigma=sigma)
        if save_fname is not None:
            data = np.array([e, dos, idos]).T
            np.savetxt(
                save_fname,
                data,
                header="#Energy  DOS   IDOS  Efermi: %s" % self._efermi)
        if axis is None:
            ax = plt.subplot()
        ax.plot(e, dos, color=color, label='DOS')
        ax.plot(e, idos, color='gray', label='IDOS')
        return ax


class atoms_model(etb_model):
    def __init__(self, atoms=None, basis_dict=None, basis_set=None, nspin=2):
        if atoms is not None:
            self.atoms = atoms
            self.basis_dict = basis_dict
            if basis_dict is not None:
                self.bset = atoms_to_basis(
                    atoms, basis_dict=basis_dict, nspin=nspin)
            elif basis_set is not None:
                self.bset = basis_set
            else:
                raise ValueError(
                    'basis_dict and basis_set should not both be None')
            super(atoms_model, self).__init__(
                dim_k=3,
                dim_r=3,
                lat=self.atoms.cell,
                orb=self.bset.get_scaled_positions(),
                nspin=nspin)
        else:
            super(atoms_model, self).__init__()

        self._has_rho = False
        self.Utype = None
        self.Hubbard_dict = {}

    def set_Hubbard_U(self,
                      Utype='Dudarev',
                      Hubbard_dict={},
                      DC=True,
                      DC_shift=False,
                      dim=5,
                      DC_type='FLL-ns'):
        self.Utype = Utype
        self.Hubbard_dict = Hubbard_dict
        if DC_type not in ['FLL-ns', 'Held', 'FLL-s', 'NoDC']:
            raise ValueError(
                "DC_type should be one of ['FLL-ns', 'Held', 'FLL-s', 'NoDC']")
        if Utype == 'SUN' or Utype == 0:
            uterm = Hubbard_U_SUN(
                bset=self.bset,
                Hubbard_dict=Hubbard_dict,
                DC=DC,
                DC_shift=DC_shift)
            self._diag_only = False
        elif Utype == 'Dudarev' or Utype == 2:
            # Double counting is already in, so don't do double-double counting
            uterm = Dudarev_term(
                bset=self.bset,
                Hubbard_dict=Hubbard_dict,
                DC=DC,
                DC_shift=DC_shift,
                DC_type=DC_type)
            #uterm = Hubbard_U_Dudarev(
            #    bset=self.bset,
            #    Hubbard_dict=Hubbard_dict,
            #    DC=False,
            #    DC_shift=DC_shift)
            self._diag_only = False
        elif Utype == 'Liechtenstein' or Utype == 1:
            #uterm = Hubbard_U_Liechtenstein(
            #    bset=self.bset,
            #    Hubbard_dict=Hubbard_dict,
            #    DC=DC,
            #    DC_shift=DC_shift,
            #    DC_type=DC_type)
            uterm = Lichtenstein_term(
                bset=self.bset,
                Hubbard_dict=Hubbard_dict,
                DC=DC,
                DC_shift=DC_shift,
                DC_type=DC_type)
            self._diag_only = False
        elif Utype == 'Kanamori' or Utype == 3:
            #uterm = Hubbard_U_Kanamori(
            uterm = Kanamori_term(
                bset=self.bset,
                Hubbard_dict=Hubbard_dict,
                DC=DC,
                DC_shift=DC_shift,
                DC_type=DC_type)
            self._diag_only = True
        else:
            raise ValueError(
                "Utype=%s, should be SUN|Liechtenstein|Dudarev|Kanamori or 0|1|2|3."
                % Utype)
        self.set_U_term(uterm)

    def make_supercell(self, sc_matrix, ijR=False):
        smaker = SupercellMaker(sc_matrix)
        # atoms
        sc_atoms = smaker.sc_atoms(self.atoms)

        # basis_set
        sc_bset = self.bset.make_supercell(sc_matrix)
        sc_model = atoms_model(
            atoms=sc_atoms, basis_set=sc_bset, nspin=self._actual_nspin)

        # hopping
        # pythtb hopping list of [hop, ind_i, ind_j, np.array(ind_R)]
        if ijR:
            sc_hoppings = defaultdict(
                lambda: np.zeros((sc_model._norb, sc_model._norb), dtype=float)
            )

            for R, mat in self.data.items():
                for i in range(self.nbasis):
                    for j in range(self.nbasis):
                        for sc_i, sc_j, sc_R in smaker.sc_ijR_only(
                                i, j, R, self.nbasis):
                            sc_hoppings[sc_R][sc_i, sc_j] = mat[i, j]
            sc_model._hoppings = sc_hoppings
        else:
            hop_dict = dict()
            for ind_R, ijdict in self._hoppings.items():
                if isinstance(ijdict, dict):
                    for (ind_i, ind_j), hop in ijdict.items():
                        hop_dict[(ind_i, ind_j, tuple(ind_R))] = hop
                else:
                    ijdict[np.abs(ijdict) < 1e-5] = 0.0
                    s = coo_matrix(ijdict)
                    for i in range(s.nnz):
                        hop_dict[(s.row[i], s.col[i],
                                  tuple(ind_R))] = s.data[i]
            pos = self.bset.get_scaled_positions()

            #print("build sc_hop")
            sc_hoppings = smaker.sc_ijR(hop_dict, n_basis=len(pos))
            for key, val in sc_hoppings.items():
                ind_i, ind_j, ind_R = key
                sc_model.set_hop(val, ind_i, ind_j, ind_R)

        sc_onsite = smaker.sc_trans_invariant(self._site_energies)

        del smaker

        sc_model.set_onsite(sc_onsite)

        if self.Utype is not None:
            sc_model.set_Hubbard_U(
                Utype=self.Utype, Hubbard_dict=self.Hubbard_dict)
        return sc_model

    def save(self, fname):
        with open(fname, 'wb') as myfile:
            pickle.dump(self, myfile)

    def load(self, fname):
        with open(fname, 'rb') as myfile:
            p = pickle.load(myfile)
        self.__dict__.update(p.__dict__)

    def save_results(self, fname):
        pass

    def get_site_nelectron(self, spin=None):
        n = np.zeros((len(self.bset.atoms), self._actual_nspin), dtype=float)
        for b in self.bset:
            n[b.site, b.spin] += self._rho[b.index, b.index]
        if spin is None:
            return np.sum(n, axis=1)
        else:
            return n[:, spin]

    def get_site_charges(self):
        return self.get_site_nelectron()

    def get_site_magnetic_moments(self):
        return self.get_site_nelectron(0) - self.get_site_nelectron(1)

    def calc_piecewise_linear(self, U, J, step=0.1):
        nel0 = self._nel
        energies = []
        nels = np.arange(-1.0, 1.000001, step=step)
        for nel in nels:
            model = copy.deepcopy(self)
            model._nel = nel0 + nel
            model.scf_solve()
            energies.append(model.get_energy())
        return nels, energies

    def plot_piecewise_linear(self, U, J, step=0.25, axis=None):
        if axis is None:
            fig, axis = plt.subplots()
        nels, energies = self.calc_piecewise_linear(U, J, step=step)

        n = len(nels) - 1
        mid = round(n / 2)
        linfunc = interp1d(
            x=[nels[0], nels[mid], nels[-1]],
            y=[energies[0], energies[mid], energies[-1]])

        energies = np.array(energies)
        axis.plot(
            nels,
            energies - linfunc(nels),
            marker='o',
            label='U=%.2f, J=%.2f' % (U, J))
        axis.set_xlabel('nel')
        axis.set_ylabel('Energy (eV)')
        return axis


def load_pickle_to_model(fname):
    model = atoms_model()
    with open(fname, 'rb') as myfile:
        p = pickle.load(myfile)
    model.__dict__.update(p.__dict__)
    return model


def test_atoms_model():
    t = -1
    atoms = Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1])
    model = atoms_model(atoms, basis_dict={'H': ['s']}, nspin=2)
    model.set_Hubbard_U(Utype='SUN', Hubbard_dict={'H': {'U': 9, 'J': 0}})
    model.set_hop(t, 0, 0, ind_R=(1, 0, 0))
    model.set_hop(t, 1, 1, ind_R=(1, 0, 0))
    model.set_hop(t, 0, 0, ind_R=(0, 1, 0))
    model.set_hop(t, 1, 1, ind_R=(0, 1, 0))
    model.set_hop(t, 0, 0, ind_R=(0, 0, 1))
    model.set_hop(t, 1, 1, ind_R=(0, 0, 1))
    print(model.atoms)
    print(model.bset)
    print(model.bset.get_scaled_positions())
    print(model._U_term.bset)
    print(model._U_term.Udict)

    sc_mat = np.diag([3, 3, 3])
    scmodel = model.make_supercell(sc_mat)
    scmodel.set_initial_spin([1.1] * 26 + [-1])
    print(scmodel.atoms)
    print(scmodel.bset)
    print(scmodel.bset.get_scaled_positions())
    print(scmodel._U_term.bset)
    print(scmodel._U_term.Udict)
    print(model._hoppings)
    print(scmodel._hoppings)

    scmodel.set(nel=15.9, mixing=0.5)

    scmodel.set_kmesh([4, 4, 4])
    scmodel.scf_solve()
    print(scmodel.get_charges())
    print(scmodel.get_magnetic_moments())
    scmodel.plot_band(supercell_matrix=sc_mat, split_spin=True, npoints=200)
    plt.show()


if __name__ == '__main__':
    test_atoms_model()
