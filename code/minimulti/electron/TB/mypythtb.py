from pythtb import tb_model
import pythtb
from ase.dft.dos import DOS
import numpy as np
from occupations import Occupations
from ase.dft.kpoints import monkhorst_pack
from plot import plot_band_weight


class dummy_tb(object):
    def __init__(self):
        pass

    def set_orbital_label(self):
        pass

    def get_eigenvalues(self, kpt=0, spin=None, refresh=False):
        pass

    def get_number_of_bands(self):
        """
        number of bands.
        """
        pass

    def get_number_of_spins(self):
        pass

    def get_k_point_weights(self):
        pass

    def get_fermi_level(self):
        pass

    def get_bz_k_points(self):
        return self._kpts

    def get_ibz_k_points(self):
        raise NotImplementedError


class mymodel(tb_model):
    # This is fully compatible with tb_model, only a few functions
    # are overided so it is faster
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        super(mymodel, self).__init__(
            dim_k=dim_k, dim_r=dim_r, lat=lat, orb=orb, per=per, nspin=nspin)


class etb_model(mymodel, dummy_tb):
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        mymodel.__init__(
            self,
            dim_k=dim_k,
            dim_r=dim_r,
            lat=lat,
            orb=orb,
            per=per,
            nspin=nspin)

        self._efermi = None
        self._nel = None
        self._smearing = None
        self._sigma = None
        self._eigenvals = None
        self._eigenvecs = None
        self._eigenvecs = None
        self._nspin = nspin

        self._efermi = None
        self._occupations = None
        self._kpts = None
        # _kweight is a  array [w1,w2,....].
        self._kweights = None
        self._old_occupations = None
        self._verbose = False
        self._eps = 0.001
        self._nbar = np.ndarray([len(self._orb), 2])

    def read_basis(self, fname='basis.txt'):
        basis = []
        with open(fname) as myfile:
            for line in myfile:
                if line.strip() != '':
                    basis.append(line.strip().split()[0])
        self._basis = basis
        return basis

    def get_onsite_energies(self):
        return self._onsite_energies()

    def set(self, nel=None, smearing='gaussian', sigma=0.1):
        self._nel = nel
        self._smearing = smearing
        self._width = sigma

    def set_kpoints(self, kpts, kweights=None):
        """
        set the kpoints to calculate. each kpoint can be a
        """
        self._kpts = kpts
        if kweights is None:
            self._kweights = np.array([1.0 / len(self._kpts)] *
                                      len(self._kpts))
        else:
            self._kweights = kweights

    def set_kmesh(self, kmesh):
        self.set_kpoints(monkhorst_pack(kmesh))

    def get_number_of_bands(self):
        return self._norb

    def get_number_of_spins(self):
        return self._nspin

    def get_energy_level_occupations(self):
        pass

    def get_orbital_occupations(self):
        pass

    def on_site_energies(self):
        pass

    def get_fermi_level(self):
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
            #print self.solve_all(k_list=self._kpts,eig_vectors=True)
            self._eigenvals, self._eigenvecs = self.solve_all(
                k_list=self._kpts, eig_vectors=True)
        return self._eigenvals

    def get_eigenvalues(self, kpt=0, spin=None, refresh=False):
        """
        Ak_spin. Calculate the eigenvalues and eigen vectors. the eigenvalues are returned.
        self._eigenvals are returned.
        """
        if self._eigenvals is None or refresh:
            #print self.solve_all(k_list=self._kpts,eig_vectors=True)
            self._eigenvals, self._eigenvecs = self.solve_all(
                k_list=self._kpts, eig_vectors=True)
        if spin is None or self._nspin == 1:
            return self._eigenvals[:, kpt]
        else:
            ## seperate the spin up/ down
            ## project the evec to spin up/down basis
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
                #return self._eigenvals[::2,kpt]
                return eval_up
            if spin == 1 or spin == 'DOWN':
                return eval_dn
                #return self._eigenvals[1::2,kpt]

    def get_dos(self, width=None, smearing='gaussian', npts=501):
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
        return dos.get_energies(), dos.get_dos()

    def get_occupations(self, nel=None, width=0.05, refresh=False):
        """
        calculate occupations of each eigenvalue.
        the the shape of the occupation is the same as self._eigenvals.
        [eig_k1,eigk2,...], each eig_k is a column with the length=nbands.

        :param nel: number of electrons.

        :Returns:
          self._occupations (np.ndarray) index:[band,kpt,orb,spin] if nspin==2 else [band,kpt,orb] same as eigenvec
        """
        if nel is not None:
            self._nel = nel
        nel = self._nel
        self.get_eigenvalues(refresh=refresh)
        #print self._kweights
        #print self._eigenvals
        if self._nspin == 1:
            occ = Occupations(nel, width, self._kweights, nspin=self._nspin)
            self._occupations = occ.occupy(self._eigenvals)
            self._efermi = occ.get_mu()

        elif self._nspin == 2:
            raise NotImplementedError(
                "current implement on fix_spin is not correct.")
            nel_up, nel_dn = nel
            eig_up = self.eigenvals[::2]
            eig_dn = self.eigenvals[1::2]

            occ_up = Occupations(nel_up, width, self._kweights, nspin=1)
            occupations_up = occ_up.occupy(eig_up)
            efermi_up = occ_up.get_mu()

            occ_dn = Occupations(nel_dn, width, self._kweights, nspin=1)
            occupations_dn = occ_dn.occupy(eig_dn)
            efermi_dn = occ_dn.get_mu()

            self._occupations[::2] = occupations_up
            self._occupations[1::2] = occupations_dn

            self.efermi = (efermi_up, efermi_dn)
        return self._occupations

    def get_orbital_occupations(self, refresh=True):
        """
        self.occupations:
        if spin==1: the indexes are [orb];
        if spin==2: the indexes are [orb,spin]
        """
        A2 = np.abs(self._eigenvecs)**2
        #first sum over band
        #print self._occupations.shape
        #print(A2.sum(axis=0))
        # occupations: index same as eigenval. [band, k]
        ni, nk = self._occupations.shape
        V2 = np.zeros(A2.shape, dtype=float)
        if self._nspin == 1:
            for i in range(ni):
                for j in range(nk):
                    V2[i, j] = self._occupations[i, j] * A2[
                        i, j] * self._kweights[j]
            self._nbar = V2.sum(axis=(0, 1))
        elif self._nspin == 2:
            for i in range(ni):
                for j in range(nk):
                    V2[i, j] = self._occupations[i, j] * A2[
                        i, j] * self._kweights[j]
            #V2=self._occupations.flatten()*A2.reshape(ni*nk,ni)/len(self._kweights)
            #V2=(self._occupations*A2).sum(axis=(0,1))#/len(self._kweights)

            self._nbar = V2.sum(axis=(0, 1))
        return self._nbar

    def get_band_energy(self):
        """
        Not free energy. total energy. sum of occupied levels.
        """
        self.energy = (self._kweights *
                       (self._occupations * self._eigenvals)).sum()

    def get_projection(self, orb, spin=0):
        """
        get the projection to nth orb.

        :param orb: the index of the orbital.
        :param spin: if spin polarized, 0 or 1

        :returns: eigenvecs[iband,ikpt]
        """
        if self._nspin == 2:
            return np.self._eigenvecs[:, :, orb, spin]
        else:
            return self._eigenvecs[:, :, orb] * self._eigenvecs[:, :,
                                                                orb].conjugate()

    def plot_projection_band(self,
                             orb,
                             spin=0,
                             xs=None,
                             color='blue',
                             axis=None,
                             xticks=None):
        """
        plot the projection of the band to the basis
        """
        if xs is None:
            kslist = [list(range(len(self._kpts)))] * self._norb
        else:
            kslist = [list(xs)] * self._norb
        ekslist = self._eigenvals
        wkslist = np.abs(self.get_projection(orb, spin=spin))

        #fig,a = plt.subplots()
        return plot_band_weight(
            kslist,
            ekslist,
            wkslist=wkslist,
            efermi=None,
            yrange=None,
            output=None,
            style='width',
            color=color,
            axis=axis,
            width=10,
            xticks=xticks)

    def get_pdos(self):
        """
        get projected dos to the basis set.
        """
        pass
