import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from netCDF4 import Dataset
from collections import defaultdict
from scipy.optimize import curve_fit
from minimulti.utils.supercell import SupercellMaker
from scipy.sparse import coo_matrix
from itertools import groupby

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from minimulti.learn.polynomial import PolynomialDerivative


class ijR(object):
    def __init__(self,
                 nbasis,
                 data=None,
                 positions=None,
                 sparse=False,
                 double_site_energy=2.0):
        self.nbasis = nbasis
        if data is not None:
            self.data = data
        else:
            self.data = defaultdict(
                lambda: np.zeros((nbasis, nbasis), dtype=float))
        if positions is None:
            self.positions = np.zeros((nbasis, 3))
        else:
            self.positions = positions
        self.sparse = sparse
        self.double_site_energy = double_site_energy
        if sparse:
            self._matrix = csr_matrix

    def to_sparse(self):
        for key, val in self.data:
            self.data[key] = self._matrix(val)

    def make_supercell(self, supercell_matrix=None, scmaker=None):
        if scmaker is None:
            scmaker = SupercellMaker(sc_matrix=supercell_matrix)
        ret = ijR(nbasis=self.nbasis * scmaker.ncell)
        if self.positions is None:
            ret.positions = None
        else:
            ret.positions = scmaker.sc_pos(self.positions)
        ret.sparse = self.sparse
        ret.double_site_energy = self.double_site_energy
        for R, mat in self.data.items():
            for i in range(self.nbasis):
                for j in range(self.nbasis):
                    for sc_i, sc_j, sc_R in scmaker.sc_ijR_only(
                            i, j, R, self.nbasis):
                        ret.data[sc_R][sc_i, sc_j] = mat[i, j]
        return ret

    @property
    def Rlist(self):
        return list(self.data.keys())

    @property
    def nR(self):
        return len(self.Rlist)

    @property
    def site_energies(self):
        return self.data[(0, 0, 0)].diagonal() * self.double_site_energy

    @property
    def hoppings(self):
        data = copy.deepcopy(self.data)
        np.fill_diagonal(data[(0, 0, 0)], 0.0)
        return data

    @staticmethod
    def _positive_R_mat(R, mat):
        nzR = np.nonzero(R)[0]
        if len(nzR) != 0 and R[nzR[0]] < 0:
            newR = tuple(-np.array(R))
            newmat = mat.T.conj()
        elif len(nzR) == 0:
            newR = R
            newmat = (mat + mat.T.conj()) / 2.0
        else:
            newR = R
            newmat = mat
        return newR, newmat

    def _to_positive_R(self):
        new_ijR = ijR(self.nbasis, sparse=self.sparse)
        for R, mat in self.data:
            newR, newmat = self._positive_R_mat(R, mat)
            new_ijR[newR] = newmat
        return new_ijR

    def shift_position(self, rpos):
        pos = self.positions
        shift = np.zeros((self.nbasis, 3), dtype='int')
        shift[:, :] = np.round(pos - rpos)
        newpos = copy.deepcopy(pos)
        for i in range(self.nbasis):
            newpos[i] = pos[i] - shift[i]
        d = ijR(self.nbasis)
        d.positions = newpos
        for R, v in self.data.items():
            for i in range(self.nbasis):
                for j in range(self.nbasis):
                    sR = tuple(np.array(R) - shift[i] + shift[j])
                    nzR = np.nonzero(sR)[0]
                    if len(nzR) != 0 and sR[nzR[0]] < 0:
                        newR = tuple(-np.array(sR))
                        d.data[newR][j, i] += v[i, j]
                    elif len(nzR) == 0:
                        newR = sR
                        d.data[newR][i, j] += v[i, j] * 0.5
                        d.data[newR][j, i] += np.conj(v[i, j]) * 0.5
                    else:
                        d.data[sR][i, j] += v[i, j]
        return d

    def save(self, fname):
        root = Dataset(fname, 'w', format="NETCDF4")
        root.createDimension("nR", self.nR)
        root.createDimension("three", 3)
        root.createDimension("nbasis", self.nbasis)
        R = root.createVariable("R", 'i4', ("nR", "three"))
        data = root.createVariable("data", 'f8', ("nR", "nbasis", "nbasis"))
        positions = root.createVariable("positions", 'f8', ("nbasis", "three"))
        R[:] = np.array(self.Rlist)
        data[:] = np.array(tuple(self.data.values()))
        positions[:] = np.array(self.positions)
        root.close()

    @staticmethod
    def load_ijR(fname):
        root = Dataset(fname, 'r')
        nbasis = root.dimensions['nbasis'].size
        Rlist = root.variables['R'][:]
        m = ijR(nbasis)
        mdata = root.variables['data'][:]
        for iR, R in enumerate(Rlist):
            m.data[tuple(R)] = mdata[iR]
        return m

    @staticmethod
    def from_tbmodel(model):
        ret = ijR(nbasis=model.size)
        for R, v in model.hoppings.items():
            ret.data[R] = v
        ret.positions = np.reshape(model.pos, (model.nbasis, 3))
        return ret

    @staticmethod
    def from_tbmodel_hdf5(fname):
        from tbmodels import Model
        m = Model.from_hdf5_file(fname)
        ret = ijR(nbasis=m.size)
        for R, v in m.hop.items():
            ret.data[R] = v
        ret.positions = np.reshape(m.pos, (m.size, 3))
        return ret

    def to_spin_polarized(self, order=1):
        """
        repeat 
        order =1 : orb1_up, orb1_dn, orb2_up, orb2_dn...
        order =2 : orb1_up, orb2_up, ... orb1_dn, orb2_dn...
        """
        ret = ijR(self.nbasis * 2)
        if self.positions is None:
            ret.positions = None
        else:
            ret.positions = np.repeat(self.positions, 2, axis=0)
        for R, mat in self.data.items():
            ret.data[R][::2, ::2] = mat
            ret.data[R][1::2, 1::2] = mat
        return ret

    def gen_ham(self, k):
        Hk = np.zeros((self.nbasis, self.nbasis), dtype='complex')
        np.fill_diagonal(Hk, self.site_energies)
        for R, mat in self.hoppings.items():
            phase = np.exp(2j * np.pi * np.dot(k, R))
            Hk += mat * phase + (mat * phase).T.conj()
        return Hk

    def solve_all(self, kpts):
        nk = len(kpts)
        evals = np.zeros((nk, self.nbasis))
        evecs = np.zeros((nk, self.nbasis, self.nbasis))
        for ik, k in enumerate(kpts):
            evals_k, evecs_k = eigh(self.gen_ham(k))
            evals[ik, :] = evals_k
            evecs[ik, :, :] = evecs_k
        return evals, evecs

    def plot_band(self, kpts, color='green'):
        evals, evecs = self.solve_all(kpts)
        for i in range(self.nbasis):
            plt.plot(evals[:, i], color=color)

    def validate(self):
        # make sure all R are 3d.
        for R in self.data.keys():
            if len(R) != 3:
                raise ValueError("R should be 3d")


class EPCData():
    def __init__(self,
                 ref_positions,
                 nphonon,
                 phonon_amplitude=None,
                 ijR_list=None,
                 ncoeff=0,
                 func=None):
        self.ref_positions = ref_positions
        self.nphonon = nphonon
        self.phonon_amplitude = phonon_amplitude
        self.ijR_list = ijR_list
        if ijR_list is not None:
            self.nbasis = ijR_list[0].nbasis
        self.coeffs = None
        self.ncoeff = ncoeff
        self.func = func

    def check(self, R=(0, 0, 0), i=None, j=None):
        # check consistency
        nbasis_list = np.array([x.nbasis for x in ijR_list])
        if not (nbasis_list == nbasis_list[0]).all():
            raise ValueError("not all models have same norb")
        nbasis = nbasis_list[0]

        if i is None:
            irange = range(nbasis)
        else:
            irange = [i]
        if j is None:
            jrange = range(nbasis)
        else:
            jrange = [j]
        for i in irange:
            for j in jrange:
                ydata = [m.data[tuple(R)][i, j] for m in ijR_list]
                plt.plot(self.phonon_amplitude, ydata)
        plt.show()

    def format_data(self, cutoff=1e-5):
        """
        format data so it can fit into sklearn fit.
        X: (nsample, nfeature) so each row is the feature vector.
        y: (nsamples, ntarget) so each row is the values for all ijR 
        nsamples is the number of configurations.
        """
        # R list
        RR = tuple(set(x.Rlist) for x in self.ijR_list)
        Rlist = set.union(*RR)
        Rlist = sorted(list(Rlist))
        ndata = len(self.ijR_list)
        nbasis = self.ijR_list[0].nbasis
        # for each R, build dok matrix
        # [[iR, i, j],[...]]
        self.Rlist = Rlist
        self.indlist = np.zeros((0, 3), dtype=int)
        # (nijR, ind) = (nsample, ntarget)
        self.vallist = np.zeros((ndata, 0))
        for iR, R in enumerate(self.Rlist):
            ms = np.array([ijR.data[R] for ijR in self.ijR_list])
            spp = np.zeros((nbasis, nbasis), dtype=bool)
            for m in ms:
                spp += (np.abs(m) > cutoff)
            irow, icol = np.where(spp)
            n = len(irow)
            self.indlist = np.vstack(
                [self.indlist,
                 np.array([[iR] * n, irow, icol]).T])
            # val: id: [ind_ijR, ind_matrix], ind_ms -> ntarget in sklearn fit.
            val = ms[:, irow, icol]
            self.vallist = np.hstack([self.vallist, val])
        return self.indlist, self.vallist

    def save(self, fname):
        with open(fname, 'wb') as myfile:
            d = {
                'ref_positions': self.ref_positions,
                'nbasis': self.nbasis,
                'nphonon': self.nphonon,
                'ncoeff': self.ncoeff,
                'Rlist': self.Rlist,
                'indlist': self.indlist,
                'model': self.model
            }
            pickle.dump(d, myfile)

    def gen_model(self, phonon_amplitude):
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        for R in self.coeffs[0]:
            m.data[R] = self.func(*[c[R] for c in self.coeffs])
        return m

    @staticmethod
    def load_fitting(fname):
        with open(fname, 'rb') as myfile:
            d = pickle.load(myfile)
        cls = EPCData(ref_positions=d['ref_positions'], nphonon=d['nphonon'])
        cls.nbasis = d['nbasis']
        cls.ncoeff = d['ncoeff']
        cls.Rlist = d['Rlist']
        cls.indlist = d['indlist']
        cls.model = d['model']
        return cls

    def predict_ijR(self, amplitude):
        y = self.model.predict([amplitude])[0]
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        istart = 0
        for k, g in groupby(self.indlist, lambda x: x[0]):
            g = np.array(list(g))
            R = tuple(self.Rlist[k])
            mat = np.zeros((self.nbasis, self.nbasis))
            iend = istart + len(g)
            val = y[istart:iend]
            mat = coo_matrix((val, (g[:, 1], g[:, 2])),
                             shape=(self.nbasis, self.nbasis),
                             dtype=float).todense()
            istart = iend
            m.data[R] = mat
        return m

    def predict_model(self, amplitude, ref_model, spin=True):
        m = self.predict_ijR(amplitude)
        if spin:
            m = m.to_spin_polarized()
        return ijR_to_model(copy.deepcopy(ref_model), m)

    def predict_H1(self, amplitude, i):
        """
        predict first order derivative of Hamiltonian. and generate a ijR object.
        """
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        dpoly = PolynomialDerivative(
            self.model.named_steps['polynomialfeatures'],
            self.model.named_steps['linearregression'].coef_, i)
        y = dpoly.predict([amplitude])

        istart = 0
        for k, g in groupby(self.indlist, lambda x: x[0]):
            g = np.array(list(g))
            R = tuple(self.Rlist[k])
            mat = np.zeros((self.nbasis, self.nbasis))
            iend = istart + len(g)
            val = y[istart:iend]
            mat = coo_matrix((val, (g[:, 1], g[:, 2])),
                             shape=(self.nbasis, self.nbasis),
                             dtype=float).todense()
            istart = iend
            m.data[R] = mat
        return m

    def polyfit(self, degree=6, fname='polymodel.pickle'):
        poly = PolynomialFeatures(degree=degree)
        model = make_pipeline(poly, LinearRegression())
        self.indlist, self.vallist = self.format_data()
        X_train, X_test, y_train, y_test = train_test_split(
            self.phonon_amplitude, self.vallist, test_size=0.2)
        model.fit(self.phonon_amplitude, self.vallist)
        score = model.score(self.phonon_amplitude, self.vallist)
        print(score)
        score = model.score(X_test, y_test)
        print(score)
        self.model = model
        return model

    def fit_data(self, phonon_amplitude, ijR_list):
        func = self.func
        ncoeff = self.ncoeff
        # check consistency
        nbasis_list = np.array([x.nbasis for x in ijR_list])
        if not (nbasis_list == nbasis_list[0]).all():
            raise ValueError("not all models have same norb")
        nbasis = nbasis_list[0]

        # build list of R
        RR = [x.Rlist for x in ijR_list]
        Rset = set(RR[0])
        for R in RR:
            Rset.intersection_update(R)
        Rlist = tuple(Rset)

        R = (0, 0, 0)
        for i in [1]:
            for j in range(nbasis):
                ydata = [m.data[R][i, j] for m in ijR_list]
                plt.plot(phonon_amplitude, ydata)
        plt.show()

        coeffs = [ijR(nbasis) for i in range(ncoeff)]
        xdata = np.array(phonon_amplitude).T
        for R in Rlist:
            for i in range(nbasis):
                for j in range(nbasis):
                    ydata = np.real([m.data[R][i, j] for m in ijR_list])
                    coeff, pcov = curve_fit(
                        func, xdata, ydata, p0=[0.001] * ncoeff, method='trf')
                    perr = np.linalg.norm(np.diag(pcov))
                    if perr > 1:
                        print("coeff:", coeff)
                        print(pcov)
                        print(perr)
                        print(xdata)
                        print(ydata)
                        print(R, i, j)
                        plt.plot(phonon_amplitude, ydata)
                        plt.show()
                    for icoeff in range(ncoeff):
                        coeffs[icoeff].data[R][i, j] = coeff[icoeff]
        for icoeff in range(ncoeff):
            coeffs[icoeff].save('coeff_%s.nc' % icoeff)
        self.nbasis = nbasis
        self.coeffs = [dict(c.data) for c in coeffs]
        return coeffs

    def load_coeffs(self, coeffs=None, coeff_files=None, coeff_path=None):
        if coeff_path is not None:
            coeff_files = [
                os.path.join(coeff_path, 'coeff_%s.nc' % i)
                for i in range(self.ncoeff)
            ]
        if coeff_files is not None:
            self.coeffs = []
            for fname in coeff_files:
                c = ijR.load_ijR(fname)
                self.coeffs.append(dict(c.data))

    def gen_model(self, phonon_amplitudes):
        m = ijR(nbasis=self.nbasis, positions=self.ref_positions)
        for R in self.coeffs[0]:
            m.data[R] = self.func(*[c[R] for c in self.coeffs])
        return m


def ijR_to_model(model, dijR):
    model._hoppings = dijR.hoppings
    model._site_energies = dijR.site_energies
    return model
