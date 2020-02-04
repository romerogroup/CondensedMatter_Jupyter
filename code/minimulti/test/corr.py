from netCDF4 import Dataset
import os
import numpy as np
import scipy as sp
from itertools import product
import numba
from numba import int32, complex128, float64
import matplotlib.pyplot as plt


@numba.njit(fastmath=True)
def sq(jlist, Rlist, sa, q):
    total = 0 + 0.0j
    savg = np.sum(sa) / len(sa)
    for i in range(len(jlist)):
        j = jlist[i]
        R = Rlist[i]
        total += (sa[j] - savg) * np.exp(-2.0j * np.pi * np.dot(q, R))
    return total


@numba.njit(fastmath=True)
def sqs(jlist, Rlist, S, q):
    """
    S(t, xyz) from S(t, R(j), xyz)
    input:
     - jlist: j of S
     - Rlist: R of S
     - S: 
    S(t, alpha):
    """
    ntime, nitem, ndim = S.shape
    ret = np.zeros((ntime, ndim), dtype=complex128)
    savg = np.zeros((ntime, ndim), dtype=float64)
    # avg over site
    savg = np.sum(S, axis=1) / nitem
    # normalize why?
    #for itime in range(ntime):
    #    savg[itime] = savg[itime] / np.linalg.norm(savg[itime])
    for i in range(nitem):
        j = jlist[i]
        R = Rlist[i]
        if R[0]<7 and R[1]<7 and R[2]<7:
            phase = np.exp(-2.0j * np.pi * np.dot(q, R))
        else:
            phase = 0.0j
        for itime in range(ntime):
            for idim in range(ndim):
                ret[itime, idim] += (
                    S[itime, j, idim] - savg[itime, idim]) * phase
                #S[itime, j, idim]) * phase
    return ret


def zero_padding(x):
    return np.concatenate([np.zeros(len(x) * 2), x, np.zeros(len(x) * 2)])


def myfft0(x):
    x[0] = 0.0
    x-=np.average(x)
    win = np.blackman(len(x))
    f = np.fft.fft(zero_padding(x * win))
    N = len(f)
    ind_freq = np.arange(1, N // 2, dtype=int)
    r = np.abs(f[ind_freq])**2 + np.abs(f[-ind_freq])**2
    return r / np.sqrt(np.sum(r**2))


def myfft(x):
    from scipy.signal import welch
    return np.log(welch(x, nperseg=len(x), scaling='spectrum'))


class SpinHistParser(object):
    def __init__(self, fname):
        self.fname = fname
        self.ds = Dataset(fname, 'r')
        self._parse()

    def _parse(self):
        pass

    @property
    def S(self):
        return self.ds.variables['S'][1:, :, :]

    @property
    def Rvec(self):
        return self.ds.variables['Rvec'][:, :]


class CorrelationFuncions(object):
    def __init__(self, sc_matrix, Rvecs, S):
        self.sc_matrix = sc_matrix
        self.Rvecs = Rvecs
        self.ilist = list(range(len(self.Rvecs)))
        self.Ridict = {}
        for i, R in enumerate(self.Rvecs):
            self.Ridict[tuple(R)] = i
        self.S = S

    def i_for_R(self, R):
        """
        find the index i for a R
        """
        return self.Ridict[tuple(R)]

    def _gen_Rlist(self):
        """
        generate a list of R inside Rrange
        """
        rrange = self.rrange
        Rlist = product(
            range(-rrange[0], rrange[0]),
            range(-rrange[1], rrange[1]),
            range(-rrange[2], rrange[2]),
        )
        Rlist = list(Rlist)
        return Rlist

    def _gen_ij_R(self, R):
        """
        generate j so that Rij=R
        the list of i is given by self.ilist
        for R=0, 
        """
        R = np.array(R, dtype=float)
        abc = np.diag(self.sc_matrix)
        Rwrap = np.remainder(self.Rvecs + R, abc[None, :])
        return list(range(
            len(Rwrap))), [self.i_for_R(Ri) for Ri in Rwrap], Rwrap

    def sq_all(self, q):
        """
        s(t, iq, alpha) = \sum_j S(t, j, alpha ) * exp(-2pi q Rj) 
        """
        ilist, jlist, Rlist = self._gen_ij_R(R=0)
        ilist = np.array(ilist)
        jlist = np.array(jlist)
        Rlist = np.array(Rlist)
        return sqs(
            np.array(jlist), np.array(Rlist, dtype='float'), self.S,
            np.array(q))

    def sqfft(self, sa):
        s = np.reshape(sa, (18, 18, 18))
        Ak = np.fft.fftn(s)
        return Ak

    def plot_spin_wave(self, qpoints, nfreq=500):
        from matplotlib import cm
        x = np.arange(len(qpoints))
        y = np.arange(nfreq)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X, dtype=float)
        for iq, q in enumerate(qpoints):
            p = self.sq_all(q)
            p = p - np.average(p, axis=0)[None, :]
            pc = np.correlate(p[:, 1][1:], p[:, 1][1:nfreq], 'valid')
            #plt.plot(pc, linewidth=1, label='%s'%q, marker='.')
            #plt.plot(myfft(pc)[:nfreq] + q * 10, label="%.2f" % q)

            Z[:, iq] = myfft0(pc)[:nfreq]
        norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
        print(abs(Z).max())
        print(abs(Z).min())
        #plt.contourf(X, Y, Z, 10, norm=norm)
        plt.imshow(Z, aspect='auto', interpolation='bicubic', origin='lower')
        #plt.imshow(Z, aspect='auto', interpolation=None, origin='lower')
        plt.show()


def test(fname="./mb.out_spinhist.nc", ncell=12):
    f = SpinHistParser(fname)
    cf = CorrelationFuncions(
        sc_matrix=np.eye(3) * ncell, Rvecs=f.Rvec, S=f.S)

    qpoints = [(q, q, q) for q in np.arange(0.0, 0.51, 0.03)]
    cf.plot_spin_wave(qpoints=qpoints)

    for q in np.arange(0.0, 0.51, 0.03):
        p = cf.sq_all(q=[q, q, q])
        p = p - np.average(p, axis=0)[None, :]
        pc = np.correlate(p[:, 1][1:], p[:, 1][1:100], 'valid')
        plt.plot(myfft0(pc)[:300] + q * 10, label="%.2f" % q)
    plt.xlabel('1e12 Hz')
    plt.legend()
    plt.show()


#test(fname="./mb.out_spinhist.nc")
test(fname='./tmulti5_1.out_spinhist.nc', ncell=12)
