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
    S(t, i, alpha):
    """
    ntime, nitem, ndim = S.shape
    ret = np.zeros((ntime, ndim), dtype=complex128)
    savg = np.zeros((ntime, ndim), dtype=float64)
    # avg over site
    savg = np.sum(S, axis=1) / nitem
    # normalize why?
    #for itime in range(ntime):
    #    savg[itime] = savg[itime] / np.linalg.norm(savg[itime])
    for i in range(len(jlist)):
        j = jlist[i]
        R = Rlist[i]
        phase = np.exp(-2.0j * np.pi * np.dot(q, R))
        for itime in range(ntime):
            for idim in range(ndim):
                ret[itime, idim] += (
                    S[itime, j, idim] - savg[itime, idim]) * phase
                    #S[itime, j, idim]) * phase
    return ret


#@numba.njit(fastmath=True)
@numba.njit("c16[:,:,:](f8[:,:,:], f8[:])", fastmath=True)
def sws(S, wlist):
    nw = len(wlist)
    ntime, nspin, ndim = S.shape
    Sw = np.zeros((nw, nspin, ndim), dtype=complex128)
    for iw in range(nw):
        for it in range(S.shape[0]):
            phase = np.exp(-2j * np.pi * wlist[iw] * it)
            for ispin in range(nspin):
                for idim in range(ndim):
                    Sw[iw, ispin, idim] += S[it, ispin, idim] * phase
    return Sw


@numba.njit(fastmath=True)
def SwR_to_Swq(S, Rlist, q):
    nR = len(Rlist)
    nw, nR, ndim = S.shape
    Swq = np.zeros((nw, ndim), dtype=numba.complex128)
    for iR in range(nR):
        phase = -2j * np.pi * np.dot(Rlist[iR, :], q[:])
        Swq[:, :] += S[:, iR, :] * phase
    return Swq


def zero_padding(x):
    return np.concatenate([np.zeros(len(x) * 2), x, np.zeros(len(x) * 2)])


def myfft0(x):
    x[0] = 0.0
    win = np.blackman(len(x))
    f = np.fft.fft(zero_padding(x * win))
    N = len(f)
    ind_freq = np.arange(1, N // 2, dtype=int)
    r = np.abs(f[ind_freq])**2  + np.abs(f[-ind_freq])**2
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
    def __init__(self, sc_matrix, rrange, Rvecs, S):
        self.sc_matrix = sc_matrix
        self.rrange = rrange
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
        """
        R = np.array(R, dtype=float)
        abc = np.diag(self.sc_matrix)
        Rwrap = np.remainder(self.Rvecs + R, abc[None, :])
        return list(range(
            len(Rwrap))), [self.i_for_R(Ri) for Ri in Rwrap], Rwrap

    def C_ia_jb(self, S_ia, S_jb):
        """
        The correlation between S_ia and S_jb
        S_ia: array. Index: time or ensemble
        S_jb: array. Index: time or ensemble
        e.g. S_ia= S[:, i, a] from spinhist
        """
        n = len(S_ia)
        return np.dot(S_ia, S_jb) / n - np.average(S_ia) * np.average(S_jb)

    def C_R(self, R, S, a, b):
        ilist, jlist, Rjlist = self._gen_ij_R(R)
        ntime, nspin, _ = S.shape
        total = 0.0
        ilist, jlist = self._gen_ij_R(R)
        for i, j in zip(ilist, jlist):
            total += self.C_ia_jb(S_ia=S[:, i, a], S_jb=S[:, j, b])
        return total / ntime

    def sq(self, sa, q):
        total = 0.0
        # average over spin
        # here i=0
        ilist, jlist, Rlist = self._gen_ij_R(R=0)
        ilist = np.array(ilist)
        jlist = np.array(jlist)
        Rlist = np.array(Rlist)
        total = sq(jlist, Rlist, sa, q)
        #savg = np.average(sa)
        #for j, R in zip(jlist, Rlist):
        #    #print(np.dot(q, R))
        #    total += (sa[j] - savg) * np.exp(-2.0j * np.pi * np.dot(q, R))
        return total

    def sq_all(self, q):
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

    def sws(self):
        """
        S(w, index, direction) 
        """
        wlist = np.arange(self.S.shape[0]) / self.S.shape[0]
        savg = np.average(self.S, axis=0)
        S = self.S - savg[None, :, :]
        return sws(self.S, np.array(wlist, dtype=float))
        #return np.fft.fft(S, axis=0)

    def Swq(self, q):
        SwR = self.sws()
        savg = np.average(SwR, axis=1)
        SwR -= savg[:, None, :]
        nw, nspin, ndim = SwR.shape
        return SwR_to_Swq(SwR, np.array(self.Rvecs, dtype=float), np.array(q))

    def structure_factor_q(self, q, a, b):
        """
        This is a function of time.
        S(q,t)
        """
        ret = []
        sf0 = self.sq(self.S[:, a], q).conjugate()
        for s in self.S:
            ret.append(sf0 * self.sq(s[:, b], q))
        return ret

    def dynamic_structure_factor(self, q, a, b):
        """
        This is a function of w.
        """
        st = self.structure_factor_q(q, a, b)
        return np.fft.fft(st)

    def Ct_ia_jb(self, S_ia, S_jb, ndt):
        """
        The correlation between S_ia and S_jb
        """
        Scut_ia = S_ia[:-ndt]
        Scut_jb = S_jb[ndt:]
        n = len(S_ia)
        return np.dot(Scut_ia,
                      Scut_jb) / n - np.average(Scut_ia) * np.average(Scut_jb)

    def Ct_R(self, R, S, a, b, ndt):
        ilist = self.ilist
        jlist = self._gen_ij_R(R)
        ntime, nspin, _ = S.shape
        total = 0.0
        ilist, jlist = self._gen_ij_R(R)
        for i, j in zip(ilist, jlist):
            total += self.C_ia_jb(S_ia=S[:-ndt, i, a], S_jb=S[:ndt, j, b])
        return total / (ntime - ndt)

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
        #plt.imshow(Z, aspect='auto', interpolation='bicubic', origin='lower')
        plt.imshow(Z, aspect='auto', interpolation=None, origin='lower')
        plt.show()


def test():
    f = SpinHistParser("./mb.out_spinhist.nc")
    cf = CorrelationFuncions(
        sc_matrix=np.eye(3) * 12, rrange=(4, 4, 4), Rvecs=f.Rvec, S=f.S)

    qpoints = [(q, q, q) for q in np.arange(0.0, 0.51, 0.03)]
    cf.plot_spin_wave(qpoints=qpoints)

    for q in np.arange(0.0, 0.51, 0.05):
        p = cf.sq_all(q=[q, q, q])
        p = p - np.average(p, axis=0)[None, :]
        pc = np.correlate(p[:, 1][1:], p[:, 1][1:100], 'valid')
        plt.plot(myfft0(pc)[:300] + q * 10, label="%.2f" % q)
    plt.xlabel('1e12 Hz')
    plt.legend()
    plt.show()


test()
