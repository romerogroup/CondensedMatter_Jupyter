import numpy as np
import numba
import matplotlib.pyplot as plt


@numba.njit(fastmath=True)
def myft(S, axis=-1):
    ns = S.shape[axis]
    wlist = np.arange(ns) / ns
    ret = np.zeros((ns), dtype=numba.complex128)
    for iw in range(ns):
        for i in range(ns):
            ret[iw] += S[i] * np.exp(-2.0j * np.pi * wlist[iw] * i)
    return ret


def test():
    S = np.arange(4.0)
    print(S)
    S = S - np.average(S, axis=-1)
    print(myft(S))
    print(np.fft.fft(S))


def zero_padding(x):
    return np.concatenate([np.zeros(len(x) * 2), x, np.zeros(len(x) * 2)])


def myfft0(x):
    x[0] = 0.0
    x-=np.average(x)
    win = np.blackman(len(x))
    #win =np.ones(len(x))
    f = np.fft.fft(zero_padding(x * win))
    #f=x
    N = len(f)
    ind_freq = np.arange(1, N // 2, dtype=int)
    r = np.abs(f[ind_freq])**2 + np.abs(f[-ind_freq])**2
    return r / np.sqrt(np.sum(r**2))


def myfft(x):
    from scipy.signal import welch
    #return np.log(welch(x, nperseg=len(x), scaling='spectrum'))
    return welch(x, nperseg=len(x), scaling='spectrum')


def corr(p, window=80, maxframe=500):
    pc = np.correlate(p[1:maxframe], p[1:window], 'valid')
    #plt.plot(myfft0(pc)[:])
    plt.plot(p, label='signal')
    plt.plot(pc, label='corr')
    plt.legend()
    plt.figure()
    plt.plot(myfft0(pc))
    pfft = np.fft.fft(p-np.average(p))
    pfft /= np.linalg.norm(pfft)
    plt.plot(pfft)
    plt.show()


def signal(N=1000, w=1.0/20):
    x=np.arange(N, dtype=float)
    y=np.cos(2*np.pi*w*x) * np.exp(-x/30)
    y+=np.random.random(N)*1
    return y

corr(signal())
