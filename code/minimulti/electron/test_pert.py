from scipy.linalg import eigh
import numpy as np


def test():
    H0 = np.random.random([4, 4])
    H0 = H0 + H0.T

    V = np.random.random([4, 4])
    V = H0 + H0.T

    H = H0 + V

    dH = np.random.random([4, 4]) * 0.1
    dH = dH + dH.T

    evals0, evecs0 = eigh(H)
    print(evals0)
    evals, evecs = eigh(H + dH)
    print(evals)

    print(evals - evals0)

    print(np.diag(evecs0.T.conj().dot(dH).dot(evecs0)))
    print(np.diag(evecs0.T.conj()*evecs0*dH))

    m = evecs0.T.conj().dot(H0 + dH).dot(evecs0)
    n = evecs0.T.conj().dot(H0).dot(evecs0)
    print(np.diag(m - n))


test()
