import numpy as np
from scipy.linalg import qr, eig, svd, eigh
import matplotlib.pyplot as plt


def scdm(U, ortho=True, piv=None):
    """
    U: eigenvectors, each column is a eigenvector
    """
    if piv is None:
        Q, R, piv = qr(U.T, mode='full', pivoting=True)
        piv = piv[0:U.shape[1]]
    else:
        piv=np.array(piv, dtype=int)

    if ortho:
        #U= np.dot(U, Q)
        Umn, Smn, Vmn = svd(U[piv])
        Amn = Umn.dot(Vmn.T)
        U = U.dot(Amn)
    else:
        U[piv, :].T
        U = np.dot(U, U[piv, :].T)
    return U, piv, Amn

def weight_func(x, mu=0, sigma=2.4):
    return np.exp(-1.0 *(x-mu)**2/sigma**2)

def plot_weight_func():
    x=np.arange(-2, 2, 0.1)
    plt.plot(x, weight_func(x))
    plt.show()

#plot_weight_func()


def select_with_weight(evals, wfn, nwann):
    # select columns
    occ=weight_func(evals)
    Q, R, piv= qr(wfn.dot(np.diag(occ)), mode='full', pivoting=True)
    cols = piv[0: nwann]
    return cols

def test_select_with_weight():
    n = 4
    nwann=1
    H = np.random.rand(n, n)
    H += H.T
    evals, evecs = eigh(H)
    print("evals:", evals)
    cols=select_with_weight(evals, evecs, nwann )
    print(cols)

#test_select_with_weight()


def get_Hprim(H0, U):
    n = U.shape[1]
    Hprim = np.zeros((n, n), dtype='complex')
    #for i in range(n):
    #    for j in range(n):
    #        Hprim[i, j]=np.dot(U[:,i], H0).dot(U[:,j])
    Hprim = U.T @ H0 @ U
    return Hprim


def get_Hprim_from_Amn(U, H0, Amn):
    #return Amn.dot(H0).dot(Amn.T)
    print("UHU", U.T @ H0 @ U)
    # Note U.T @H0 @ U is the eigenvalue diagonal matrix
    return (Amn.T @ U.T) @ H0 @ (U @ Amn)



def test():
    n = 4
    H = np.random.rand(n, n)
    H += H.T
    print(H)
    evals, evecs = eigh(H)
    print("evals")
    print(evals)
    piv=[0,1,2,3]
    U = evecs[:, piv]
    U0 = U

    print("U:", U)
    U, piv, Amn = scdm(U, ortho=True, piv=None)
    print("Amn", Amn)

    print("U")
    print(U)
    print(piv)

    #Hprim=get_Hprim(H, U)
    Hprim = get_Hprim_from_Amn(U0, H, Amn)
    print(Hprim)
    print("new evals")
    print(eigh(Hprim))


test()
