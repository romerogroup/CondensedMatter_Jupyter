import numpy as np
from scipy.linalg import eigh


def rebase_mat(basis0, basis1):
    """
    matrix which change basis0 to basis1
    """
    return np.linalg.inv(basis1) @ basis0


def rebase_wfn(wfn, basis):
    """
    wfn: (nbasis, nband)
    basis: (nbasis, nbasis)
    """
    return np.linalg.inv(basis) @ wfn


def rebase_H(H, basis):
    """
    Change a matrix to a new basis set.
    """
    return np.linalg.inv(basis) @ H @ basis

def inv_rebase_H(H, basis):
    """
    if H is rebased to basis form H0, get H0
    One application of this is to get the matrix from its eigenvectors and eigenvalues
    where the eigenvalue diagnonal matrix is just the H0 in the basis of eigenvectors.
    """
    return basis @ H @ np.linalg.inv(basis)

def eigen_to_mat(evals, evecs):
    return evecs @ np.diag(evals) @ np.linalg.inv(evecs)



def test():
    # generate random matrix
    mat = np.random.rand(3, 3)
    mat = mat + mat.T

    # evals and evecs
    evals, wfn = eigh(mat)
    print(evals)
    print(wfn)

    # change basis to eigenstate
    H = rebase_H(mat, wfn)
    eval0 = rebase_wfn(wfn, wfn)

    # evals and evecs in eigenspace
    evals1, wfn1 = eigh(H)
    # should be the same
    print(evals1)
    # should be I
    print(wfn1)


if __name__ == '__main__':
    test()
