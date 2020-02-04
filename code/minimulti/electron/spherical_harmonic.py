import numpy as np
from scipy.misc import factorial as fact
from itertools import product
from functools import lru_cache


# Angular matrix elements of particle-particle interaction
# (2l+1)^2 ((l 0) (k 0) (l 0))^2 \sum_{q=-k}^{k} (-1)^{m1+m2+q}
# ((l -m1) (k q) (l m3)) ((l -m2) (k -q) (l m4))
def angular_matrix_element(l, k, m1, m2, m3, m4):
    r"""
    Calculate the angular matrix element
    .. math::
       (2l+1)^2
       \begin{pmatrix}
            l & k & l \\
            0 & 0 & 0
       \end{pmatrix}^2
       \sum_{q=-k}^k (-1)^{m_1+m_2+q}
       \begin{pmatrix}
            l & k & l \\
         -m_1 & q & m_3
       \end{pmatrix}
       \begin{pmatrix}
            l & k  & l \\
         -m_2 & -q & m_4
       \end{pmatrix}.
    Parameters
    ----------
    l : integer
    k : integer
    m1 : integer
    m2 : integer
    m3 : integer
    m4 : integer
    Returns
    -------
    ang_mat_ele : scalar
                  Angular matrix element.
    """
    ang_mat_ele = 0
    for q in range(-k, k + 1):
        ang_mat_ele += three_j_symbol((l,-m1),(k,q),(l,m3))* \
                three_j_symbol((l,-m2),(k,-q),(l,m4))* \
                (-1.0 if (m1+q+m2) % 2 else 1.0)
    ang_mat_ele *= (2 * l + 1)**2 * (three_j_symbol((l, 0), (k, 0), (l, 0))**2)
    return ang_mat_ele


# Wigner 3-j symbols
# ((j1 m1) (j2 m2) (j3 m3))
def three_j_symbol(jm1, jm2, jm3):
    r"""
    Calculate the three-j symbol
    .. math::
       \begin{pmatrix}
        l_1 & l_2 & l_3\\
        m_1 & m_2 & m_3
       \end{pmatrix}.
    Parameters
    ----------
    jm1 : tuple of integers
          (j_1 m_1)
    jm2 : tuple of integers
          (j_2 m_2)
    jm3 : tuple of integers
          (j_3 m_3)
    Returns
    -------
    three_j_sym : scalar
                  Three-j symbol.
    """
    j1, m1 = jm1
    j2, m2 = jm2
    j3, m3 = jm3

    if (m1 + m2 + m3 != 0 or m1 < -j1 or m1 > j1 or m2 < -j2 or m2 > j2
            or m3 < -j3 or m3 > j3 or j3 > j1 + j2 or j3 < abs(j1 - j2)):
        return .0

    three_j_sym = -1.0 if (j1 - j2 - m3) % 2 else 1.0
    three_j_sym *= np.sqrt(fact(j1+j2-j3)*fact(j1-j2+j3)* \
            fact(-j1+j2+j3)/fact(j1+j2+j3+1))
    three_j_sym *= np.sqrt(fact(j1-m1)*fact(j1+m1)*fact(j2-m2)* \
            fact(j2+m2)*fact(j3-m3)*fact(j3+m3))

    t_min = max(j2 - j3 - m1, j1 - j3 + m2, 0)
    t_max = min(j1 - m1, j2 + m2, j1 + j2 - j3)

    t_sum = 0
    for t in range(t_min, t_max + 1):
        t_sum += (-1.0 if t % 2 else 1.0)/(fact(t)*fact(j3-j2+m1+t)* \
                fact(j3-j1-m2+t)*fact(j1+j2-j3-t)*fact(j1-m1-t)*fact(j2+m2-t))

    three_j_sym *= t_sum
    return three_j_sym


def U_J_to_F(l, U, J):
    F = np.zeros((l + 1), dtype=np.float)
    F[0] = U
    if l == 0:
        pass
    elif l == 1:
        F[1] = J * 5.0
    elif l == 2:
        F[1] = J * 14.0 / (1.0 + 0.625)
        F[2] = 0.625 * F[1]
    elif l == 3:
        F[1] = 6435.0 * J / (286.0 + 195.0 * 0.668 + 250.0 * 0.494)
        F[2] = 0.668 * F[1]
        F[3] = 0.494 * F[1]
    else:
        raise ValueError("  l should be 0,1,2,3 for s, p, d, and f")
    return F


def unitary_transform_coulomb_matrix(a, u):
    '''Perform a unitary transformation (u) on the Coulomb matrix (a).
    '''
    a_ = np.asarray(a).copy()
    m_range = range(a.shape[0])
    for i, j in product(m_range, m_range):
        a_[i, j, :, :] = u.T.conj().dot(a_[i, j, :, :].dot(u))
    a_ = a_.swapaxes(0, 2).swapaxes(1, 3)
    for i, j in product(m_range, m_range):
        a_[i, j, :, :] = u.T.conj().dot(a_[i, j, :, :].dot(u))
    return a_


s2 = np.sqrt(2.0)
Ylm_to_eg_t2g = np.array(
    [[0, 1 / s2, 1j / s2, 0, 0], [0, 0, 0, -1j / s2, 1 / s2], [1, 0, 0, 0, 0],
     [0, 0, 0, 1j / s2, 1 / s2], [0, 1 / s2, -1j / s2, 0, 0]])

def validate_Ylm_to_eg_t2g():
    Vdagger = V_mat.transpose().conjugate()
    Vinv = np.linalg.inv(V_mat)
    assert np.isclose(Vdagger, Vinv).all()

@lru_cache(maxsize=8)
def V_Slater(l, U, J, to_eg_t2g=False):
    U_matrix = np.zeros(
        (2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1), dtype=np.complex)
    F = U_J_to_F(l, U, J)
    m_range = range(-l, l + 1)
    for n, F in enumerate(F):
        k = 2 * n
        for m1, m2, m3, m4 in product(m_range, m_range, m_range, m_range):
            U_matrix[m1+l, m3+l, m2+l, m4+l] += \
                    F * angular_matrix_element(l,k,m1,m2,m3,m4)
    if to_eg_t2g:
        U_matrix = unitary_transform_coulomb_matrix(U_matrix, Ylm_to_eg_t2g.T)
    return U_matrix


def test():
    U = 0
    J = 1
    l = 2
    U = V_Slater(l=l, U=U, J=J)
    A = np.zeros((2 * l + 1, 2 * l + 1), dtype=float)
    for i in range(5):
        for j in range(5):
            A[i, j] = U[i, j, j, i]

    B = np.zeros((2 * l + 1, 2 * l + 1), dtype=float)
    for i in range(5):
        for j in range(5):
            B[i, j] = U[i, i, j, j]
    print("=====Slater")

    UU = unitary_transform_coulomb_matrix(U, Ylm_to_eg_t2g)
    B = np.zeros((2 * l + 1, 2 * l + 1), dtype=float)
    for i in range(5):
        for j in range(5):
            B[i, j] = UU[i, i, j, j]

    A = np.zeros((2 * l + 1, 2 * l + 1), dtype=float)
    for i in range(5):
        for j in range(5):
            A[i, j] = UU[i, j, j, i]

    print(B)
    print(B - A + 1.0 - np.eye(5))

test()
