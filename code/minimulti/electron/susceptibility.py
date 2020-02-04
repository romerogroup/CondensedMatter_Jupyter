import numpy as np
import numba
from ase.dft.kpoints import monkhorst_pack
from minimulti.electron.occupations import Occupations


#@numba.njit(fastmath=True)
def find_index_k(kpts, q):
    kpts_p = np.mod(kpts + 1e-9, 1)
    jkpts = np.zeros(len(kpts), dtype=int)
    for ik, k in enumerate(kpts):
        for ikp, kp in enumerate(kpts_p):
            if np.allclose(np.mod(k + q + 1e-9, 1), kp, atol=1.e-3):
                jkpts[ik] = ikp
    if len(jkpts) != len(set(jkpts)):
        raise ValueError(
            "Cannot find all the k+q point. Please check if the k-mesh and the q-point is compatible."
        )
    return jkpts


@numba.njit(fastmath=True)
def calc_chi0(evals,
              evecs,
              kpts,
              jkpts,
              occ,
              q,
              omega=0.0,
              eps=0.05,
              weight=None):
    # :param eigenvecs: the eigenvec matrix. indexes are [band,kpt,orb,spin] or [band,kpt,orb]
    # evals: band, kpt
    # chi(q)
    chi = 0
    nband, nkpt, norb = evecs.shape
    for ikpt in range(nkpt):
        kpt = kpts[ikpt, :]
        jkpt = jkpts[ikpt]
        for ib in range(nband):
            for jb in range(nband):
                delta_occ = occ[ib, ikpt] - occ[jb, jkpt]
                if weight is not None:
                    delta_occ *= weight[ib, ikpt] * weight[jb, jkpt]
                if abs(delta_occ > 1e-5):
                    #chi_ij = np.vdot(evecs[ib, ikpt, :],
                    #                 evecs[jb, jkpt, :])
                    #chi_ij=chi_ij.conjugate()*chi_ij
                    #chi_ij=chi_ij**2

                    chi_ij = 0.0j
                    for io in range(norb):
                        for jo in range(norb):
                            chi_ij += evecs[ib, ikpt,
                                            io] * evecs[ib, ikpt, jo].conjugate(
                                            ) * evecs[jb, jkpt,
                                                      jo] * evecs[jb, jkpt,
                                                                  io].conjugate(
                                                                  )
                    #print("chi_ij:", ib, jb, chi_ij)
                    #print(delta_occ)
                    chi_ij *= delta_occ / (
                        evals[ib, ikpt] - evals[jb, jkpt] - omega - eps * 1j)
                    chi += chi_ij
    return np.real(chi)/ nkpt


def calc_chi0_list(model, qlist, omega=0.0, kmesh=None, supercell_matrix=None):
    if kmesh is not None:
        k_list = monkhorst_pack(kmesh)
    else:
        k_list = model._kpts
    evals, evecs = model.solve_all(
        eig_vectors=True, k_list=k_list, convention=1, zeeman=False)

    if supercell_matrix is not None:
        weight = model.unfold(
            k_list, sc_matrix=supercell_matrix, evals=evals, evecs=evecs)
        k_list = [np.dot(k, supercell_matrix) for k in k_list]
        qlist = [np.dot(q, supercell_matrix) for q in qlist]
    else:
        weight = None

    occ = Occupations(
        model._nel, model._width, model._kweights, nspin=model._actual_nspin)
    occupations = occ.occupy(evals)

    ret = []
    for iq, q in enumerate(qlist):
        jkpts = find_index_k(k_list, q)
        chi_q = calc_chi0(
            evals,
            evecs,
            k_list,
            jkpts,
            occupations,
            q,
            omega=omega,
            weight=weight)
        ret.append(chi_q)
    return np.array(ret)


def test_find_index_k():
    kpts = monkhorst_pack(size=[1, 2, 3])
    q = np.array((0., 0.5, 1.0 / 3))
    print(np.mod(kpts + 1e-9, 1))
    print("k+q:")
    print(kpts + q + 1e-9)
    js = find_index_k(kpts, q)
    print(kpts[js])


if __name__=='__main__':
    test_find_index_k()
