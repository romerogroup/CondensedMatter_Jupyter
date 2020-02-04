import numpy as np
from scipy.linalg import qr, eigh, svd
from scipy.special import erfc

from minimulti.scdm.rebase import rebase_wfn, rebase_H


class SCDMk(object):
    def __init__(
            self,
            evals,
            wfn,
            positions,
            kpts,
            nwann,
            occupation_func,
            anchors=None,
            anchor_kpoint=[0.0, 0.0, 0.0],
            scols=None,
            change_basis=None,
    ):
        self.evals = evals
        if change_basis is not None:
            self.wfn = np.zeros_like(wfn)
            for i, wf in enumerate(wfn):
                self.wfn[i] = rebase_wfn(wf, change_basis)
        else:
            self.wfn = wfn
        self.positions = positions
        self.kpts = kpts
        self.nwann = nwann

        self.occupation_func = occupation_func
        self.anchor_kpoint = anchor_kpoint

        # calculate occupation functions
        self.occ = self.occupation_func(self.evals)

        self.anchors = anchors
        # calculate cols from anchor points
        if anchors is not None:
            cols1 = self.set_anchor_points(anchors)
        else:
            cols1 = []
        if len(cols1) > self.nwann:
            raise ValueError("number of anchors larger than nwann")

        cols2 = self.get_anchor_cols()

        # add cols which are not already from anchor points.
        cols = []
        cols += cols1
        for col in cols2:
            if col not in cols:
                cols.append(col)
        cols = np.array(cols)
        self.cols = cols[:self.nwann]

        if scols is not None:
            self.cols = np.array(scols)

    def get_projection(self, mode1, k1, mode2, k2):
        phase1 = np.zeros(self.mode1.shape[0], dtype='complex')  # one per atom
        phase2 = np.zeros(self.mode2.shape[0], dtype='complex')
        for i, pos in enumerate(self.positions):
            phase1[i] = np.exp(-2j * np.pi * np.dot(k1, pos))
            phase2[i] = np.exp(-2j * np.pi * np.dot(k2, pos))
        psi1 = np.diag(phase1) @ mode1
        psi2 = np.diag(phase2) @ mode2
        return np.vdot(psi1)

    def get_projection_to_anchors(self):
        nband = self.wfn.shape[1]
        nkpts = len(self.kpts)
        projs = np.zeros((nkpts, nband), dtype='float')  # one proj per band
        # anchor point wavefunctions with phase removed
        psi_anchors = []
        for ia, k_anchor in enumerate(self.anchors):
            # find
            ind = None
            for ik, k in enumerate(self.kpts):
                if np.allclose(k, k_anchor):
                    ind = ik
                    break
            if ind is None:
                raise ValueError(
                    "The anchor kpoint %s is not in the list of k-points." %
                    (k_anchor))

            indices = self.anchors[k_anchor]
            for ind in indices:
                phase = np.zeros(self.wfn.shape[1], dtype='complex')
                for i, pos in enumerate(self.positions):
                    phase[i] = np.exp(-2j * np.pi * np.dot(k_anchor, pos))
                wfnk = self.wfn[ik, :, :]
                ## project to selected basis and weight
                psi = np.diag(phase) @ wfnk[:, ind]  #@ np.diag(self.occ[ik])
                psi_anchors.append(psi.copy())

        for ikpt, kpt in enumerate(self.kpts):
            phase = np.zeros(self.wfn.shape[1], dtype='complex')
            for i, pos in enumerate(self.positions):
                phase[i] = np.exp(-2j * np.pi * np.dot(kpt, pos))
            for iband in range(nband):
                psi_kb = np.diag(phase) @ self.wfn[ikpt, :, iband]
                for psi_a in psi_anchors:
                    p = np.vdot(psi_kb, psi_a)
                    projs[ikpt, iband] += np.real(np.conj(p) * p)
        return projs

    def set_anchor_points(self, anchors):
        """
        anchors: dictionary of {kpoints, indices}
        the kpoints and indices should be tuples.
        eg. {(.5,.5,.5): (1, 2)}, means the 2nd and 3rd eigenstate at the kpoint (.5,.5,.5)
        Mulitiple anchor points for more than one kpoints are fine,
        the maixmum number should be no more than nwann.
        """
        cols = set()
        for ia, k_anchor in enumerate(anchors):
            # find
            ind = None
            #print(k_anchor)
            for ik, k in enumerate(self.kpts):
                if np.allclose(k, k_anchor):
                    ind = ik
            if ind is None:
                raise ValueError(
                    "The anchor kpoint %s is not in the list of k-points." %
                    (k_anchor))

            indices = anchors[k_anchor]
            phase = np.zeros(self.wfn.shape[1], dtype='complex')
            for i, pos in enumerate(self.positions):
                phase[i] = np.exp(-2j * np.pi * np.dot(k_anchor, pos))
            wfnk = self.wfn[ik]
            ## project to selected basis and weight
            psi = np.diag(phase) @ np.diag(
                self.occ[ik]) @ wfnk[:, indices]  #@ np.diag(self.occ[ik])
            Q, R, piv = qr(psi.T.conj(), mode='full', pivoting=True)
            piv = piv[0:psi.shape[1]]
            cols = cols.union(piv)
        return list(cols)

    def get_anchor_cols(self):
        ind_gamma = None
        for i, k in enumerate(self.kpts):
            if np.allclose(k, self.anchor_kpoint):
                ind_gamma = i
        if ind_gamma is None:
            raise ValueError("Gamma not found in kpoints")
        # calculate gamma columns
        wfn_gamma = self.wfn[ind_gamma]

        # select columns
        Q, R, piv = qr(
            wfn_gamma.dot(np.diag(self.occ[ind_gamma])),
            mode='full',
            pivoting=True)
        self.cols = piv[0:self.nwann]
        return self.cols

    def get_Amn(self, anchor_only=False, use_proj=False):
        self.Amn = []
        if use_proj:
            self.projs = self.get_projection_to_anchors()
        for ik, k in enumerate(self.kpts):
            phase = np.zeros(self.wfn.shape[1], dtype='complex')
            #phase=np.zeros(self.wfn.shape[1], dtype='float')
            for i, pos in enumerate(self.positions):
                phase[i] = np.exp(2j * np.pi * np.dot(k, pos))

            wfnk = self.wfn[ik]
            ## project to selected basis and weight
            #psi =  np.diag(phase) @ wfnk[:, :]  @np.diag(self.projs[ik]) @ np.diag(
            #        self.occ[ik])

            #Q, R, piv= qr(psi, mode='full', pivoting=True)
            #self.cols = piv[0: self.nwann]

            if anchor_only:
                psi = np.diag(phase[self.cols]) @ wfnk[self.cols, :]
            elif use_proj:
                psi = np.diag(
                    phase[self.cols]) @ wfnk[self.cols, :]  # @ np.diag(
                #self.occ[ik])@np.diag(self.projs[ik])

            else:
                psi = np.diag(phase[self.cols]) @ wfnk[self.cols, :] @ np.diag(
                    self.occ[ik])  #@np.diag(self.projs[ik])

            psi = psi.T.conj()
            U, S, V = svd(psi, full_matrices=False)
            Amn_k = U @ (V.T.conj())
            self.Amn.append(Amn_k)
        return np.array(self.Amn)

    def H0_to_H(
            self,
            Amn,
    ):
        Hk_prim = []
        for ik, k in enumerate(self.kpts):
            wfn = self.wfn[ik, :, :] @ Amn[ik, :, :]
            Hk_prim.append(wfn.T.conj() @ Hk0[ik, :, :] @ wfn)
        return Hk_prim


def occupation_func(ftype=None, mu=0.0, sigma=1.0):
    if ftype == None or ftype == "unity":

        def func(x):
            return np.ones_like(x, dtype=float)

    if ftype == 'Fermi':

        def func(x):
            return 0.5 * erfc((x - mu) / sigma)
    elif ftype == 'Gauss':

        def func(x):
            return np.exp(-1.0 * (x - mu)**2 / sigma**2)
    elif ftype == 'window':

        def func(x):
            return 0.5 * erfc((x - mu) / 0.01) - 0.5 * erfc((x - sigma) / 0.01)
    else:
        raise NotImplementedError("function type %s not implemented." % ftype)
    return func


def Amnk_to_Hk(Amn, psi, Hk0, kpts):
    Hk_prim = []
    for ik, k in enumerate(kpts):
        wfn = psi[ik, :, :] @ Amn[ik, :, :]
        hk = wfn.T.conj() @ Hk0[ik, :, :] @ wfn
        #hk = wfn.T @ Hk0[ik, :, :] @ wfn
        Hk_prim.append(hk)
    return np.array(Hk_prim)


def Hk_to_Hreal(Hk, kpts, Rpts):
    nbasis = Hk.shape[1]
    nk = len(kpts)
    nR = len(Rpts)
    for iR, R in enumerate(Rpts):
        HR = np.zeros((nR, nbasis, nbasis))
        for ik, k in enumerate(kpts):
            phase = np.exp(2j * np.pi * np.dot(R, k))
            HR[iR] += Hk[ik] * phase
    return HR


def test():
    a = np.arange(-5, 5, 0.01)
    import matplotlib.pyplot as plt
    plt.plot(a, 0.5 * erfc((a - 0.0) / 0.5))
    plt.show()
