import numpy as np
from scipy.linalg import qr, eigh, svd

def scdm(U, ortho=True):
    Q, R, piv=qr(U.T.conj(), mode='full', pivoting=True)
    piv = piv[0: U.shape[1]]

    if ortho:
        U= np.dot(U, Q)
        # An alternative method
        #Umn, Smn, Vmn = svd(U[piv])
        #Amn = Umn.dot(Vmn.T)
        #U = U.dot(Amn)
    else:
        U= np.dot(U, U[piv,: ].T.conj())
    return U, piv

def get_Hprim(H0, U):
    """
    H0: N*N matrix
    U: N*n matrix
    return n*n matrix
    """
    # n=U.shape[1]
    # Hprim=np.zeros((n, n), dtype='complex')
    # for i in range(n):
    #     for j in range(n):
    #         Hprim[i, j]=np.dot(U[:,i], H0).dot(U[:,j])
    Hprim = U.T.conj() @ H0 @ U
    return Hprim


def occupation_func(evals, mu, sigma, type='conduction'):
    if type=='conduction':
        return np.exp(-1.0 *(evals-mu)**2/sigma**2)
    elif type=="valence":
        return np.exp(-1.0 *(evals-mu)**2/sigma**2)

def frac_pos(wfn):
    pass

def conduction_wannier(evals, wfn,  kpts, nwann, positions, occupation_func, mu, sigma):
    """
    """
    # find gamma point
    ind_gamma=None
    for i, k in enumerate(kpts):
        if np.allclose(k, [0,0,0]):
            ind_gamma=i
    if ind_gamma is None:
        raise ValueError("Gamma not found in kpoints")

    # calculate occupation functions
    occ=occupation_func(evals, mu=-0.6, sigma=1.0)
    # calculate gamma columns
    wfn_gamma=wfn[ind_gamma]

    # select columns
    Q, R, piv= qr(wfn_gamma.dot(np.diag(occ[ind_gamma])), mode='full', pivoting=True)
    cols = piv[0: nwann]

    Amn=[]
    psi_scdm=[]
    for ik, k in enumerate(kpts):
        #phase=np.zeros(wfn.shape[1], dtype='complex')
        phase=np.zeros(wfn.shape[1], dtype='float')
        for i,pos in enumerate(positions):
            phase[i]=np.exp(-2j*np.pi* np.dot(k, pos))
        wfnk=wfn[ik]
        #project to selected basis and weight
        psi=np.diag(phase[cols]) @ wfnk[cols, :]@ np.diag(occ[ik])
        #psi= wfnk[:, cols].T#@ np.diag(occ[ik])
        #psi=np.diag(phase[cols]) @ wfnk[cols, :]
        #psi= wfnk[cols, :]
        psi=psi.T.conj()
        #psi=wfnk
        U, S, V=svd(psi, full_matrices=False)
        Amn_k=U @ (V.T.conj())

        Amn.append(Amn_k)
    return np.array(Amn) #, psi_scdm

def Amnk_to_Hk(Amn, psi, Hk0, kpts):
    Hk_prim=[]
    for ik, k in enumerate(kpts):
        wfn= psi[ik,:,:] @ Amn[ik,:,:]
        #wfn= eigh (Hk0[ik])[1]
        Hk_prim.append(wfn.T.conj() @ Hk0[ik,:,:] @ wfn)
        #Hk_prim.append(Hk0[ik] )
    return np.array(Hk_prim)



def Hk_to_Hreal(Hk, kpts, Rpts):
    nbasis=Hk.shape[1]
    nk=len(kpts)
    nR=len(Rpts)
    for iR, R in enumerate(Rpts):
        HR=np.zeros((nR, nbasis, nbasis))
        for ik, k in enumerate(kpts):
            phase=np.exp(2j*np.pi*np.dot(R, k))
            HR[iR]+=Hk[ik]*phase
    return HR



def test_Amn_to_Hk():
    H=np.random.rand(4, 4)
    H=H+H.T
    evals, evecs=np.linalg.eigh(H)
    Hprim=evecs.T.conj() @ H @evecs
    for i in range(4):
        for j in range(4):
            Hprim[i, j]=np.dot(evecs[:,i], H).dot(evecs[:,j])

    evals2=eigh(Hprim)[0]
    print(evals)
    d=Hprim
    print(d)
    d=(evecs.T.conj() @ H @evecs)
    print(d)
    print(evals2)
#test_Amn_to_Hk()

def test(ne=2):
    n=4
    H=np.random.rand(n, n)
    H+=H.T
    print(H)
    evals, evecs=eigh(H)
    print("evals")
    print(evals)
    print("evecs")
    print(evecs)
    U=evecs[:,:ne] #evec with lowest eval
    

    U, piv=scdm(U, ortho=True) # select column 
    print("Amn") # Amn
    print(U)
    print("piv")
    print(piv)  

    Hprim=get_Hprim(H, U)  # 
    print(Hprim)
    print(eigh(Hprim))

test()
