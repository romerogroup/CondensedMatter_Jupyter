from mytb import mytb
#from minimulti.scdm.scdm import *
import numpy as np
from minimulti.scdm.scdmk import SCDMk, occupation_func, Amnk_to_Hk
from scipy.linalg import eigh
import matplotlib.pyplot as  plt

def test():
    orb=[(0, 0, 0), (0,0,0.25), (0,0, 0.5)]
    tb = mytb(3, 3, lat=np.eye(3), orb=orb)
    tb.set_onsite(1, 2)
    tb.set_onsite(0, 1)
    tb.set_onsite(0, 0)
    tb.set_hop(1, 0, 1, (0, 0, 0))
    tb.set_hop(1, 1, 0, (0, 0, 0))
    #tb.set_hop(1, 1, 0, (0, 0, 1))
    tb.set_hop(1, 0, 2, (0,0,-1))
    tb.set_hop(1, 2, 0, (0,0,1))
    tb.set_hop(1, 2, 1, (0,0,0))
    tb.set_hop(1, 1, 2, (0,0,0))

    kpath = [(0, 0, x) for x in np.arange(0, 2, 0.02)]
    Hk=np.array([tb.make_hamk(k) for k in kpath])
    evals, evecs = tb.solve_all(kpath)
    wfn=evecs.copy()

    for i in range(evals.shape[1]):
        plt.plot(evals[:,i])

    kpts=kpath
    scdmk=SCDMk( evals=evals, wfn=evecs, positions=orb, kpts=kpath, nwann=2,
            occupation_func=occupation_func(ftype='Gauss', mu=-2.0, sigma=3.01),
            anchors=None)
    Amn=scdmk.get_Amn()

    Hk_prim = Amnk_to_Hk(Amn, wfn, Hk, kpts)
    print(Hk_prim.shape)

    evals=[]
    for ik , k in enumerate(kpath):
        ev, _ = eigh(Hk_prim[ik])
        evals.append(ev)

    evals=np.array(evals)

    #print(evals)
    for i in range(evals.shape[1]):
        plt.plot(evals[:,i], marker='.')
    plt.show()


test()
