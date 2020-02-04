import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy as sp

def schordinger_1d_solver(V, x):
    hbar=1
    m=1
    n=len(V)
    dx=x[1]-x[0]
    Kmat=np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1),-1)-np.diag(2.0*np.ones(n), 0)
    Kmat[0,n-1]=-1
    #Kmat[n-1,0]=-1
    H=-hbar*hbar/(2.0*m) * (1.0/dx**2)* Kmat +np.diag(V)
    evals, evecs = sp.linalg.eigh(H)
    return evals, evecs/np.sqrt(dx)

def pot():
    a=4
    N=200
    Va=30
    x=np.linspace(0.0,a, N, endpoint=False)
    V=np.zeros(N)
    V=Va*signal.square(2*np.pi*(x+0.25))
    V[0:N//4]=Va
    V[-N//4:]=Va
    return V, x


def plot():
    V, x = pot()
    plt.plot(x, V/30)

    i=4
    evals, evecs= schordinger_1d_solver(V, x)
    #plt.plot(x,evecs[:,i]*evecs[:,i].conj())

    a=10
    phase=np.sin((i+1)*np.pi*(x/a))
    plt.plot(x, phase/1.5)
    plt.plot(x,np.real(evecs[:,i]), alpha=0.3)
    plt.show()

plot()


