#! /usr/bin/env python
import numpy as np
import sys
if sys.version_info < (2,6):
    MAX_EXP_ARGUMENT = np.log(1E90)
else:
    MAX_EXP_ARGUMENT = np.log(sys.float_info.max)


def fermi(e,mu,width,nspin=1):
    """
    Fermi function. f(e)= n/ (exp((e-mu)/width)+1)

    :param e: energy
    :param mu: fermi energy
    :param width: the smearing width
    :param nspin (optional, default 1): number of spins. if nspin =1 , n=2 else n=1.
    """
    args = (e-mu)/width
    # np.exp will return np.inf with RuntimeWarning if the input value
    # is too large, better to feed it nu.inf in the beginning
    args = np.where(args < MAX_EXP_ARGUMENT, args, np.inf)
    exps = np.exp(args)
    if nspin==1:
        return 2.0/(exps+1)
    elif nspin==2:
        return 1.0/(exps+1)

def cubic_monkhorst(n):
    """
    generate cubic monkhosrt kpoints using kpoints.x
    """
    if n%2==0:
        s=1
    else:
        s=0
    kp_input="""1


%s %s %s
%s %s %s
f
    """%(n,n,n,s,s,s)
    import os
    with open('/tmp/kin.txt','w') as myfile:
        myfile.write(kp_input)
    os.system("cat /tmp/kin.txt| kpoints.x")
    with open('mesh_k') as myfile:
        lines=myfile.readlines()
    nk=int(lines[0])
    kpts=[]
    w=0
    for i in range(1,nk+1):
        l=lines[i]
        t=np.array([float(x) for x in l.strip().split()])
        kpts.append(t[1:])
        w+=t[-1]
    kpts=np.array(kpts)
    kpts[:,-1]=kpts[:,-1]/w
    return kpts

def split_up_down_matrix(H):
    """
    a matrix H with the basis (1up,1down,2up,2down,...). If all H_iup_jdown=0, the matrix can be split into
    Hup, Hdown. Hup=H_iup_j_up, Hdown_idn_j_dn

    :param H: The matrix
    :Returns:
      Hup,Hdown
    """
    Hup=H[::2,::2]
    Hdn=H[1::2,1::2]
    return Hup,Hdn

def join_up_down_matrix(Hup,Hdn):
    """
    join Hup,Hdn -> H. basis (1up,1dn,2up,2dn,...)
    :param Hup,Hdn: Hup and Hdn are two matrix.
    """
    m,n=Hup.shape
    H=np.zeros((m*2,n*2))
    H[::2,::2]=Hup
    H[1::2,1::2]=Hdn
    return H
