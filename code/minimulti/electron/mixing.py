#!/usr/bin/env python

"""
Mixing methods.
"""

from collections import deque
import numpy as np
import numpy.linalg


class Mixer(object):
    """
    mixer, default is a linear mixer. rho_new=(1-beta)*rho_old+beta*rho_new
    """
    def __init__(self,beta=0.1,nmaxold=5,weight=50.0,dotprod=None):
        self.beta=beta
        self.nmaxold=nmaxold
        self.weight=weight
        self.history_rho=deque()


    def mix(self,new_rho):
        """
        :param new_rho: new density.
        """
        # if new_rho is not a array
        new_rho=np.array(new_rho)
        rho_shape=new_rho.shape
        new_rho=new_rho.flatten()

        if len(self.history_rho)!=0:
            new_rho=self.history_rho[-1]*(1.0-self.beta)+new_rho*self.beta
        if len(self.history_rho)<=self.nmaxold:
            self.history_rho.append(new_rho)
        else:
            self.history_rho.append(new_rho)
            self.history_rho.popleft()

        new_rho=new_rho.reshape(rho_shape)
        return new_rho

class Pulay_Mixer(Mixer):
    """
    Pulay mixer. see http://vergil.chemistry.gatech.edu/notes/diis/node2.html
    """
    def __init__(self,beta=0.1,nmaxold=5,weight=50.0,dotprod=None):
        super(Pulay_Mixer,self).__init__(beta=beta,nmaxold=nmaxold,weight=weight,dotprod=dotprod)
        self.history_delta_rho=deque()

    def mix(self,new_rho):
        """
        Pulay mixing. The first steps are simple linear mixing.
        :param new_rho: new density
        :returns: the mixed new density.
        """
        new_rho=np.array(new_rho)
        rho_shape=new_rho.shape
        new_rho=new_rho.flatten()


        if len(self.history_rho)==0:
            pass
        elif len(self.history_rho)<=self.nmaxold:
            new_rho=self.history_rho[-1]*(1.0-self.beta)+new_rho*self.beta
            self.history_rho.append(new_rho)
            self.history_delta_rho.append(new_rho-self.history_rho[-1])
        else:
            self.history_rho.append(new_rho)
            self.history_delta_rho.append(new_rho-self.history_rho[-1])
            dimBp=self.nmaxold+2
            Bp=np.ones((dimBp,dimBp))
            for i in range(dimBp-1):
                for j in range(i,dimBp-1):
                    Bp[i,j]=np.dot(self.history_delta_rho[i],self.history_delta_rho[j])
                    Bp[j,i]=Bp[i,j]
            Bp[:,-1]=-1
            Bp[-1,:]=-1
            Bp[-1,-1]=0

            b=np.zeros(dimBp)
            b[-1]=-1.0
            #Cp is the C1,C2,...Cn,lambda, where lambda is the lagrange multiplier
            Cp=numpy.linalg.solve(Bp,b)
            new_rho=new_rho*Cp[-2]
            for i in range(dimBp-1):
                new_rho+=self.history_rho[i]*Cp[i]

        if len(self.history_rho)>self.nmaxold:
            self.history_delta_rho.popleft()
            self.history_rho.popleft()
            self.history_delta_rho.pop()

        new_rho=new_rho.reshape(rho_shape)
        return new_rho
