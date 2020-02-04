from myTB import mytb
import numpy as np

class mytb_plus_U_one_orb(mytb):
    """
    Tight binding plus U. For one orbital model only.
    """
    def __init__(self,dim_k,dim_r,lat,orb,per=None,nspin=1,nel=None,width=0.2,verbose=True,fix_spin=False):
        mytb.__init__(dim_k,dim_r,lat,orb,per=per,nspin=nspin,nel=nel,width=width,verbose=verbose,fix_spin=fix_spin)
        self._U=None
        self._onsite_en=np.zeros(len(orb))

    def set_U(self,U):
        """
        set the U.
        """
        if isinstance(U,int) or isinstance(U,float):
            self._U=np.array([U]*self._norb)
        else:
            self._U=U

    def save_H0(self):
        """
        save the current hamiltonian (real space) as self._H0
        """
        raise NotImplementedError


    def set_onsite(self,onsite_en,ind_i=None,mode='set'):
        """
        set onsit_energies. self._onsite_en is the onsite en.
        """
        if ind_i is None:
            if mode=='set' or mode=='reset':
                self._onsite_en=onsite_en
            elif mode=='add':
                self._onsite_en+=onsite_en
            else:
                raise ValueError('mode wrong')
        else:
            if mode=='set':
                self._onsite_en=np.zeros((self._norb,self._nspin**2))
                self._onsite_en[ind_i]=onsite_en
            elif mode=='reset':
                self._onsite_en[ind_i]=onsite_en
            else:
                self._onsite_en[ind_i]+=onsite_en

        mytb.set_onsite(self,onsite_en,ind_i=ind_i,mode=mode)


    def set_onsite_with_U(self,onsite_en=None,mode='reset'):
        """
        set onsite energy with U
        """
        if onsite_en is None:
            if len(self._onsite_en.shape)==1:
                onsite_en=self._onsite_en
            else:
                onsite_en=self._onsite_en[:,0]
        n=self._nbar
        e=np.zeros((self._norb,4))
        #print self._norb,self._nspin
        e[:,0]=onsite_en+self._U*(n[:,1]+n[:,0])/2.0
        e[:,3]=self._U*(n[:,1]-n[:,0])/2.0
        print "U:",self._U
        print "n:", n
        print "e:",list(e)
        self.set_onsite(list(e),mode=mode)

    def solve_mean_field_n(self):
        """
        the full solve process with mean field approximation.

        self.occupations
        """
        print self._nbar,self._old_nbar
        self._old_nbar=self._nbar.copy()
        self.set_onsite_with_U(mode='reset')
        self.get_occupations(nel=self._nel,width=self._width,refresh=True)
        self.get_orbital_occupations()
        beta=0.1
        self._nbar=self._nbar*beta+self._old_nbar*(1-beta)
        i=0
        print self._nbar,self._old_nbar
        dn=np.max(np.abs(self._old_nbar-self._nbar))
        while dn>self._eps:
            i+=1
            if self._verbose:
                print('Iter%s: dn=%s \n%s\n'%(i,dn,self._nbar))
            self._old_nbar=self._nbar.copy()
            self.set_onsite_with_U(mode='reset')
            self.get_occupations(nel=self._nel,width=self._width,refresh=True)
            self.get_orbital_occupations()
            dn=np.max(np.abs(self._old_nbar-self._nbar))
            self._nbar=self._nbar*beta+self._old_nbar*(1-beta)
        return self._occupations

    def set_initial_nbar(self,nbar=None,order=None):
        """
        set the initial nbar

        :param nbar: set nbar to nbar. if nspin=2, ndarray((norb,2)), else ndarray(norb)
        :param order: 'FM'|'PM'|'AFM'|'random' (default)
        """
        if nbar is not None:
            self._nbar=nbar
        elif order=='FM':
            self._nbar=np.array([[1.0*self._nel/self._norb,0]]*self._norb)
        elif order=='PM':
            self._nbar=np.array([[0.5*self._nel/self._norb,0.5*self._nel/self._norb]]*self._norb)
        elif order=='AFM':
            #self._nbar=np.array([[1.0*self._nel/self._norb,0],[0,self._nel/self._norb]]*self._norb/2)
            self._nbar=np.zeros((self._norb,2))
            print("nbar shape: ",self._nbar.shape)
            self._nbar[::2,0]=np.array([[1.0*self._nel/self._norb]]*(self._norb/2))
            self._nbar[::2,1]=np.array([[1.0*self._nel/self._norb]]*(self._norb/2))

        elif order=='random':
            self._nbar=np.random.rand(self._norb,self._nspin)
            self._nbar=self._nbar*self._nel/np.sum(self._nbar)
        else:
            raise ValueError("order should be FM/PM/AFM/random")
        self._old_nbar=self._nbar.copy()
