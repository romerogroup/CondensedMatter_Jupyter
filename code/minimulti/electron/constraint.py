"""
Constraint of charge.
"""
import numpy as np


class SiteChargeConstraint(object):
    def __init__(
            self,
            bset,
            V,
            charge_dict,  # "{site: charge}"
            Vstep=8,
    ):
        self.bset = bset
        self.V = V
        self.Vstep=Vstep
        self.charge_dict = charge_dict
        self._prepare()
        self._lastdn=0.0
        self._dn=0.0
        self._dV=0.0
        self._Vmindn=0.0
        self._mindn=100
        self.downcount=0

    def _prepare(self):
        self._group_bset()

    def _group_bset(self):
        # correlated subspace: bset with same site and l.
        self._corr_group = {}  # site, l : (U, J, dim, index_bset)
        for site, b in self.bset.group_by_site():
            b = list(b)
            ids = np.array([bb.index for bb in b])
            charge = self.charge_dict[site]
            self._corr_group[site] = (ids, self.V, charge)

    def V_site(self, V, charge, rho):
        dn=np.trace(rho) - charge
        return dn, 2.0 * self.V * dn * np.eye(rho.shape[0])

    def update_V(self):
        ldn=self._lastdn
        dn=self._dn
        self._lastdn=self._dn

        if dn< self._mindn:
            self._mindn=dn
            self._Vmindn=self.V
        s=np.abs(dn)*self.Vstep
        if s>0.5:
            s=0.5
        if np.abs(dn)>0.01:
            if abs(dn)-abs(ldn)<-0:
                self.downcount+=1
            elif abs(dn)-abs(ldn)<0.01: # a little tolrence for fluctuation
                pass
            else:# abs(dn)-abs(ldn)<0.2:  #0.01~0.2
                if self.downcount>0:
                    self.downcount=0
                self.downcount-=1
            if self.downcount>2:
                self._dV+=s
            elif self.downcount<-1:
                self._dV =- s*0.8
            else:
                self._dV+=s*0.01*(np.random.random()-0.5)
        else:
            self._dV=0.0

        if self._dV>1.0:
            self._dV=1.0
        elif self._dV<-0.4:
            self._dV=-0.4

        self.V+=self._dV

        if self.V<0:
            self.V=self._Vmindn+0.3*np.random.random()
            self._dV=0.0
        print(self._dn, self.V)

    def V_and_E(self, rho):
        nbasis = len(self.bset)
        Vcon = np.zeros((nbasis, nbasis), dtype=complex)
        Econ = 0.0
        self.update_V()
        for site, val in self._corr_group.items():
            self._dn=0.0
            ids, V, charge = val
            ind = np.ix_(ids, ids)
            rhoi = rho[ind]
            dn, Vsite = self.V_site(V, charge, rhoi)
            self._dn+=np.abs(dn)
            Vcon[ind] = Vsite
            Econ -= np.sum(Vsite * rhoi)
        return Vcon, Econ
