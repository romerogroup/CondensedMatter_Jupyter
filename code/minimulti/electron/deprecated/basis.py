#!/usr/bin/env python
from collections import OrderedDict, defaultdict
import numpy as np
from ase_utils.symbol import symbol_number
import itertools
import copy


class basis(object):
    """
    Basis class. A basis object has the attributes: site, label, spin, id, site_name, element, position.
    :param site: The site index.
    :param label: can be anythong hashable. so that the basis can be uniquely defined.Some examples are (3,2,-1) or '3dxy'
    :param spin: can be None | 'UP' |'DOWN'
    :param site_name: (optional) the name of the site. eg. 'Fe3' It is not a functional part of the basis.
    :param element: (optional) which element is on the site. eg. 'Fe'
    :param position: (optional) the postion of the basis. Use this if the basis set are wannier functions.
    id: each basis is given a id.
    """

    def __init__(self,
                 site,
                 label,
                 spin,
                 site_name=None,
                 element=None,
                 position=None):
        self.id = None
        self.site = site
        self.label = label
        self.spin = spin
        self.site_name = site_name
        self.element = element
        self.position = position

    def set_id(self, id):
        self.id = id

    def copy(self):
        return copy.copy(self)

    def attr(self):
        return (self.site, self.label, self.spin)

    def __hash__(self):
        # define __hash_- and __eq__ so that it can be used as dict keys. also ignore the id when used.
        return hash(self.attr())

    def __eq__(self, other):
        return self.attr() == other.attr()

    def __repr__(self):
        return "Basis: %s with spin %s at site %s. Id= %s" % (self.label,
                                                              self.spin,
                                                              self.site,
                                                              self.id)


class basis_set(object):
    """
    the set of basis.

    :param basis_list:
        a iterable object. each element is a basis object. Note the the id of the basis will be set according to the sequence of its apperance.
    """

    def __init__(self, basis_list=None):
        """
        """
        # dict: attr-id pairs
        self.len = 0
        self.dict = OrderedDict()
        #self.basis_attr=set()
        if basis_list is not None:
            for basis in basis_list:
                self.add_basis(basis)
        self.basis_at_site = defaultdict(list)

    def add_basis(self, basis):
        """
        add a basis to the basis_set

        :param basis: The basis to be added.
        """
        nbasis = basis.copy()
        if nbasis not in self.dict:
            id = len(self.dict)
            self.dict[nbasis] = id
            nbasis.set_id(id)
            self.basis_at_site[nbasis.site].append(nbasis)
            self.len += 1

    def __len__(self):
        """
        return the length
        """
        return self.len

    def get_id(self, basis):
        """
        get the id of a basis.
        """
        return self.dict[basis]

    def get_basis(self, id=None):
        """
        get the basis with id id. if id is None, return the list of all basis.
        :param id: the id of the basis. default is None
        """
        if id is None:
            return self.dict.keys()
        else:
            return self.dict.keys()[id]

    def get_nbasis(self):
        return len(self.dict)

    def __getitem__(self, id):
        """
        get the bais with id.
        """
        return self.dict.keys()[id]

    def get_basis_at_site(self, site):
        """
        get the basis list at site.

        :param site:
            the site index.
        """
        return self.basis_at_site[site]

    def set_basis_from_ase(self, atoms, basis_dict):
        """
        set basis from ase atoms and a dict which define the relation between the atoms and the basis.

        :param atoms: ASE atoms.
        :param basis_dict: A dict,{'specy':[('name1',spin1),(name2,spin2,...)]}. eg. {'Ti':[('dxy','UP'),('dxy','DOWN')],...}, spin can be 'UP','DOWN',None

        """
        symbols = atoms.get_chemical_symbols()
        sndict = symbol_number(atoms)
        scaled_positions = atoms.get_scaled_positions()
        for i in range(len(atoms)):
            sym = symbols[i]
            if sym in basis_dict:
                for d in basis_dict[sym]:
                    self.add_basis(
                        basis(
                            i,
                            d[0],
                            d[1],
                            sndict.keys()[i],
                            element=sym,
                            position=scaled_positions[i]))

    def output_orb(self):
        """
        output orb for pythTB. (orb: the positions of the centers of the basis wave fucntion.)
        """
        orb = []
        for basis in self.dict:
            orb.append(basis.position)
        return orb

    def nospin_to_spin(self):
        sbasis = basis_set()
        for b in self.get_basis():
            b_up = copy.copy(b)
            b_dn = copy.copy(b)
            b_up.spin = 0
            b_dn.spin = 1
            sbasis.add_basis(b_up)
            sbasis.add_basis(b_dn)


def simple_basis_set(nbasis, nspin):
    """
    generate simple basis set from nbasis
    """
    bset = basis_set()
    for i in range(nbasis):
        if nspin == 1:
            b = basis(site=i, label=None, spin=None)
            bset.add_basis(b)
        else:
            bup = basis(site=i, label=None, spin=0)
            bdn = basis(site=i, label=None, spin=1)
            bset.add_basis(bup)
            bset.add_basis(bdn)
    return bset


def nospin_to_spin(O, ham0, Upqrs, bset=None, nbasis=None):
    """
    #the order: (b1_up,b2_up,b3_up,...bn_up, b1_dn,b2_dn,...,bn_dn)
    order (b1_up, b1_dn, b2_up, b2_dn, ...)
    O --> O_spin
    ham0 --> ham0_spin
    Upqrs --> Upqrs_spin

    :param O:
    :param ham0:
    :param Upqrs:
    :param nbasis: number of basis
    """
    #print Upqrs.shape
    if nbasis is None:
        if bset is not None:
            nbasis = bset.get_nbasis()
        else:
            raise ValueError("bset and nbasis cannot be both None.")
    elif bset is None:
        bset = simple_basis_set(nbasis, nspin=1)

    bas = list(range(nbasis))
    bas_spin = [(i, 'UP') for i in bas] + [(i, 'DOWN') for i in bas]
    #print bas_spin
    O = np.array(O)
    zero_array = np.zeros((nbasis, nbasis))
    O_spin = np.hstack((np.vstack((O, zero_array)), np.vstack((zero_array,
                                                               O))))
    ham0_spin = np.hstack((np.vstack((ham0, zero_array)), np.vstack(
        (zero_array, ham0))))
    Upqrs_spin = dict()
    for ind in itertools.product(*([range(nbasis * 2)] * 4)):
        p, q, r, s = ind
        if bas_spin[p][1]==bas_spin[r][1] and \
            bas_spin[q][1]==bas_spin[s][1]:
            #print bas_spin[p][0],bas_spin[q][0],bas_spin[r][0],bas_spin[s][0]
            U = Upqrs[bas_spin[p][0], bas_spin[q][0], bas_spin[r][0], bas_spin[
                s][0]]

        else:
            U = 0.0
        if U > 1e-5:
            Upqrs_spin[p, q, r, s] = U

    return O_spin, ham0_spin, Upqrs_spin
