from collections import namedtuple, defaultdict
from itertools import groupby
from ase.atoms import Atoms
import numpy as np
from minimulti.utils.supercell import SupercellMaker

Basis = namedtuple('Basis', field_names=('site', 'label', 'spin', 'index'))

# map from labels to l, m pairs.
# order of VASP and abinit
label_lm = {
    # d
    'd3z^2-r^2': (2, 0),
    'd3z2-r2': (2, 0),
    'dz^2': (2, 0),
    'dz2': (2, 0),
    'dx^2-y^2': (2, -2),
    'dx2-y2': (2, -2),
    'dx^2': (2, -2),
    'dx2': (2, -2),
    'dxy': (2, 2),
    'dxz': (2, -1),
    'dyz': (2, 1),

    # p
    'px': (1, -1),
    'pz': (1, 0),
    'py': (1, 1),

    # s
    's': (0, 0)
}

# order of Vee
label_lm = {
    'd3z^2-r^2': (2, 0),
    'd3z2-r2': (2, 0),
    'dz^2': (2, 0),
    'dz2': (2, 0),
    'dx^2-y^2': (2, 1),
    'dx2-y2': (2, 1),
    'dx^2': (2, 1),          
    'dx2': (2, 1),
    'dxy': (2, 2),
    'dyz': (2, 3),
    'dxz': (2, 4),
    # p
    'px': (1, -1),
    'pz': (1, 0),
    'py': (1, 1),
    # s
    's' : (0, 0)

}


def get_site(basis):
    return basis.site


def get_site_and_l(basis):
    return basis.site, label_lm[basis.label][0]


def get_site_lm(basis):
    return basis.site, label_lm[basis.label][0], label_lm[basis.label][1]


def get_site_lmspin(basis):
    return (basis.site, label_lm[basis.label][0], label_lm[basis.label][1],
            basis.spin)


class BasisSet(list):
    def __init__(self):
        self.atoms = None
        self.nspin = 1
        self._id_dict = None
        self._llist = []
        self._mlist = []

    def _build_id_dict(self):
        self._id_dict = dict(((basis.site, basis.label, basis.spin),
                              basis.index) for basis in self)

    def get_id(self, site, label, spin):
        if self._id_dict is None:
            self._build_id_dict()
        return self._id_dict[(site, label, spin)]

    def get_basis_id(self, basis):
        return self.get_id(basis.site, basis.label, basis.spin)

    def from_basis_list(self, basis_list):
        if basis_list is not None:
            for basis in basis_list:
                self.append(basis)

    def append(self, basis):
        super(BasisSet, self).append(basis._replace(index=len(self)))

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_sites(self):
        return [e.site for e in self]

    def get_nsites(self):
        return len(self.get_sites())

    def get_spins(self):
        return [e.spin for e in self]

    def get_indices(self):
        return [e.index for e in self]

    def get_labels(self):
        return [e.label for e in self]

    def get_lm_list(self):
        """
        labels -> l, m
        """
        if self._llist == [] or self._mlist == []:
            for e in self:
                if e.label in label_lm:
                    self._lm_list.append(label_lm[e.label])
                else:
                    raise ValueError(
                        "in basis set, label %s is not a valid label")
        return self._llist, self._mlist

    def get_l(self, id):
        return self.get_lm_list()[0][id]

    def get_m(self, id):
        return self.get_lm_list()[1][id]

    def group_by_site_and_l(self):
        return groupby(sorted(self, key=get_site_lmspin), get_site_and_l)

    def group_by_site(self):
        return groupby(sorted(self, key=get_site_lmspin), get_site)

    def get_atomic_number(self, id):
        self.atoms.get_atomic_numbers()[self[id].site]

    def get_atomic_numbers(self):
        return tuple(self.atoms.get_atomic_numbers()[e.site] for e in self)

    def get_chemical_symbol(self, id):
        self.atoms.get_chemical_symbols()[self[id].site]

    def get_chemical_symbols(self):
        return tuple(self.atoms.get_chemical_symbols()[e.site] for e in self)

    def get_positions(self):
        return np.array(
            tuple(self.atoms.get_positions()[e.site] for e in self))

    def get_position(self, e):
        return self.atoms.get_positions()[e.site]

    def get_scaled_positions(self):
        return np.array(
            tuple(self.atoms.get_scaled_positions()[e.site] for e in self))

    def get_scaled_position(self, e):
        return np.array(self.atoms.get_scaled_positions()[e.site])

    def set_nspin(self, nspin):
        self.nspin = nspin

    def get_nspin(self):
        return self.nspin

    def group_by_site(self):
        return groupby(sorted(self), lambda x: x.site)

    def add_spin_index(self):
        """
        Nospin-> spin up & down
        """
        bset_spin = BasisSet()
        for b in self:
            bset_spin.append(b._replace(spin=0))
            bset_spin.append(b._replace(spin=1))
        bset_spin.set_nspin(2)
        bset_spin.set_atoms(self.atoms)
        return bset_spin

    def make_supercell(self, sc_matrix=None, smaker=None):
        """
        make supercell
        sc_matrix: supercell matrix.
        """
        if self.atoms is not None:
            natoms = len(self.atoms)
        else:
            natoms = set(e.site for e in self)

        if smaker is None:
            smaker = SupercellMaker(sc_matrix)

        if self.atoms is not None:
            sc_atoms = smaker.sc_atoms(self.atoms)

        sc_sites = smaker.sc_index(self.get_sites(), n_ind=natoms)
        sc_labels = smaker.sc_trans_invariant(self.get_labels())
        sc_spins = smaker.sc_trans_invariant(self.get_spins())
        sc_indices = smaker.sc_index(self.get_indices())

        sc_bset = BasisSet()
        sc_bset.set_atoms(sc_atoms)
        for site, label, spin, ind in zip(sc_sites, sc_labels, sc_spins,
                                          sc_indices):
            sc_bset.append(Basis(site, label, spin, ind))
        sc_bset.set_nspin(self.nspin)
        return sc_bset


def gen_basis_set(nsites, nlabels, nspin=None):
    """
    generate basis on nlabels of basis on each of nsites.
    """
    bset = BasisSet()
    atoms = Atoms(
        symbols=['H'] * nsites, cell=[1, 1, 1], positions=[[1, 1, 1]] * nsites)
    bset.set_atoms(atoms)
    for isite in range(nsites):
        for ilabel in range(nlabels):
            for ispin in range(nspin):
                bset.append(
                    Basis(index=0, site=isite, label=ilabel, spin=ispin))
    return bset


def atoms_to_basis(atoms, basis_dict, nspin=1):
    """
    e.g.
    atoms = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
    basis = atoms_to_basis(atoms, basis_dict={'H': ('s')}, nspin=2)
    """
    symbols = atoms.get_chemical_symbols()
    bset = BasisSet()
    bset.set_atoms(atoms)
    bset.set_nspin(nspin)
    for i, sym in enumerate(symbols):
        if sym in basis_dict:
            if not isinstance(basis_dict[sym], (list, tuple)):
                raise ValueError(
                    'In basis_dict, the value should be lists or tuples')
            for label in basis_dict[sym]:
                if nspin == 0:
                    bset.append(Basis(site=i, label=label, spin=0, index=0))
                else:
                    bset.append(Basis(site=i, label=label, spin=0, index=0))
                    bset.append(Basis(site=i, label=label, spin=1, index=0))
    return bset


def test():
    b1 = Basis(index=0, site=0, label='dxy', spin=None)
    b2 = Basis(index=0, site=1, label='dxy', spin=None)
    b3 = Basis(index=0, site=2, label='dxy', spin=None)
    b4 = Basis(index=0, site=1, label='dxy', spin=None)

    Bs = BasisSet()
    Bs.from_basis_list([b1, b2, b3, b4])
    print(Bs)

    for s, Bg in Bs.group_by_site():
        print("s:", s)
        for B in Bg:
            print(B)

    from ase.atoms import Atoms
    atoms = Atoms(symbols='H3', positions=[[0, 0, 0]] * 3, cell=[1, 1, 1])
    Bs.set_atoms(atoms)
    print(Bs.get_atomic_numbers())
    print(Bs.get_chemical_symbols())

    Bspin = Bs.add_spin_index()
    print(Bspin)
    print(len(Bspin))

    print(gen_basis_set(2, 3, nspin=1))
    atoms = Atoms(symbols='H', positions=[(0, 0, 0)], cell=[1, 1, 1])
    basis = atoms_to_basis(atoms, basis_dict={'H': ('s', )}, nspin=2)
    print(basis)

    print(basis.make_supercell(sc_matrix=np.eye(3) * 2))


if __name__ == '__main__':
    test()
