import numpy as np
from ase.neighborlist import PrimitiveNeighborList
from minimulti.utils.symbol import symnum_to_sym, symbol_number
from math import sqrt
from numpy.linalg import norm

class bond():
    """
    bond has these attributes: atom1,atom2,vector,orientation, name1, name2
    """
    def __init__(self,
                 atom1,
                 atom2,
                 vector,
                 orientation,
                 offset,
                 name1='Unkown',
                 name2='Unkown'):
        self.atom1 = atom1
        self.atom2 = atom2
        self.vector = vector
        self.length = norm(self.vector)
        self.orientation = orientation
        self.offset = offset
        if name1 == 'Unkown':
            self.name1 = 'Unkown' + str(atom1)
        else:
            self.name1 = name1
        if name2 == 'Unkown':
            self.name2 = 'Unkown' + str(atom2)
        else:
            self.name2 = name2

    def __repr__(self):
        return "Bond: from %s (No.%s) to %s (No.%s) in orientation %s.\n offset: %s.\n vector: %s\n length: %s\n" % (
            self.name1, self.atom1, self.name2, self.atom2, self.orientation,
            self.offset, self.vector, self.length)


class myNeighborList(PrimitiveNeighborList):
    """
    Inherated from ase.calculators.NeighborList. But add the functionanity to select the neighbors according to the species.
    default bothways is set to True.
    """

    def __init__(self,
                 cutoffs,
                 skin=0.3,
                 sorted=False,
                 self_interaction=True,
                 bothways=False):
        PrimitiveNeighborList.__init__(
            self,
            cutoffs,
            skin=skin,
            sorted=sorted,
            self_interaction=self_interaction,
            bothways=bothways)
        self.sndict = {}
        self.axis = np.eye(3)

    def filter_species(self, a, b, n1, n2):
        """
        filter the species of atoms
        a and b are chemical symbols, n1 and n2 are atom indexes.
        """
        return symnum_to_sym(n1) == a and symnum_to_sym(n2) == b

    def my_get_neighbors(self, a, to_species=None, show_vec=False):
        """
        modify ase NeighborList so that a can be symnum.
        """
        if isinstance(a, int):
            r = self.get_neighbors(a)
            v = self.disp_vectors[a]
        elif isinstance(a, str):
            r = self.get_neighbors(self.sndict[a])
            v = self.disp_vectors[self.sndict[a]]

        if to_species is None:
            inn, offset_nn, vnn = r[0], r[1], v
        else:
            inn = []
            offset_nn = []
            vnn = []
            s = self.atoms.get_chemical_symbols()
            iatoms, offsets = r

            for i, ofs, iv in zip(iatoms, offsets, v):
                if s[i] in to_species:
                    inn.append(i)
                    offset_nn.append(ofs)
                    vnn.append(iv)
        if show_vec:
            ret = inn, offset_nn, vnn
        else:
            ret = inn, offset_nn
        return ret

    def get_bonds(self, a, to_species=None):
        """
        detailed infomation of the bonds connecting to a.

        :param a: the index or the symbol_number of a
        :param to_species: the list of the species connecting to a. eg. ['Ti','O']. Don't forget to include a itself if necessery.
        """
        if isinstance(a, int):
            ai = a
        else:
            ai = self.sndict[a]
        #symbols=self.atoms.get_chemical_symbols()
        inn, offset_nn, vnn = self.my_get_neighbors(
            a, to_species=to_species, show_vec=True)
        nsdict = dict(zip(self.sndict.values(), self.sndict.keys()))
        bonds = []
        for i, offset, v in zip(inn, offset_nn, vnn):
            ori = self.bond_orientation(v)
            #self.bond_orientation(ai,i,offset)
            b = bond(ai, i, v, ori, offset, name1=nsdict[ai], name2=nsdict[i])
            bonds.append(b)
        return bonds

    def get_all_bonds(self, from_species=None, to_species=None):
        """
        get all the bonds. Note that the parameter 'bothways' in the build()  function.

        :param from_species: a list of species.
        :param to_species: another list of species.
        """
        if from_species is None:
            bonds = []
            for i in range(len(self.atoms)):
                bonds += self.get_bonds(i, to_species=to_species)
        else:
            bonds = []
            symbols = self.atoms.get_chemical_symbols()
            for i in range(len(self.atoms)):
                if symbols[i] in from_species:
                    bonds += self.get_bonds(i, to_species=to_species)
        return bonds

    def set_axis(self, x=[1.0, 0.0, 0.0], y=[0, 1, 0], z=[0, 0, 1]):
        x0 = np.array(x) / norm(np.array(x))
        y0 = np.array(y) / norm(np.array(y))
        z0 = np.array(z) / norm(np.array(z))
        self.axis = np.array([x0, y0, z0])

    def bond_orientation(self, vector):
        """
        naive implementation of finding the bond orietation. Fails to work if the direction is not along x,y z.
        """
        vec = vector  #self.atoms[a].position-self.atoms[b].position
        proj = self.axis.dot(vec.reshape((3, 1))).reshape(3)
        proj = np.concatenate((proj, -proj))

        ori = ['x', 'y', 'z', '-x', '-y', '-z']

        o = ori[np.argmax(proj)]
        p = np.sort(proj)
        if abs(p[0] - p[1]) < 0.01:
            o = 'Other'
        return o

    def build(self, atoms):
        """Build the list. Modified so that the vectors are also stored."""
        self.atoms = atoms
        self.sndict = symbol_number(atoms)

        self.positions = atoms.get_positions()
        self.pbc = atoms.get_pbc()
        self.cell = atoms.get_cell()
        if len(self.cutoffs) > 0:
            rcmax = self.cutoffs.max()
        else:
            rcmax = 0.0

        icell = np.linalg.inv(self.cell)
        scaled = np.dot(self.positions, icell)
        scaled0 = scaled.copy()

        N = []
        for i in range(3):
            if self.pbc[i]:
                scaled0[:, i] %= 1.0
                v = icell[:, i]
                h = 1 / sqrt(np.dot(v, v))
                n = int(2 * rcmax / h) + 1
            else:
                n = 0
            N.append(n)

        offsets = (scaled0 - scaled).round().astype(int)
        positions0 = np.dot(scaled0, self.cell)
        natoms = len(atoms)
        indices = np.arange(natoms)

        self.nneighbors = 0
        self.npbcneighbors = 0
        self.neighbors = [np.empty(0, int) for a in range(natoms)]
        self.displacements = [np.empty((0, 3), int) for a in range(natoms)]
        self.disp_vectors = [np.empty((0, 3), float) for a in range(natoms)]
        for n1 in range(0, N[0] + 1):
            for n2 in range(-N[1], N[1] + 1):
                for n3 in range(-N[2], N[2] + 1):
                    if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                        continue
                    displacement = np.dot((n1, n2, n3), self.cell)
                    for a in range(natoms):
                        d = positions0 + displacement - positions0[a]
                        i = indices[(d**2).sum(1) <
                                    (self.cutoffs + self.cutoffs[a])**2]
                        if n1 == 0 and n2 == 0 and n3 == 0:
                            if self.self_interaction:
                                i = i[i >= a]
                            else:
                                i = i[i > a]
                        self.nneighbors += len(i)
                        self.neighbors[a] = np.concatenate((self.neighbors[a],
                                                            i))
                        self.disp_vectors[a] = np.concatenate(
                            (self.disp_vectors[a], d[i]))
                        disp = np.empty((len(i), 3), int)
                        disp[:] = (n1, n2, n3)
                        disp += offsets[i] - offsets[a]
                        self.npbcneighbors += disp.any(1).sum()
                        self.displacements[a] = np.concatenate(
                            (self.displacements[a], disp))

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            disp_vectors2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp, disp_vec in zip(self.neighbors[a],
                                             self.displacements[a],
                                             self.disp_vectors[a]):
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
                    disp_vectors2[b].append(-disp_vec)

            for a in range(natoms):
                self.neighbors[a] = np.concatenate(
                    (self.neighbors[a], np.array(neighbors2[a], dtype=int))
                )  ##here dtype should be int.Becuase np.array([]) is float64 .

                self.displacements[a] = np.array(
                    list(self.displacements[a]) + displacements2[a])
                self.disp_vectors[a] = np.array(
                    list(self.disp_vectors[a]) + disp_vectors2[a])

        if self.sorted:
            for a, i in enumerate(self.neighbors):
                mask = (i < a)
                if mask.any():
                    j = i[mask]
                    offsets = self.displacements[a][mask]
                    for b, offset in zip(j, offsets):
                        self.neighbors[b] = np.concatenate((self.neighbors[b],
                                                            [a]))
                        self.displacements[b] = np.concatenate(
                            (self.displacements[b], [-offset]))
                        self.disp_vectors[b] = np.concatenate(
                            (self.disp_vectors[b], [-offset]))
                    mask = np.logical_not(mask)
                    self.neighbors[a] = self.neighbors[a][mask]
                    self.displacements[a] = self.displacements[a][mask]
                    self.disp_vectors[a] = self.disp_vectors[a][mask]
        self.nupdates += 1
