#!/usr/bin/env python
import os
import numpy as np
from ase.units import Bohr, Hartree
from ase.io import read
from collections import namedtuple
from minimulti.abstract.datatype import IFClike, Rijv

ifc_elem = namedtuple("ifc_elem",
                      ('i', 'j', 'R', 'tot', 'sr', 'ewald', 'distance'))


class IFC(list):
    def __init__(self, dipdip):
        super(ifc, self).__init__()
        self._dipdip = dipdip

    def has_dipdip(self):
        return self._dipdip


def read_value(line, key):
    try:
        l = line.strip().split()
        if l[0] == key:
            return l[1]
        else:
            return False
    except:
        return False


class IFCParser():
    def __init__(self, atoms, fname='abinit_ifc.out', cell=np.eye(3)):
        with open(fname) as myfile:
            self.lines = myfile.readlines()
        self.atoms = atoms
        self.dipdip = True
        self.R = []
        self.G = []
        self.ifc = []
        self.atom_positions = []
        self.split()
        self.read_info()
        self.read_lattice()
        self.atoms.set_positions(self.atom_positions)
        self.natom = len(self.atoms)
        self.read_ifc()

    def split(self):
        lines = self.lines
        self.info_lines = []
        self.lattice_lines = []
        self.ifc_lines = {}

        inp_block = False
        lat_block = False
        ifc_block = False

        for iline, line in enumerate(lines):
            # input variables
            if line.strip().startswith("-outvars_anaddb:"):
                inp_block = True
            if line.strip().startswith("========="):
                inp_block = False
            if inp_block:
                self.info_lines.append(line.strip())

            # lattice vectors
            if line.strip().startswith("read the DDB information"):
                lat_block = True
            if line.strip().startswith("========="):
                lat_block = False
            if lat_block:
                self.lattice_lines.append(line.strip())
            # IFC block
            if line.strip().startswith(
                    "Calculation of the interatomic forces"):
                ifc_block = True
            if line.strip().startswith("========="):
                ifc_block = False
            if ifc_block:
                if line.strip().startswith('generic atom number'):
                    iatom = int(line.split()[-1])
                    atom_position = list(
                        map(float, lines[iline + 1].split()[-3:]))
                    self.atom_positions.append(np.array(atom_position) * Bohr)
                if line.find('interaction with atom') != -1:
                    jinter = int(line.split()[0])
                    self.ifc_lines[(iatom, jinter)] = lines[iline:iline + 15]

    def read_info(self):
        self.invars = {}
        keys = ('asr', 'chneut', 'dipdip', 'ifcflag', 'ifcana', 'ifcout',
                'natifc', 'brav')
        lines = self.info_lines
        for line in lines:
            for key in keys:
                val = read_value(line, key)
                if val:
                    self.invars[key] = int(val)
        #print(self.invars)
        if not ('dipdip' in self.invars and self.invars['dipdip'] == 1):
            self.dipdip = False
        else:
            self.dipdip = True

    def read_lattice(self):
        lines = self.lattice_lines
        #print lines
        for line in lines:
            if line.startswith('R('):
                self.R.append(
                    np.array(list(map(float,
                                      line.split()[1:4]))) * Bohr)
                self.G.append(
                    np.array(list(map(float,
                                      line.split()[5:8]))) / Bohr)
        #print(self.R)

    def read_ifc_elem(self, iatom, iinter):
        lines = self.ifc_lines[(iatom, iinter)]
        w = lines[0].strip().split()
        jatom = int(w[-3])
        jcell = int(w[-1])
        pos_j = [float(x) * Bohr for x in lines[1].strip().split()[-3:]]
        distance = float(lines[2].strip().split()[-1]) * Bohr

        atom_positions = self.atoms.get_positions()

        Ri = atom_positions[iatom - 1]
        Rj = atom_positions[jatom - 1]
        dis = pos_j - Rj
        #print self.atoms.get_cell()
        R = np.linalg.solve(self.atoms.get_cell(), dis)
        #print R
        R = [int(round(x)) for x in R]
        if not np.isclose(
                np.dot(self.atoms.get_cell(), R) + atom_positions[jatom - 1],
                pos_j).all():
            print("Warning: Not equal",
                  np.dot(self.atoms.get_cell(), R) + atom_positions[jatom - 1],
                  pos_j)

        # check distance

        # Fortran format F9.5: some values are not seperated by whitespaces.
        # Maybe better fit that in abinit code.
        ifc0 = list(map(float, lines[3].strip().split()[0:3]))
        ifc1 = list(map(float, lines[4].strip().split()[0:3]))
        ifc2 = list(map(float, lines[5].strip().split()[0:3]))
        ifc = np.array([ifc0, ifc1, ifc2]).T * Hartree / Bohr**2

        #print lines[3]
        # ifc0 = map(float,
        #            [lines[3][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        # ifc1 = map(float,
        #            [lines[4][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        # ifc2 = map(float,
        #            [lines[5][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        #ifc = np.array([ifc0, ifc1, ifc2]).T * Hartree / Bohr**2
        #ifc = np.array([ifc0, ifc1, ifc2]) * Hartree / Bohr**2

        if not ('dipdip' in self.invars and self.invars['dipdip'] == 1):
            ifc_ewald = None
            ifc_sr = None
        else:
            ifc0 = list(
                map(float,
                    [lines[3][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)]))
            ifc1 = list(
                map(float,
                    [lines[4][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)]))
            ifc2 = list(
                map(float,
                    [lines[5][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)]))
            #ifc0=  map(float, lines[3].strip().split()[3:6])
            #ifc1=  map(float, lines[4].strip().split()[3:6])
            #ifc2=  map(float, lines[5].strip().split()[3:6])
            ifc_ewald = np.array([ifc0, ifc1, ifc2])

            ifc0 = list(
                map(float,
                    [lines[3][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)]))
            ifc1 = list(
                map(float,
                    [lines[4][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)]))
            ifc2 = list(
                map(float,
                    [lines[5][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)]))
            #ifc0=  map(float, lines[3].strip().split()[6:9])
            #ifc1=  map(float, lines[4].strip().split()[6:9])
            #ifc2=  map(float, lines[5].strip().split()[6:9])
            ifc_sr = np.array([ifc0, ifc1, ifc2])

        # sanity check
        poses = self.atoms.get_positions()
        jpos = pos_j
        ipos = poses[iatom - 1]
        np.array(jpos)
        if not np.abs(np.linalg.norm(np.array(jpos) - ipos) - distance) < 1e-5:
            print("Warning: distance wrong", "iatom: ", iatom, ipos, "jatom",
                  jatom, jpos)
        return ifc_elem(
            i=iatom - 1,
            j=jatom - 1,
            R=R,
            tot=ifc,
            ewald=ifc_ewald,
            sr=ifc_sr,
            distance=distance)

    # return {
    #     "iatom": iatom,
    #     "jatom": jatom,
    #     "R": R,
    #     "jcell": jcell,
    #     "jpos": pos_j,
    #     "distance": distance,
    #     "ifc": ifc,
    #     "ifc_ewald": ifc_ewald,
    #     "ifc_sr": ifc_sr
    # }

    def read_ifc(self):
        natom, niteraction = sorted(self.ifc_lines.keys())[-1]
        for iatom in range(1, natom + 1):
            for iint in range(1, niteraction + 1):
                self.ifc.append(self.read_ifc_elem(iatom, iint))

    def get_atoms(self):
        return self.atoms

    def get_ifc(self, max_distance=None):
        """
        term: tot, ewald, sr
        return iatom, jatom, R, ifc
        """
        if max_distance is not None:
            ret = [ifce for ifce in self.ifc if ifce.distance < max_distance]
        else:
            ret = self.ifc
        return ret

    def get_total_ifc(self, max_distance=None):
        """
        term: tot, ewald, sr
        return iatom, jatom, R, ifc
        """
        ret = Rijv(shape=(self.natom * 3, self.natom * 3), sparse=False)
        for ifce in self.ifc:
            if max_distance is None or ifce.distance < max_distance:
                for i in range(3):
                    for j in range(3):
                        if abs(ifce.tot[i, j]) > 1e-7:
                            ret[tuple(
                                ifce.R)][(ifce.i * 3 + i,
                                          ifce.j * 3 + j)] = ifce.tot[i, j]
        return ret

    def save_to_netcdf(self, fname):

        ifc = self.get_total_ifc()
        Rlist = tuple(ifc.keys())
        nR = len(Rlist)
        ifc_vals = np.array(list(ifc.values()))

        from netCDF4 import Dataset
        root = Dataset(fname, 'w')
        three = root.createDimension("three", 3)
        natom = root.createDimension("natom", self.natom)
        natom3 = root.createDimension("natom3", self.natom * 3)
        ifc_nR = root.createDimension("ifc_nR", nR)

        ref_masses = root.createVariable("ref_masses", 'f8', ("natom", ))
        ref_masses.description = "REFerence atom MASSES"
        ref_zion = root.createVariable("ref_zion", 'i4', ("natom", ))
        ref_zion.description = "REFerence atom ZION"
        ref_cell = root.createVariable("ref_cell", 'f8', ("three", "three"))

        ref_cell.description = "REFerence structure CELL"
        ref_xred = root.createVariable("ref_xred", 'f8', ("natom", "three"))

        ref_xred.description = "REFerence structure XRED"

        ifc_Rlist = root.createVariable("ifc_Rlist", "i4", ("ifc_nR", "three"))

        ifc_Rlist.description = "IFC RLIST"
        ifc_vallist = root.createVariable("ifc_vallist", "f8",
                                          ("ifc_nR", "natom3", "natom3"))

        ifc_vallist.description = "IFC VALUE LIST"
        ref_cell.unit = "Angstrom"
        ref_masses.unit = "atomic"
        ifc_vallist.unit = "eV/Angstrom^2"

        ref_masses[:] = self.atoms.get_masses()
        ref_zion[:] = self.atoms.get_atomic_numbers()
        ref_cell[:] = self.atoms.get_cell()
        ref_xred[:] = self.atoms.get_scaled_positions()

        ifc_Rlist[:] = np.array(Rlist)
        ifc_vallist[:] = ifc_vals
        root.close()


    def write_IFC_list(self, prefix='IFC_'):
        """
        write IFC to a file which can be more easily read by C.
        """
        TTfile = prefix + 'Total.txt'
        with open(TTfile, 'w+') as myfile:
            myfile.write('%d\n' % (len(self.ifc)))
            for ifce in self.ifc:
                myfile.write('%3d %3d ' % (ifce.i, ifce.j))
                myfile.write(' '.join(["%8f" % x for x in ifce.R]))
                myfile.write(' ')
                myfile.write(' '.join([
                    "%12f" % x for x in ifce.tot.reshape(9) / Hartree * Bohr**2
                ]))
                myfile.write('\n')
        if self.dipdip:
            SRfile = prefix + 'SR.txt'
            DDfile = prefix + 'DD.txt'
            with open(SRfile, 'w+') as myfile:
                myfile.write('%d\n' % (len(self.ifc)))
                for ifce in self.ifc:
                    myfile.write('%3d %3d ' % (ifce.i, ifce.j))
                    myfile.write(' '.join(["%8f" % x for x in ifce.R]))
                    myfile.write(' ')
                    myfile.write(' '.join([
                        "%12f" % x
                        for x in ifce.sr.reshape(9) / Hartree * Bohr**2
                    ]))
                    myfile.write('\n')
            with open(DDfile, 'w+') as myfile:
                myfile.write('%d\n' % (len(self.ifc)))
                for ifce in self.ifc:
                    myfile.write('%3d %3d ' % (ifce.i, ifce.j))
                    myfile.write(' '.join(["%8f" % x for x in ifce.R]))
                    myfile.write(' ')
                    myfile.write(' '.join([
                        "%12f" % x
                        for x in ifce.ewald.reshape(9) / Hartree * Bohr**2
                    ]))
                    myfile.write('\n')


def test(mag='G'):
    path = "/Users/hexu/projects/SrMnO3/scripts/model"

    atoms = read(os.path.join(path, 'POSCAR'))
    ifcfile = os.path.join(path, 'FM_ifc.txt')

    ifc = IFCParser(atoms, fname=ifcfile)

    #print(ifc.get_ifc()[0])
    t = ifc.get_total_ifc()
    #print(t.keys())
    #print(list(t.values())[0])
    #ifc.write_IFC_list(prefix='IFC_%s_' % mag)
    ifc.save_to_netcdf('ifc.nc')


if __name__ == '__main__':
    #for mag in ['A', 'C', 'G', 'FM']:
    #    test(mag)
    test()
