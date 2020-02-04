#!/usr/bin/env python
import re
import numpy as np
from math import sqrt
from cmath import exp
from ase.units import Bohr, eV, Hartree
import pythtb
import matplotlib.pyplot as plt
from ase.build import make_supercell


class ifc_parser():
    def __init__(self, atoms, fname='abinit_ifc.out', primitive_atoms=None):
        with open(fname) as myfile:
            self.lines = myfile.readlines()
        self.atoms = atoms
        self.primitive_atoms = primitive_atoms
        self.R = []
        self.G = []
        self.ifc = []
        self.atom_positions = []
        self.split()
        self.read_info()
        self.read_lattice()
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
                    atom_position = map(float, lines[iline + 1].split()[-3:])
                    self.atom_positions.append(np.array(atom_position) * Bohr)
                if line.find('interaction with atom') != -1:
                    jinter = int(line.split()[0])
                    self.ifc_lines[(iatom, jinter)] = lines[iline:iline + 11]

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

    def read_lattice(self):
        lines = self.lattice_lines
        #print lines
        for line in lines:
            if line.startswith('R('):
                self.R.append(np.array((map(float, line.split()[1:4]))) * Bohr)
                self.G.append(np.array((map(float, line.split()[5:8]))) / Bohr)
        #print(self.R)

    def read_ifc_elem(self, iatom, iinter):
        lines = self.ifc_lines[(iatom, iinter)]
        w = lines[0].strip().split()
        jatom = int(w[-3])
        jcell = int(w[-1])
        pos_j = [float(x) * Bohr for x in lines[1].strip().split()[-3:]]
        distance = float(lines[2].strip().split()[-1])

        # Fortran format F9.5: some values are not seperated by whitespaces.
        # Maybe better fit that in abinit code.
        #ifc0=  map(float, lines[3].strip().split()[0:3])
        #ifc1=  map(float, lines[4].strip().split()[0:3])
        #ifc2=  map(float, lines[5].strip().split()[0:3])
        #ifc=np.array([ifc0, ifc1, ifc2])

        #print lines[3]
        ifc0 = map(float,
                   [lines[3][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        ifc1 = map(float,
                   [lines[4][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        ifc2 = map(float,
                   [lines[5][i * 9 + 1:i * 9 + 9 + 1] for i in range(3)])
        ifc = np.array([ifc0, ifc1, ifc2])

        if not ('dipdip' in self.invars and self.invars['dipdip'] == 1):
            ifc_ewald = None
            ifc_sr = None
        else:
            ifc0 = map(
                float,
                [lines[3][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)])
            ifc1 = map(
                float,
                [lines[4][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)])
            ifc2 = map(
                float,
                [lines[5][i * 9 + 2:i * 9 + 9 + 2] for i in range(3, 6)])
            #ifc0=  map(float, lines[3].strip().split()[3:6])
            #ifc1=  map(float, lines[4].strip().split()[3:6])
            #ifc2=  map(float, lines[5].strip().split()[3:6])
            ifc_ewald = np.array([ifc0, ifc1, ifc2])

            ifc0 = map(
                float,
                [lines[3][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)])
            ifc1 = map(
                float,
                [lines[4][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)])
            ifc2 = map(
                float,
                [lines[5][i * 9 + 3:i * 9 + 9 + 3] for i in range(6, 9)])
            #ifc0=  map(float, lines[3].strip().split()[6:9])
            #ifc1=  map(float, lines[4].strip().split()[6:9])
            #ifc2=  map(float, lines[5].strip().split()[6:9])
            ifc_sr = np.array([ifc0, ifc1, ifc2])
        return {
            "iatom": iatom,
            "jatom": jatom,
            "jcell": jcell,
            "jpos": pos_j,
            "distance": distance,
            "ifc": ifc,
            "ifc_ewald": ifc_ewald,
            "ifc_sr": ifc_sr
        }

    def read_ifc(self):
        natom, niteraction = sorted(self.ifc_lines.keys())[-1]
        for iatom in range(1, natom + 1):
            for iint in range(1, niteraction + 1):
                self.ifc.append(self.read_ifc_elem(iatom, iint))

    def map_ifc_to_primitive_cell(self):
        if self.primitive_atoms is None:
            return
        pcell = self.primitive_atoms.get_cell()
        new_ifc = []
        for ifce in self.ifc:
            # remove atoms outside of primitive cell
            if ifce['iatom'] <= len(self.primitive_atoms):
                jpos = ifce['jpos']
                for i, pos in enumerate(self.primitive_atoms.get_positions()):
                    dis = jpos - pos
                    R = np.linalg.solve(pcell, dis)
                    if np.all(
                            np.isclose(
                                np.mod(R + 1e-4, [1, 1, 1]), [0, 0, 0],
                                atol=1e-3)):
                        jatom = i + 1
                        #print "Found: " ,jatom
                        break
                ifce['jatom'] = jatom
                new_ifc.append(ifce)
        self.ifc = new_ifc

    def make_model(self):
        if self.primitive_atoms is not None:
            atoms = self.primitive_atoms
        else:
            atoms = self.atoms
        lat = atoms.get_cell()
        apos = atoms.get_scaled_positions()
        masses = atoms.get_masses()
        orb = []
        for pos in apos:
            for i in range(3):
                orb.append(pos)
        model = pythtb.tb_model(3, 3, lat=lat, orb=orb)
        atom_positions = atoms.get_positions()
        for ifce in self.ifc:
            #print ifce
            iatom = ifce['iatom']
            jatom = ifce['jatom']
            jpos = ifce['jpos']
            R0 = atom_positions[ifce['jatom'] - 1]
            dis = R0 - jpos
            #print self.atoms.get_cell()
            R = np.linalg.solve(atoms.get_cell(), dis)
            #print R
            R = [int(round(x)) for x in R]
            #print R
            for a in range(3):
                for b in range(3):
                    i = 3 * (iatom - 1) + a
                    j = 3 * (jatom - 1) + b
                    val = ifce['ifc'][a, b] * Hartree / np.sqrt(
                        masses[iatom - 1] * masses[jatom - 1])
                    #print i,j,R
                    if iatom == jatom and a == b and R == [0, 0, 0]:
                        model.set_onsite(val * 2, ind_i=i)
                    else:
                        model.set_hop(val, i, j, R, allow_conjugate_pair=True)
                #print self.atoms.get_positions()
        self.model = model

    def solve_model(self):
        my_model = self.model
        # generate k-point path and labels
        path = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0, 0, 0],
                [0.5, 0.5, 0.5]]
        label = (r'$\Gamma $', r'$X$', r'$M$', r'$\Gamma$', r'$R$')
        (k_vec, k_dist, k_node) = my_model.k_path(path, 301)

        print('---------------------------------------')
        print('starting calculation')
        print('---------------------------------------')
        print('Calculating bands...')

        # solve for eigenenergies of hamiltonian on
        # the set of k-points from above
        evals = my_model.solve_all(k_vec)
        s = np.sign(evals)
        v = np.sqrt(evals * s)
        evals = s * v * 15.633302
        # plotting of band structure
        print('Plotting bandstructure...')

        # First make a figure object
        fig, ax = plt.subplots()

        # specify horizontal axis details
        ax.set_xlim([0, k_node[-1]])
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        ax.axhline(y=0, linestyle='--')
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k')

        # plot bands
        for n in range(len(self.primitive_atoms) * 3):
            ax.plot(k_dist, evals[n])
        # put title
        ax.set_title("Checkerboard band structure")
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy")
        # make an PDF figure of a plot
        fig.tight_layout()
        fig.savefig("phonon.pdf")
        plt.show()

        print('Done.\n')


def read_value(line, key):
    try:
        l = line.strip().split()
        if l[0] == key:
            return l[1]
        else:
            return False
    except:
        return False
