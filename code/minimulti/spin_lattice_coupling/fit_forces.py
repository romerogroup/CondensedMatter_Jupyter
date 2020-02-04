import numpy as np
from numpy import sqrt
import xml.etree.ElementTree as ET
from netCDF4 import Dataset
from minimulti.utils.supercell import map_to_primitive
from minimulti.utils.symbol import symbol_number


class OijuFitter(object):
    """
    This class is for fitting and saving Oiju data in primitive cell.
    In unitcell, i, j, u becomes
    (i, Ri=0, j, Rj, u, Ru)
    where i, j, u are indices in unitcell.
    """

    def __init__(self, atoms, sc_atoms, scell, nspin):
        """
        atoms: primitive cell structure
        nspin: number of magnetic atoms
        """
        self.atoms = atoms
        self.sc_atoms = sc_atoms
        self.scell = scell
        self.nspin = nspin
        self.data = {}

    def fit(self, ijdict, fname_template):
        """
        ijdict: {(isc, jsc): (i,j, Rj)}
        fname_template: a function fname_template(i,j,si,sj)
            is the xml filename with Si and Sj at site i and j.
        """
        for key, val in ijdict.items():
            isc, jsc = key
            i, j, Rj = val
            tmp = self.get_Oiju_supercell(isc, jsc, fname_template)
            self.to_unitcell(i, j, Rj, tmp)
        self.add_swap_ij()

    def get_Oiju_supercell(self, i, j, fname_template):
        """
        fit Oiju for a spin pair i and j in supercell.
        i, j are the indices of spin in supercell.
        """
        forceuu = read_forces(fname_template(i, j, 1, 1))
        forceud = read_forces(fname_template(i, j, 1, -1))
        forcedu = read_forces(fname_template(i, j, -1, 1))
        forcedd = read_forces(fname_template(i, j, -1, -1))
        return (forceuu + forcedd - forceud - forcedu) / 8.0

    def to_unitcell(self, i, j, Rj, Oiju):
        """
        map a Oiju from one supercell to unitcell.
        Oiju is a 3*natom matrix.
        """
        ulist, Rulist = map_to_primitive(self.sc_atoms, self.atoms)
        sdict = symbol_number(self.sc_atoms)
        for u_atom, (u, Ru, valu) in enumerate(zip(ulist, Rulist, Oiju)):
            i_atom = sdict['Mn%s' % i]
            j_atom = sdict['Mn%s' % j]
            t = self.weight(Rj, Ru)
            #t = self.weight(i_atom, j_atom, u_atom)
            for Ru_shifted, w in zip(*t):
                if w > 1e-3:
                    # Note u+1 because the index starts from 1.
                    self.data[(i, j, tuple(Rj), u + 1,
                               tuple(Ru_shifted))] = valu * w

    def add_swap_ij(self):
        """
        Oiju=Ojiu. Thus in primitive cell,
        (j, Rj, i, Ri=0, u, Ru) -> (j,Rj=0, i, Ri-Rj, u, Ru-Rj)
        by translation of Rj
        """
        sdata = {}
        for key, val in self.data.items():
            i, j, Rj, u, Ru = key
            # j=i is assumed here. It is not always True
            # FIXME: implement j!=i
            sdata[(j, i, tuple(-np.array(Rj)), u,
                   tuple(np.array(Ru) - Rj))] = val
        self.data.update(sdata)

    def weight(self, Rj, Ru, cutoff=sqrt(3) + 0.001):
        Rs = []
        ws = []
        T = np.diag(self.scell)
        dmin = 100.0
        R = None
        for x in (Ru[0], Ru[0] - T[0]):
            for y in (Ru[1], Ru[1] - T[1]):
                for z in (Ru[2], Ru[2] - T[2]):
                    di = np.linalg.norm((x, y, z))
                    dj = np.linalg.norm(np.array((x, y, z)) - np.array(Rj))
                    if (di + dj < dmin):
                        dmin = di + dj
                        R = (x, y, z)
        if dmin < 4:
            Rs.append(R)
            ws.append(1.0)
        return Rs, ws

    def weight0(self, Rj, Ru, cutoff=sqrt(3) + 0.001):
        Rs = []
        ws = []
        T = np.diag(self.scell)
        for x in (Ru[0], Ru[0] - T[0]):
            for y in (Ru[1], Ru[1] - T[1]):
                for z in (Ru[2], Ru[2] - T[2]):
                    di = np.linalg.norm((x, y, z))
                    dj = np.linalg.norm(np.array((x, y, z)) - np.array(Rj))
                    Rs.append((x, y, z))
                    if (di < cutoff) or (dj < cutoff):
                        ws.append(1.0)
                    else:
                        ws.append(0.0)
        return Rs, ws

    def weight1(self, Rj, Ru, cutoff=sqrt(3) + 0.001):
        Rs = [Ru]
        ws = [1.0]
        return Rs, ws

    def write_netcdf_scalarij(self, fname):
        self.ilist = []
        self.jlist = []
        self.ulist = []
        self.Rjlist = []
        self.Rulist = []
        self.vallist = []
        self.ndata = 0
        for key, val in self.data.items():
            i, j, Rj, u, Ru = key
            for uu in range(3):
                if abs(val[uu]) > 5e-3:
                    for di in range(3):
                        self.ndata += 1
                        self.ilist.append((i-1)*3+di+1)
                        self.jlist.append((j-1)*3+di+1)
                        self.ulist.append((u - 1) * 3 + uu + 1)
                        self.Rjlist.append(Rj)
                        self.Rulist.append(Ru)
                        self.vallist.append(val[uu])

        ds = Dataset(fname, "w")
        # dimensions
        ds.createDimension('Oiju_ndata', self.ndata)
        ds.createDimension('three', 3)
        ds.createDimension('natom', len(self.atoms))
        ds.createDimension('nspin', self.nspin)

        # vars
        v_qpoint = ds.createVariable(varname='ref_spin_qpoint', datatype='f8', dimensions=('three'))
        v_axis= ds.createVariable(varname='ref_spin_rotate_axis', datatype='f8', dimensions=('three'))
        v_ilist = ds.createVariable(
            varname='Oiju_ilist', datatype='i4', dimensions=('Oiju_ndata'))
        v_jlist = ds.createVariable(
            varname='Oiju_jlist', datatype='i4', dimensions=('Oiju_ndata'))
        v_ulist = ds.createVariable(
            varname='Oiju_ulist', datatype='i4', dimensions=('Oiju_ndata'))
        v_Rjlist = ds.createVariable(
            varname='Oiju_Rjlist',
            datatype='i4',
            dimensions=('Oiju_ndata', 'three'))
        v_Rulist = ds.createVariable(
            varname='Oiju_Rulist',
            datatype='i4',
            dimensions=('Oiju_ndata', 'three'))
        v_vallist = ds.createVariable(
            varname='Oiju_vallist', datatype='f8', dimensions=('Oiju_ndata'))



        # long names and units
        v_qpoint.setncatts({"long_name": u"reference spin qpoint"})
        v_axis.setncatts({"long_name": u"reference spin rotation axis"})

        v_ilist.setncatts({
            "long_name":
            u"index i in Oiju, id_spin for isotropic Jij, start from 1"
        })
        v_jlist.setncatts({
            "long_name":
            u"index j in Oiju, id_spin for isotropic Jij, start from 1"
        })
        v_ulist.setncatts({
            "long_name":
            u"index u in Oiju, including id_atom and xyz, start from 1"
        })
        v_Rjlist.setncatts({"long_name": u"R vector of j in Oiju"})
        v_Rulist.setncatts({"long_name": u"R vector of u in Oiju"})
        v_vallist.setncatts({
            "long_name": u"value of Oiju term, 1d vector.",
            "unit": u'eV/Angstrom'
        })

        # save
        v_qpoint[:]=np.array([0.5, 0.5, 0.5])
        v_axis[:]=np.array([1.0, 0.0, 0.0])
        v_ilist[:] = self.ilist
        v_jlist[:] = self.jlist
        v_ulist[:] = self.ulist
        v_Rjlist[:, :] = self.Rjlist
        v_Rulist[:, :] = self.Rulist
        v_vallist[:] = self.vallist
        ds.close()

    def write_netcdf(self, fname):
        self.ilist = []
        self.jlist = []
        self.ulist = []
        self.Rjlist = []
        self.Rulist = []
        self.vallist = []
        self.ndata = 0
        for key, val in self.data.items():
            i, j, Rj, u, Ru = key
            for ii in range(3):
                for uu in range(3):
                    if abs(val[uu]) > 1e-3:
                        self.ndata += 1
                        self.ilist.append((i - 1) * 3 + ii + 1)
                        self.jlist.append((j - 1) * 3 + ii + 1)
                        self.ulist.append((u - 1) * 3 + uu + 1)
                        self.Rjlist.append(Rj)
                        self.Rulist.append(Ru)
                        self.vallist.append(val[uu])

        ds = Dataset(fname, "w")
        # dimensions
        ds.createDimension('ndata', self.ndata)
        ds.createDimension('three', 3)
        ds.createDimension('natom', len(self.atoms))
        ds.createDimension('nspin', self.nspin)

        # vars
        v_qpoint = ds.createVariable(varname='ref_spin_qpoint', datatype='f8', dimensions=('three'))
        v_axis= ds.createVariable(varname='ref_spin_rotate_axis', datatype='f8', dimensions=('three'))

        v_ilist = ds.createVariable(
            varname='Oiju_ilist', datatype='i4', dimensions=('ndata'))
        v_jlist = ds.createVariable(
            varname='Oiju_jlist', datatype='i4', dimensions=('ndata'))
        v_ulist = ds.createVariable(
            varname='Oiju_ulist', datatype='i4', dimensions=('ndata'))
        v_Rjlist = ds.createVariable(
            varname='Oiju_Rjlist',
            datatype='i4',
            dimensions=('ndata', 'three'))
        v_Rulist = ds.createVariable(
            varname='Oiju_Rulist',
            datatype='i4',
            dimensions=('ndata', 'three'))
        v_vallist = ds.createVariable(
            varname='Oiju_vallist', datatype='f8', dimensions=('ndata'))

        # long names and units
        v_qpoint.setncatts({"long_name": u"reference spin qpoint"})
        v_axis.setncatts({"long_name": u"reference spin rotation axis"})
        v_ilist.setncatts({
            "long_name":
            u"index i in Oiju, including id_spin and xyz, start from 1"
        })
        v_jlist.setncatts({
            "long_name":
            u"index j in Oiju, including id_spin and xyz, start from 1"
        })
        v_ulist.setncatts({
            "long_name":
            u"index u in Oiju, including id_atom and xyz, start from 1"
        })
        v_Rjlist.setncatts({"long_name": u"R vector of j in Oiju"})
        v_Rulist.setncatts({"long_name": u"R vector of u in Oiju"})
        v_vallist.setncatts({
            "long_name": u"value of Oiju term, 1d vector.",
            "unit": u'eV/Angstrom'
        })

        # save
        v_qpoint[:]=np.array([0.5, 0.5, 0.5])
        v_axis[:]=np.array([1.0, 0.0, 0.0])

        v_ilist[:] = self.ilist
        v_jlist[:] = self.jlist
        v_ulist[:] = self.ulist
        v_Rjlist[:, :] = self.Rjlist
        v_Rulist[:, :] = self.Rulist
        v_vallist[:] = self.vallist
        ds.close()


def read_Oiju_netcdf(fname):
    ds = Dataset(fname, "r")
    nspin = ds.dimensions['nspin'].size
    natom = ds.dimensions['natom'].size
    nnz = ds.dimensions['Oiju_ndata'].size
    v_ilist = ds.variables['Oiju_ilist'][:] - 1
    v_jlist = ds.variables['Oiju_jlist'][:] - 1
    v_ulist = ds.variables['Oiju_ulist'][:] - 1
    v_Rjlist = ds.variables['Oiju_Rjlist'][:, :]
    v_Rulist = ds.variables['Oiju_Rulist'][:, :]
    v_vallist = ds.variables['Oiju_vallist'][:]
    ds.close()
    return nspin, natom, nnz, v_ilist, v_jlist, v_ulist, v_Rjlist, v_Rulist, v_vallist


def read_forces(fname):
    """
    read forces from vasprun.xml file
    """
    tree = ET.parse(fname)
    root = tree.getroot()
    found = False
    for block in root.findall('.//varray'):
        if block.get('name') == 'forces':
            found = True
            forces = []
            for v in block.findall('v'):
                forces.append([float(x) for x in v.text.split()])
    if not found:
        raise IOError("forces not found in file %s" % fname)
    return np.array(forces, dtype=float)


def sign(x):
    if x == 1:
        return ''
    else:
        return '-'


def name_template(i, j, si, sj):
    return 'supercell444/flip_Mn%s%s_%s3_%s3/vasprun.xml' % (i, j, sign(si),
                                                             sign(sj))


def test():
    from pyDFTutils.perovskite.cubic_perovskite import gen_primitive
    from pyDFTutils.ase_utils import symbol_number
    from ase.io import read
    patoms = gen_primitive(
        name="SrMnO3",
        latticeconstant=7.6266083947907566 / 2,
        mag_order='FM',
        m=3)
    syms = list(symbol_number(patoms).keys())
    atoms = read('./supercell444/flip_Mn12_-3_-3/POSCAR')
    #ilist, Rlist = map_to_primitive(atoms, patoms)

    d = OijuFitter(atoms=patoms, sc_atoms=atoms, scell=np.eye(3) * 4, nspin=1)
    d.fit(
        ijdict={
            (1, 2): (1, 1, (0, 0, 1)),
            (1, 3): (1, 1, (0, 1, 0)),
            (1, 5): (1, 1, (1, 0, 0))
        },
        fname_template=name_template)
    d.write_netcdf_scalarij(fname='Oiju_scalarij.nc')

    read_Oiju_netcdf(fname='Oiju_scalarij.nc')


if __name__ == '__main__':
    test()
