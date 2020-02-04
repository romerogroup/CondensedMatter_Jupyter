import numpy as np
from ase.atoms import Atoms
from ase.units import J
from minimulti.constants import gyromagnetic_ratio
from ase.data import atomic_masses



class BaseSpinModelParser(object):
    """
    SpinModelParser: a general model for spin model file parser.
    """

    def __init__(self, fname):
        self._fname = fname
        self.damping_factors = []
        self.gyro_ratios = []
        self.index_spin = []
        self.cell = None
        self.zions = []
        self.masses = []
        self.positions = []
        self.spinat = []
        self._exchange = {}
        self._dmi = {}
        self._sia = {}
        self._bilinear = {}
        self._parse(fname)
        self.lattice = Atoms(
            positions=self.positions, masses=self.masses, cell=self.cell)

    def _parse(self, fname):
        raise NotImplementedError("parse function not implemented yet")

    def get_atoms(self):
        return self.atoms

    def _spin_property(self, prop):
        return [
            prop[i] for i in range(len(self.index_spin))
            if self.index_spin[i] > 0
        ]

    @property
    def atoms(self):
        return self.lattice

    @property
    def natom(self):
        return len(self.lattice)

    @property
    def nspin(self):
        return len(self.spin_spinat)

    @property
    def spin_positions(self):
        return np.array(self._spin_property(self.positions), dtype='float')

    @property
    def spin_zions(self):
        return np.array(self._spin_property(self.zions), dtype='int')

    @property
    def spin_spinat(self):
        return np.array(self._spin_property(self.spinat), dtype='float')

    @property
    def spin_damping_factors(self):
        return np.array(
            self._spin_property(self.damping_factors), dtype='float')

    @property
    def spin_gyro_ratios(self):
        return np.array(self._spin_property(self.gyro_ratios), dtype='float')

    def get_index_spin(self):
        return self.index_spin

    def exchange(self, isotropic=True):
        if isotropic:
            iso_jdict = {}
            for key, val in self._exchange.items():
                iso_jdict[key] = val[0]
            return iso_jdict
        else:
            return self._exchange

    @property
    def dmi(self):
        return self._dmi

    @property
    def has_exchange(self):
        return bool(len(self._exchange))

    @property
    def has_dmi(self):
        return bool(len(self._dmi))

    def write_netcdf(self, fname):
        from scipy.io.netcdf import netcdf_file
        from netCDF4 import Dataset
        #with netcdf_file(fname, 'w') as myfile:
        with Dataset(fname, 'w') as myfile:
            myfile.createDimension("nspin", self.nspin)
            myfile.createDimension("natom", self.natom)
            myfile.createDimension("three", 3)

            if self._exchange !={}:
                myfile.createDimension("spin_exchange_nterm", len(self.exchange()))
            if self._bilinear !={}:
                myfile.createDimension("spin_bilinear_nterm", len(self._bilinear))
            if self._dmi!={}:
                myfile.createDimension("spin_dmi_nterm", len(self._dmi))
            if self._sia!={}:
                myfile.createDimension("spin_sia_nterm", len(self._sia))

            cell = myfile.createVariable("cell", 'f8', ("three", "three"))
            cell.unit = "Angstrom"
            cell[:] = self.cell

            xcart = myfile.createVariable("xcart", "f8", ("natom", "three"))
            xcart.unit = "Angstrom"
            xcart[:] = np.array(self.positions)

            spinat = myfile.createVariable("spinat", "f8", ("natom", "three"))
            spinat[:] = np.array(self.spinat)

            index_spin = myfile.createVariable("index_spin", "i4", ("natom", ))
            index_spin[:] = self.index_spin

            gyroratio = myfile.createVariable("gyroratio", "f8", ("natom", ))
            gyroratio[:] = np.array(self.gyro_ratios)/gyromagnetic_ratio

            gilbert_damping = myfile.createVariable("gilbert_damping", "f8",
                                                    ("natom", ))
            gilbert_damping[:] = np.array(self.damping_factors)

            if self._exchange != {}:
                nterm = len(self._exchange)
                ilist = np.zeros(nterm, dtype=int)
                jlist = np.zeros(nterm, dtype=int)
                Rlist = np.zeros((nterm, 3), dtype=int)
                vallist = np.zeros((nterm, 3), dtype=float)
                for ii, (key, val) in enumerate(self._exchange.items()):
                    i, j, R = key
                    ilist[ii] = i
                    jlist[ii] = j
                    Rlist[ii] = R
                    vallist[ii] = val * J

                spin_exchange_ilist = myfile.createVariable(
                    "spin_exchange_ilist", "i4", ("spin_exchange_nterm", ))
                spin_exchange_jlist = myfile.createVariable(
                    "spin_exchange_jlist", "i4", ("spin_exchange_nterm", ))
                spin_exchange_Rlist = myfile.createVariable(
                    "spin_exchange_Rlist", "i4", ("spin_exchange_nterm", "three"))
                spin_exchange_vallist = myfile.createVariable(
                    "spin_exchange_vallist", "f8",
                    ("spin_exchange_nterm", "three"))
                spin_exchange_vallist.unit = "eV"

                spin_exchange_ilist[:] = np.array(ilist)+1
                spin_exchange_jlist[:] = np.array(jlist)+1
                spin_exchange_Rlist[:] = np.array(Rlist)
                spin_exchange_vallist[:] = np.array(vallist)

            if self._bilinear!= {}:
                nterm = len(self._bilinear)
                ilist = np.zeros(nterm, dtype=int)
                jlist = np.zeros(nterm, dtype=int)
                Rlist = np.zeros((nterm, 3), dtype=int)
                vallist = np.zeros((nterm, 3, 3), dtype=float)
                for ii, (key, val) in enumerate(self._bilinear.items()):
                    i, j, R = key
                    ilist[ii] = i
                    jlist[ii] = j
                    Rlist[ii] = R
                    vallist[ii] = val * Joule

                spin_bilinear_ilist = myfile.createVariable(
                    "spin_bilinear_ilist", "i4", ("spin_bilinear_nterm", ))
                spin_bilinear_jlist = myfile.createVariable(
                    "spin_bilinear_jlist", "i4", ("spin_bilinear_nterm", ))
                spin_bilinear_Rlist = myfile.createVariable(
                    "spin_bilinear_Rlist", "i4", ("spin_bilinear_nterm", "three"))
                spin_bilinear_vallist = myfile.createVariable(
                    "spin_bilinear_vallist", "f8",
                    ("spin_bilinear_nterm", "three", "three"))
                spin_bilinear_vallist.unit = "eV"

                spin_bilinear_ilist[:] = np.array(ilist)+1
                spin_bilinear_jlist[:] = np.array(jlist)+1
                spin_bilinear_Rlist[:] = np.array(Rlist)
                spin_bilinear_vallist[:] = vallist
