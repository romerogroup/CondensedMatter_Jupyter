import os
from netCDF4 import Dataset
import numpy as np


class SpinHistFile(object):
    def __init__(self, fname, nspin):
        self.fname = fname
        self.counter = 0
        self.nspin = nspin
        if os.path.exists(self.fname):
            os.remove(self.fname)
        ds = Dataset(self.fname, "w")
        ds.createDimension('ntime', None)
        ds.createDimension('three', 3)
        ds.createDimension('nspin', self.nspin)

        # vars
        self.S = ds.createVariable(
            varname='S', datatype='f8', dimensions=("ntime", "nspin", "three"))
        self.Ms = ds.createVariable(
            varname='M', datatype='f8', dimensions=("nspin"))
        self.spin_positions = ds.createVariable(
            varname='spin_xcart', datatype='f8', dimensions=("nspin", "three"))

        self.itime = ds.createVariable(
            varname='itime', datatype='i8', dimensions=("ntime"))

        self.time = ds.createVariable(
            varname='time', datatype='f8', dimensions=("ntime"))

        self.Rlist = ds.createVariable(
            varname="Rlist", datatype='i8', dimensions=("nspin", "three"))
        self.iprim = ds.createVariable(
            varname="iprim", datatype='i8', dimensions=("nspin"))

        # long names and units
        self.S.setncatts({"long_name": "Spin orientation"})
        self.spin_positions.setncatts(
            {"long_name": "cartesian coordinates of spins"})
        self.Ms.setncatts({"long_name": "Magnetic moment of each site"})
        self.Rlist.setncatts(
            {"long_name": "R vector from primitive to supercell"})
        self.iprim.setncatts({"long_name": "Primitive cell spin index"})
        self.itime.setncatts({"long_name": "index of time step"})
        self.time.setncatts({"long_name": "time", "unit": 's'})

        self.dataset = ds

    def write_cell(self, Ms, spin_positions, Rlist, iprim):
        self.Ms[:] = Ms
        self.Rlist[:, :] = Rlist
        self.iprim[:] = iprim
        self.spin_positions[:, :] = spin_positions

    def write_S(self, S, time, itime=None):
        if itime is None:
            itime = self.counter
        self.time[self.counter] = time
        self.itime[self.counter] = itime
        self.S[self.counter, :, :] = S
        self.counter += 1

    def close(self):
        self.dataset.close()


def read_hist(fname):
    ds = Dataset(fname, "r")
    return {
        "positions": ds.variables["spin_xcart"],
        "Rlist": ds.variables["Rlist"],
        "spin": ds.variables["S"],
        "time": ds.variables["time"],
        "total_spin": np.sum(ds.variables["S"], axis=1),
    }
    ds.close()
