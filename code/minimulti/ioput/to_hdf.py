#!/usr/bin/env python3
import tbmodels 

def to_hdf5_file(path, prefix, hdf_fname):
    m=tbmodels.Model.from_wannier_folder(folder=path, prefix=prefix)
    m.to_hdf5_file(hdf_fname)

to_hdf5_file('./', 'wannier90', "wann.hdf5")
