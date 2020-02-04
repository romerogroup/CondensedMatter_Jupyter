import unittest
import os
import numpy as np
from minimulti.utils.supercell import SupercellMaker, map_to_primitive
from ase.io import read
from pyDFTutils.perovskite.cubic_perovskite import gen_primitive
from pyDFTutils.ase_utils import symbol_number

class SCtest(unittest.TestCase):
    def test_map_to_primitive(self):
        sc_atoms=read('data/supercell.vasp')
        patoms = gen_primitive(
            name="SrMnO3",
            latticeconstant=7.6266083947907566 / 2,
            mag_order='FM',
            m=3)
        ilist, Rlist = map_to_primitive(sc_atoms, patoms)
        print(Rlist)


if __name__=='__main__':
    unittest.main()
