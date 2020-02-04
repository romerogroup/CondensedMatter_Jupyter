from abc import ABCMeta, abstractmethod


class States(object):
    def __init__(self, nspin=None, natom=None):
        self.nspin = nspin
        self.natom = natom

        if nspin is not None:
            self.S = np.zeros((nspin, 3), dtype=float)
        else:
            self.S = None

        if natom is not None:
            self.dx = np.zeros((natom, 3), dtype=float)
        else:
            self.dx = None



class AbstractManager(ABCMeta):
    @abstractmethod
    def __init__(self):
        self.potentials = potenial_list()
        self.movers = None
        self.structure = None
        self.sc_maker = None

    @abstractmethod
    def make_supercell(self):
        pass

    @abstractmethod
    def run_one_step(self):
        pass
