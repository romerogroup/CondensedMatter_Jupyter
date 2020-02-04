from abc import ABCMeta, abstractmethod

class AbstractPotential(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def coefficients(self):
        pass

    @abstractmethod
    def make_supercell(self, scmaker):
        pass

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    def get_forces(self):
        pass

    @abstractmethod
    def get_bfield(self):
        pass

    @abstractmethod
    def get_energy(self):
        pass


class ListHamiltonian(AbstractPotential):
    def __init__(self, nspin=None, natom=None ):
        self.pdict={}
        self.features


    def append(self, pot, name):
        if name in self.pdict:
            raise ValueError("Potential name %s already used"%name)
        else:
            self.pdict[name]=pot

    def get_forces(self, ):
        forces
