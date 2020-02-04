from minimulti.abstract import abstract_model

class lattice_model(abstract_model):
    def __init__(self):
        pass

    def get_atoms(self):
        return self.atoms

    def make_supercell(self):
        pass

    def set_params(self):
        pass

    def get_variables(self):
        pass

    def get_labels(self):
        pass

    def get_coefficient(self);
        pass

    def get_energy(self):
        pass

    def run_one_step(self):
        pass

    def run(self, steps=None):
        pass
