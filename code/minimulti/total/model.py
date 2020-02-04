class model(object):
    def __init__(self):
        pass

    def add_submodel(self, submodel):
        pass

    def add_coupling(self, coupling, model1, model2):
        pass

    def set_params(self):
        pass

    def run(self):
        pass

class spin_lattice(model):
    def __init__(self):
        pass

    def add_lattice_model(self, lattice_model):
        pass

    def add_spin_model(self, spin_model):
        pass

    def add_spin_lattice_coupling(self, spin_lattice_coupling):
        pass

    def get_forces(self):
        forces=self.lattice.get_forces()
        forces+=self.slc.get_forces()
        return forces

    def run_spin_step(self, nstep):
        self.slc.update_positions()
        self.slc.get_new_J()
        for i in range(nstep):
            self.slc.run_one_step()

    def run_lattice_step(self, nstep):
        self.slc.update_spin()
        self.slc.get_forces(self.positions)
    def run(self):
        

    def get_slc_force(self):
        pass

    def refresh_slc_J(self):
        pass

    def get_total_energy(self):
        pass
