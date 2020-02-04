class abstract_model(object):
    def __init__(self):
        self._variable_names=[]
        self._variables=[]
        self._params=[]
        self._atoms=None
        pass

    def read_primitive_cell_model(self, fname):
        """
        read model from file.
        """
        pass

    def save_primitive_cell_model(self, fname):
        """
        save model to file
        """
        pass

    def get_variable_names(self):
        return self._variable_names

    def get_variables(self):
        return self._variables

    def set_params(self, params):
        self.par


    def get_variables(self):
        return self.variables

    def set_params(self, params):
        pass

    def get_primitive_atoms(self):
        pass

    def get_atoms(self):
        return self._atoms

    def make_supercell(self):
        pass

