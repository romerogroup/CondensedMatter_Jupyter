class SpinParams(object):
    def __init__(self):
        """
        setting default values for spin dynamics
        """
        self.path='./'
        self.temperature=0.0
        self.time_step=1e-4
        self.total_time=1e-4
        self.hist_fname="spin_hist.nc"

    def set(self, **kwargs):
        self.__dict__.update(kwargs)
        self._validate()

    def _validate(self):
        pass

