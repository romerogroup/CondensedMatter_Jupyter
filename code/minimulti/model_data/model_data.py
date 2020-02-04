import os

class ModelData(object):
    def __init__(self, lattice_mode_names):
        self.lattice_mode_names = sorted(lattice_mode_names)

    def base_name(self, mode_dict, path='./', prefix='noname', spin='nospin'):
        name = ''
        for key in self.lattice_mode_names:
            if key in mode_dict:
                name += "_%s_%.3f" % (key, mode_dict[key])
            else:
                name += "_%s_%.3f" % (key, 0.0)
        name = os.path.join(path, prefix + name + '_' + spin )
        return name

    def wann_name(self, mode_dict, path='./', prefix='noname', spin='nospin'):
        return self.base_name(mode_dict=mode_dict, path=path, prefix=prefix, spin=spin) + '.wannc'



def test():
    mdata = ModelData(lattice_mode_names=['rot', 'br'])
    name = mdata.wann_name({
        'rot': 0.1,
        'br': 0.3
    },
                           prefix='dp',
                           spin='up',
                           path='/home/hexu')
    print(name)


if __name__=='__main__':
    test()
