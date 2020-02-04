import numpy as np
from pythtb import w90
from .unfolder import Unfolder
from .plotphon.plot import plot_band_weight
import matplotlib.pyplot as plt


class wannier_unfolder(object):
    def __init__(self, tbmodel, labels,  sc_matrix):
        self.model = tbmodel
        self.labels = labels
        self.sc_matrix = sc_matrix
        self.cell=self.model._lat
        self.positions=self.model._orb

    def unfold(self, kpts):
        self.evals, self.evecs = self.model.solve_all(
            k_list=kpts, eig_vectors=True)
        positions = self.model._orb
        # tbmodel: evecs[iband, ikpt, iorb]
        # unfolder: [ikpt, iorb, iband]
        self.unf = Unfolder(
            cell=self.cell,
            basis=self.labels,
            positions=self.positions,
            supercell_matrix=self.sc_matrix,
            #eigenvectors=np.swapaxes(self.evecs, 0, 1),
            eigenvectors=np.swapaxes(np.swapaxes(self.evecs, 0, 1), 1, 2),
            qpoints=kpts)
        #x=np.arange(len(kpts)) 
        #weights=self.unf.get_weights()
        #ax=plot_band_weight([list(x)]*self.evals.shape[1],self.evals.T , weights[:,:].T*0.98+0.000001,xticks=[['G'], [0]],style='alpha' )
        #plt.show()
        return self.unf.get_weights()

    def plot_unfolded_band(
            self,
            kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                               [0, 0, 0], [.5, .5, .5]]),
            knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            npoints=200,
            ax=None, ):
        """
        plot the projection of the band to the basis
        """
        if ax is None:
            fig, ax = plt.subplots()
        from ase.dft.kpoints import bandpath
        kvectors = [np.dot(k, self.sc_matrix) for k in kvectors]
        kpts, x, X = bandpath(kvectors, self.cell, npoints)
        kslist = [x] * len(self.positions)
        efermi = 0.0
        wkslist=self.unfold(kpts).T * 0.98 +0.01
        ekslist = self.evals
        #wkslist = np.abs(self.get_projection(orb, spin=spin, eigenvecs=evecs))
        ax = plot_band_weight(
                kslist,
                ekslist,
                wkslist=wkslist,
                efermi=None,
                yrange=None,
                output=None,
                style='alpha',
                color='blue',
                axis=ax,
                width=20,
                xticks=None)
        for i in range(len(self.positions)):
            ax.plot(x, self.evals[i, :], color='gray', alpha=1, linewidth=0.1)

        #ax.axhline(self.get_fermi_level(), linestyle='--', color='gray')
        ax.set_xlabel('k-point')
        ax.set_ylabel('Energy (eV)')
        ax.set_xlim(x[0], x[-1])
        ax.set_xticks(X)
        ax.set_xticklabels(knames)
        for x in X:
            ax.axvline(x, linewidth=0.6, color='gray')
        return ax

# Below are example. Should be moved to examples.

def test_nodefect():
    w90reader = w90(path='data_nodefect', prefix='wannier90')
    #w90reader = w90(path='data', prefix='wannier90')
    tb = w90reader.model(min_hopping_norm=0.05)
    labels = ['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4
    scmat=[[1,-1,0],[1,1,0],[0,0,2]]
    u = wannier_unfolder(tb, labels, sc_matrix=scmat)
    u.plot_unfolded_band()
    plt.savefig('STO_nodefect.pdf')
    plt.savefig('STO_nodefect.png')
    plt.show()

def test_defect():
    w90reader = w90(path='data', prefix='wannier90')
    tb = w90reader.model(min_hopping_norm=0.05)
    labels = ['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4
    scmat=[[1,-1,0],[1,1,0],[0,0,2]]
    u = wannier_unfolder(tb, labels, sc_matrix=scmat)
    u.plot_unfolded_band(
            kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                                [0, 0, 0], [.5, .5, .5]]),
            knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            npoints=200,
            ax=None, 
            )
    plt.savefig('STO_defect.pdf')
    plt.savefig('STO_defect.png')
    plt.show()

def run(path, prefix,  labels, scmat, output_figure, kvectors, knames, npoints=200, min_hopping_norm=0.0001):
    w90reader = w90(path=path, prefix=prefix)
    tb = w90reader.model(min_hopping_norm=min_hopping_norm)
    #labels = ['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4
    #scmat=[[1,-1,0],[1,1,0],[0,0,2]]
    u = wannier_unfolder(tb, labels, sc_matrix=scmat)
    u.plot_unfolded_band(
            #kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
            #                    [0, 0, 0], [.5, .5, .5]]),
            kvectors=kvectors,
            #knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
            knames=knames,
            npoints=npoints,
            ax=None, 
            )
    plt.savefig(output_figure)
    plt.show()

if __name__=="__main__":
    run(path='data',
        prefix='wannier90',
        scmat=[[1,-1,0], [1,1,0 ], [0,0,2]],
        output_figure='unfold.png', 
        labels = ['pz', 'px', 'py'] * 12 + ['dz2', 'dxy', 'dyz', 'dx2', 'dxz'] * 4,
        kvectors=np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0],
                         [0, 0, 0], [.5, .5, .5]]),
        knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'] )

#test_nodefect()
#test_defect()
