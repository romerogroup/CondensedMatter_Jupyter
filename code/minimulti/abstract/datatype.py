from collections import defaultdict
from functools import partial
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix, coo_matrix, lil_matrix
import copy


class COOlike(object):
    def __init__(self, shape, indices, data, nnz=None):
        self.shape = shape
        self.indices = np.array(indices, dtype='int')
        self.data = np.array(data)
        if nnz is None:
            self.nnz = len(self.indices)

    @property
    def row(self):
        return self.indices[:, 0]

    @property
    def col(self):
        return self.indices[:, 1]


class NDCOO(object):
    def __init__(self, shape, nnz=0):
        self.shape = shape
        self._nnz = nnz
        self._ind = []
        self._val = []

    def ind(self, axis):
        return np.asarray(self._ind)[:, axis]

    @property
    def val(self):
        return np.array(self._val)


class IFClike(defaultdict):
    """
    data like {(i, j, R): val, ...}
    """

    def __init__(self, genfunc=float):
        super(IFClike, self).__init__(genfunc)

    def __neg__(self):
        ret = copy.deepcopy(self)
        for key, val in ret.items():
            ret[key] = -val
        return ret

    def __add__(self, other):
        keys = set(self.keys()).union(set(other.keys()))
        ret = IFClike()
        for key in keys:
            ret[key] = self[key] + other[key]
        return ret

    def __sub__(self, other):
        keys = set(self.keys()).union(set(other.keys()))
        ret = IFClike()
        for key in keys:
            ret[key] = self[key] - other[key]
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        ret = IFClike()
        for key in self.keys():
            ret[key] = self[key] * other
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, x):
        ret = IFClike()
        for key in self.keys():
            ret[key] = self[key] / x
        return ret

    @property
    def ind_i(self):
        return [x[0] for x in self.keys()]

    @property
    def ind_j(self):
        return [x[1] for x in self.keys()]

    def R(self):
        return [x[2] for x in self.keys()]

    @property
    def val(self):
        return np.array(self.values())

    def to_Rijv(self, shape, dtype=float, sparse=True):
        ret = Rijv(shape=shape, sparse=sparse, dtype=dtype)
        for key, val in self.keys():
            i, j, R = key
            ret.add(R, i, j, val)
        return ret


class Rijv(defaultdict):
    def __init__(self, shape, sparse=False, dtype=float):
        assert len(shape)==2
        self.shape = tuple(shape)
        self.sparse = sparse
        self.dtype = dtype
        if sparse:
            #default_factory = lambda: dok_matrix(shape, dtype=dtype)
            default_factory = partial(lil_matrix, self.shape, dtype=dtype)
        else:
            #default_factory = lambda: np.zeros(shape, dtype=dtype)
            default_factory = partial(np.zeros, self.shape, dtype=dtype)
        super(Rijv, self).__init__(default_factory)

    def max_abs(self):
        ms=[np.max(np.abs(m)) for m in self.values()]
        return np.max(ms)

    def remove_zero(self, tol):
        for key, val in self.items():
            ind=(np.abs(val)<tol)
            if len(ind)==0:
                del self[key]
            val[ind]=0

    def emptycopy(self, dtype=None):
        if dtype is None:
            dtype=self.dtype
        return Rijv(shape=self.shape, sparse=self.sparse, dtype=dtype)

    def add(self, R, i, j, val):
        self[tuple(R)][i, j] += val

    def to_csr(self):
        if not self.sparse:
            for key, val in self.items():
                self[key] = csr_matrix(val)
        self.sparse = True

    def to_lil(self):
        if not self.sparse:
            for key, val in self.items():
                self[key] = lil_matrix(val)
        self.sparse = True



    def todense(self):
        if self.sparse:
            for key, val in self.items():
                self[key] = val.todense()
        self.sparse = False
        return self

    def __neg__(self):
        ret = self.emptycopy()
        for key, val in ret.items():
            ret[key] = -val
        return ret

    def __add__(self, other):
        keys = set(self.keys()).union(set(other.keys()))
        ret=self.emptycopy()
        for key in keys:
            ret[key] = self[key] + other[key]
        return ret

    def __sub__(self, other):
        keys = set(self.keys()).union(set(other.keys()))

        ret = self.emptycopy()
        for key in keys:
            ret[key] = self[key] - other[key]
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, complex):
            dtype=complex
        else:
            dtype=self.dtype
        ret = self.emptycopy(dtype=dtype)
        for key in self.keys():
            ret[key] = self[key] * other
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, x):
        if isinstance(x, complex):
            dtype=complex
        else:
            dtype=self.dtype

        ret = self.emptycopy()
        for key in self.keys():
            ret[key] = self[key] / x
        return ret

    def make_supercell(self, scmaker, sparse=False, tol=1e-4):
        assert self.shape[0] == self.shape[
            1], "Can only make supercell for n*n matrix"
        self.remove_zero(tol=tol)
        sc_nbasis = scmaker.ncell * self.shape[0]
        nbasis=self.shape[0]
        ret=Rijv(shape=(sc_nbasis, sc_nbasis), sparse=sparse, dtype=self.dtype)
        print(len(self))
        print(sc_nbasis)
        for c, cur_sc_vec in enumerate(
                scmaker.sc_vec):  # go over all super-cell vectors
            for R, mat in self.items():
                ind_R=np.array(R, dtype=int)
                sc_part, pair_ind = scmaker._sc_R_to_pair_ind(
                    tuple(ind_R + cur_sc_vec))
                coomat=coo_matrix(mat)
                mtmp=np.zeros((sc_nbasis, sc_nbasis), dtype=self.dtype)
                #print("sc_part,",sc_part)
                for i, j, val in zip(coomat.row, coomat.col, coomat.data):
                    sc_i = i + c * nbasis
                    sc_j = j + pair_ind * nbasis
                    mtmp[sc_i,sc_j]=val
                ret[tuple(sc_part)] += mtmp
        ret.remove_zero(tol=tol)
        return ret

    def reciprocal(self, q):
        ret=np.zeros(self.shape, dtype=complex)
        for R, mat in self.items():
            ret+= np.exp(2.0j*np.pi*np.dot(R, q))*mat
        return ret

    def to_dense_data(self):
        nR=len(self)
        val_array= np.empty(shape=(nR, self.shape[0], self.shape[1]))
        for i, mat in enumerate(self.values()):
            val_array[i,:,:] = mat
        return {'shape': self.shape,
                'Rarray': list(self.keys()),
                'val' : val_array,
        }

    def to_netcdf(self, fname):
        pass

    def save_as_dict(self, ret):
        ret=dict(self)

    @staticmethod
    def from_netcdf(fname):
        pass


def test_Rijv():
    a = Rijv(shape=(3, 3), sparse=True)
    a.add(R=(1, 2, 3), i=1, j=1, val=0.5)
    a.add(R=(0, 0, 1), i=0, j=2, val=3)
    a.todense()
    print(a)
    print(a / 9)


if __name__ == '__main__':
    test_Rijv()
