import numpy as np
import numpy.linalg as LA

class pca_assister:
    dim:int
    avr:np.ndarray
    eig_vals:np.ndarray
    eig_vecs:np.ndarray
    datas_mat:np.ndarray

    def __init__(self, datas:list[list[float]]):
        x = np.array(datas)

        self.dim = x.shape[1]
        datas_size = x.shape[0]

        self.avr = np.ndarray((self.dim,1))

        for i in range(self.dim):
            s = sum(x[:,i])
            self.avr[i,0] = s / float(datas_size)
            
        x = x.T
        x -= self.avr

        m = x @ x.T
        m /= float(datas_size)
        (self.eig_vals, self.eig_vecs) = LA.eig(m)

        s = np.argsort(-self.eig_vals)

        self.eig_vals = self.eig_vals[s]

        self.eig_vecs = self.eig_vecs.T
        self.eig_vecs = self.eig_vecs[s]
        self.eig_vecs = self.eig_vecs.T

        self.datas_mat = x


    def _pca_simple(self, d:int) -> list[list[float]]:
        m = self.eig_vecs[:,0:d]
        v = self.avr.T @ m
        r = self.datas_mat.T @ m
        r -= v

        return r.tolist()


    def pca_dim(self, how_many_dim:int) -> list[list[float]]:
        if how_many_dim <= 0:
            return []
    
        d = min(self.dim, how_many_dim)

        return self._pca_simple(d)

"""
def pca_pareto(datas:list[list[float]], ratio:float):
    if ratio <= 0.0:
        return []
    

"""
