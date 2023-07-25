import numpy as np
import numpy.linalg as LA

class pca_result:
    dim:int
    avr:list[float]
    weights:list[float]
    eig_vals:list[float]
    vectors:list[list[float]]

    def __init__(self, data:list[list[float]]):
        x = np.array(data)

        self.dim = x.shape[1]
        data_size = x.shape[0]

        for i in range(data_size):
            s = sum(x[i])
            x[i] /= s
        x = x.T

        self.weights = []
        self.eig_vals = []
        self.vectors = []

        for i in range(self.dim):
            m = x @ x.T
            (e_vals, e_vecs) = LA.eig(m)
            argmax = 0
            max = e_vals[0]
            for (j,v) in enumerate(e_vals):
                if v > max:
                    max = v
                    argmax = j
                    break

            self.eig_vals.append(max)

            norm = LA.norm(e_vecs[argmax])
            self.vectors.append(e_vecs[argmax] / norm)

            x -= x @ self.vectors[i] @ self.vectors[i].T
