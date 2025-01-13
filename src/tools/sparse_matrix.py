import numpy as np
from scipy.sparse import csr_matrix

class SparseMatrix:
    def __init__(self, data, indices, indptr, shape):
        self.matrix = csr_matrix((data, indices, indptr), shape=shape)
    
    @property
    def shape(self):
        return self.matrix.shape

    def to_dense(self):
        return self.matrix.toarray()

    def dot(self, vector):
        return self.matrix.dot(vector)

    def diagonal(self):
        return self.matrix.diagonal()

    def __getitem__(self, key):
        return self.matrix[key]

    @staticmethod
    def from_dense(dense_matrix):
        csr = csr_matrix(dense_matrix)
        return SparseMatrix(csr.data, csr.indices, csr.indptr, csr.shape)