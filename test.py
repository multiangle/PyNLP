
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
c_1 = csr_matrix((data, (row, col)),shape=(3,5))
print(c_1)
print(c_1.shape)
print('----------------')

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([4, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
c_2 = csr_matrix((data, (row, col)) )
print(c_2)
print(c_2.shape)
print('----------------')

c_3 = sparse.vstack([c_1,c_2])
print(c_3[(1,3),:])