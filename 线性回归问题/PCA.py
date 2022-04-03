import numpy as np


class PCA:
    def __init__(self, X):
        self.X = X

    def Decompose(self):
        U, S, V = np.linalg.svd(self.X, full_matrices=False)
        self.lamda = lamda = S ** 2
        self.P = V.T
        self.T = U * S
        compare = lamda[:-1] / lamda[1:]
        return compare

    def ConfirmTP(self, k):
        # 给定主成分数k，得到去处噪声后的得分T和载荷P
        T = self.T[:, :k]
        P = self.P[:, :k]
        return T, P
