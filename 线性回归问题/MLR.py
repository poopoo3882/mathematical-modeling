import numpy as np
from scipy.stats import f


class MLR:
    def __init__(self, X, Y, intercept=True):
        self.X = X
        self.Y = Y
        self.intercept = intercept

    def fit(self):
        if self.intercept:
            ones = np.ones(len(self.X))
            X = np.c_[ones, self.X]
        else:
            X = self.X
        self.A = np.linalg.inv(X.T @ X) @ X.T @ self.Y

    def predict(self, Xnew):
        if self.intercept:
            ones = np.ones(len(Xnew))
            X = np.c_[ones, Xnew]
        else:
            X = Xnew
        self.yhat = X @ self.A
        return self.yhat

    def ftest(self, alpha):
        size = self.X.shape
        n = size[0]  # 求解出矩阵的行数 即样本数
        f_arfa = f.isf(alpha, 1, n - 2)  # 调用isf求解出临界值f分布
        Y_avg = self.Y.sum() / n
        U = ((self.yhat - Y_avg) ** 2).sum()
        Qe = ((self.Y - self.yhat) ** 2).sum()  # 利用sum方法与数组运算求解各参数
        F = U / (Qe / (n - 2))
        if F > f_arfa:
            ans = True
        else:
            ans = False  # 比较 得出是否可信 并返回各参数
        return F, f_arfa, ans


    def Ftest(self, alpha):  # MLR类的方法
        n = len(self.X)  # 样本数
        k = self.X.shape[-1]  # 获取变量数
        f_arfa = f.isf(alpha, k, n - k - 1)  # f临界值

        Yaver = self.Y.mean(axis=0)
        Yhat = self.predict(self.X)  # 拟合的y值
        U = ((Yhat - Yaver) ** 2).sum(axis=0)
        Qe = ((self.Y - Yhat) ** 2).sum(axis=0)

        F = (U / k) / (Qe / (n - k - 1))
        answer = ['F临界值:', f_arfa]

        if self.Y.ndim == 1:
            answer.append(['函数F值:', F])
        else:
            for i in range(len(F)):
                answer.append(['函数' + str(i + 1) + '的F值:', F[i]])

        return answer
