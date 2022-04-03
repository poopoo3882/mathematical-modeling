from PCA import PCA
from MLR import MLR


class PCR:
    def __init__(self, X, Y, intercept=True):
        self.X = X
        self.Y = Y
        self.intercept = intercept

    def confirmPCs(self):
        self.pca = PCA(self.X)
        compare = self.pca.Decompose()
        return compare

    def fit(self, k):
        self.T, self.P = self.pca.ConfirmTP(k)
        self.mlr = MLR(self.T, self.Y, self.intercept)
        self.mlr.fit()

    def predict(self, Xnew):
        Tnew = Xnew @ self.P
        yHat = self.mlr.predict(Tnew)
        return yHat
