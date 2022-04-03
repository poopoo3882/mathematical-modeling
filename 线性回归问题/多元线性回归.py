import numpy as np

data = np.loadtxt('alldata.txt')
x = data[:-1, :].T
y = data[-1, :]

from PCR import PCR

pcr = PCR(x, y)
print(pcr.confirmPCs())
k = int(input('请确认独立变量数'))
pcr.fit(k)
yHat = pcr.predict(x)
err = abs(y - yHat) / y * 100
print(err)
