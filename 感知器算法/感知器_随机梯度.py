# X ,y  a,b
import numpy as np


def loss(X, y, a, b):
    temp = (X @ a + b) * y
    a = - temp[temp < 0].sum()
    return a, temp


x1 = [[5.1, 3.5], [4.9, 3.], [4.7, 3.2], [4.6, 3.1], [5., 3.6], [5.4, 3.9],
      [4.6, 3.4], [5., 3.4], [4.4, 2.9], [4.9, 3.1]]
x2 = [[5.5, 2.6], [6.1, 3.], [5.8, 2.6], [5., 2.3], [5.6, 2.7],
      [5.7, 3.], [5.7, 2.9], [6.2, 2.9], [5.1, 2.5], [5.7, 2.8]]
x = x1 + x2
y = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y += y1
X = np.array(x)
y = np.array(y)
a = np.random.random(2) - 0.5
b = np.random.random()
lr = 1.0
for i in range(100):
    loss1, exp = loss(X, y, a, b)
    if loss1 == 0.0:
        print(loss1, ':::', a, b)
        break
        #  找错误样本 ，按梯度修改a,b   exp:每个样本的损失值构成的数组（有正有负）
    wrongSampleNo = np.argwhere(exp < 0).flatten()  # argwhere():返回具备特点条件的数组元素的下标  flatten:将numpy数组展开为一维数组
    t = np.random.choice(wrongSampleNo, 3)  # 随机采3个样本
    for k in range(len(t)):
        a[0] += X[t[k]][0] * y[t[k]] * lr
        a[1] += X[t[k]][1] * y[t[k]] * lr
        b += y[t[k]] * lr
        # t为一个数组，其中元素为wrongSampleNo数组中的随机一个元素，表示需要改进的位置的下标
# 画图
xvalue = X[:, 0]  # 二维平面，x0作x轴，x1作y轴
xmin = min(xvalue)
xmax = max(xvalue)
xp = [xmin, xmax]
yp = [-a[0] / a[1] * xmin - b / a[1], -a[0] / a[1] * xmax - b / a[1]]
from pylab import *

cls1x = X[y == -1, 0]  # 第一类样本的x轴坐标,  用y==-1  过滤数据得到
cls1y = X[y == -1, 1]  # 第一类样本的y轴坐标
cls2x = X[y == 1, 0]
cls2y = X[y == 1, 1]
plot(cls1x, cls1y, 'b^')  # 第一类的散点图
plot(cls2x, cls2y, 'r^')  # 第二类
plot(xp, yp)  # 画分割线
show()
