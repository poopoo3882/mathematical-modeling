# X ,y  a,b
import numpy as np


def loss(X, y, a, b):  # X:样本矩阵 每个个体有长、宽两种属性  y:样本分类，根据花的种类分为+1/-1  a,b:划分直线的参数
    temp = (X @ a + b) * y
    a = - temp[temp < 0].sum()  # temp<0得到一个新的数组，temp数组的每一项若<0则为true 若>0则为false,然后再用这个
    # 辅助数组去对temp进行数据过滤。即temp[temp<0],得到一个数值均小于0的数组
    return a, temp  # a :损失函数的值（负值之和）


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
    #  找错误样本 ，按梯度修改  a,b
    for k, v in enumerate(exp):  # exp:每个样本的损失值构成的数组  enumerate:遍历数组
        if v < 0:
            a[0] += X[k][0] * y[k] * lr
            a[1] += X[k][1] * y[k] * lr
            b += y[k] * lr  # 梯度调整 此py文件采用每遇到一个错误样本就调一次直线参数
# 利用matplotlib画图
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
