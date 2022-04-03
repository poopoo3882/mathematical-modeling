from NGA import NGA
import numpy as np

def f(v):  # 目标优化函数
    x1,x2,x3=v
    y=(x1-4)**2+2*(x2-3)**2+(x3-4)**2 # 要求解的方程
    return y

def f2(v): # 黑枸杞问题
    global t , y      # t,y  是全局变量，记录时间和实测吸光值
    a,k1,k2=v       # 一个解，带3个参数
    pred=a * k1 * (np.exp(-k1 * t) - np.exp(-k2 * t)) / (k2 - k1)
    error=y - pred
    s=np.sum(error**2)   # 误差的平方，作为评估值，越小越好
    return s


nga=NGA(50,3,f,5,95,1000)
nga.solve()
ans=nga.getAnswer()
print(ans)