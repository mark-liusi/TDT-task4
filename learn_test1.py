import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
#data
x= np.linspace(0, 10, 50)# 生成等间距数组
y= 2* x+3+ np.random.randn(50)

max_time= 1000
pianyi= 0.01
w= np.random.randn()
b= np.random.randn()

for i in range(max_time):
    #计算的y
    cal_y= w*x+ b
    loss= np.mean((cal_y- y)**2)#偏差计算，mean是用来计算均值的

    dw= np.mean(2*(cal_y- y)*x)#求偏导进行梯度下降
    db= np.mean(2*(cal_y- y))#求偏导进行梯度下降

    w-= dw*pianyi#
    b-= db*pianyi

    if i % 100 == 0:#每运算100次就输出一次，查看情况
        print(f"Epoch {i}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(w, b)
plt.scatter(x, y, label="Data")                  # 原始数据点
plt.plot(x, w * x + b, color="red", label="Fit") # 拟合直线
plt.legend()#生成图例
plt.show()#显示出来
