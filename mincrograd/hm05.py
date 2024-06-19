import numpy as np
from numpy.linalg import norm
from engine import Value


# 使用梯度下降法尋找函數最低點


def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    p = p0.copy()
    for i in range(max_loops):
        fp = f(p)
        fp.backward()
        gp = [x.grad for x in p]  # 在p中獲取每個value的梯度
        glen = norm(gp)  # norm = 梯度的長度 (步伐大小)
        if i % dump_period == 0:
            print('{:05d}:f(p)={:.3f} p={} gp={} glen={:.5f}'.format(
                i, fp.data, [x.data for x in p], [x.grad for x in p], glen))
        if glen < 0.00001:  # 如果步伐已經很小了，那麼就停止吧！
            break
        gh = np.multiply(gp, -1*h)  # gh = 逆梯度方向的一小步
        p = [x + gh[i] for i, x in enumerate(p)]
    print('{:05d}:f(p)={:.3f} p={} gp={} glen={:.5f}'.format(
        i, fp.data, [x.data for x in p], [x.grad for x in p], glen))
    return p  # 傳回最低點！


def f(p):
    [x, y, z] = p
    return (x-1)**2+(y-2)**2+(z-3)**2


p = [Value(0), Value(0), Value(0)]
gradientDescendent(f, p)
