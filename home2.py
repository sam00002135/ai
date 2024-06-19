import random

citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]

l = len(citys)
path = [(i+1) % l for i in range(l)]


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5


def pathLength(p):
    dist = 0
    plen = len(p)
    for i in range(plen):
        dist += distance(citys[p[i]], citys[p[(i+1) % plen]])
    return dist


def neighbor(x):
    idx1, idx2 = random.sample(range(len(x)), 2)
    x[idx1], x[idx2] = x[idx2], x[idx1]
    return x


def hillClimbing(f, x, h=0.01, max=1000):
    for _ in range(max):
        f_x = f(x)
        new_x = neighbor(x)
        f_new_x = f(new_x)
        if f_new_x < f_x:
            x = new_x
    return x, f_x


final_path, path_length = hillClimbing(pathLength, path)
print('Final path:', final_path)
print('Path length:', path_length)
