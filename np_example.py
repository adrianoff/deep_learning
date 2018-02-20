import numpy as np
#
# a = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 2, 1]])
# print a
#
# print (a.reshape(12, 1))



n = 3
m = 2

x = np.random.rand(m, n)
w = np.random.rand(1, m)
print w.dot(x).T