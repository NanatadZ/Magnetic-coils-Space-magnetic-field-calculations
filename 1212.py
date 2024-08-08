import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.linspace(-3, 3, 4), np.linspace(-3, 3, 4))
# 两个为0      ?1,?2=np.meshgrid(pos[:,?1], pos[:,?2])
# 全不为0或   X,Y=np.meshgrid(pos[:,0], pos[:,1])  Y,Z  X,Z
Z = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [7, 8, 9, 10]])
# Bx = np.array(np.reshape(...))
levels = np.linspace(Z.min(), Z.max(), 7)

# 进行绘图
# fig, ax = plt.subplots()
# cs = ax.contourf(X, Y, Z, cmap=plt.get_cmap('Spectral'), levels=levels)
# # 添加colorbar
# cbar = fig.colorbar(cs)
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(141)
# cs = ax.contourf(X, Y, Z, cmap=plt.get_cmap('Spectral'), levels=levels)
# ax.set_aspect(1)
# fig.colorbar(cs, shrink=0.5)
# ax = fig.add_subplot(142)
# cs = ax.contourf(X, Y, Z, cmap=plt.get_cmap('Spectral'), levels=levels)
# ax.set_aspect(1)
# fig.colorbar(cs, shrink=0.5)
# plt.show()


pos0 = np.array([0, 1, 2])
pos1 = np.array([2, 3, 4])
Pos0, Pos1 = np.meshgrid(pos0, pos1)
Z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
levelsZ = np.linspace(Z.min(), Z.max(), 7)
fig = plt.figure()
ax = fig.add_subplot(142)
cs = ax.contourf(Pos0, Pos1, Z, cmap=plt.get_cmap('Spectral'), levels=levelsZ)
ax.set_aspect(1)
fig.colorbar(cs, shrink=0.5)
plt.show()
