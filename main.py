import math
import creat
import numpy
import matplotlib.pyplot as plt

# 求点的磁场
# p = [1, -0.5, 0]
# result_point = creat.point_B(1, 1, p)

# # 求线的磁场
# p = [0, 0, 1]
# v = [1, 1, 1]
# coil = [10, 0.001]
# result_line = creat.coil_line(1, 1, p, v, coil=coil, pic=1)
# print(result_line)


# # 求面的磁场
factor = [0, 0, 1, 2]
coil = [10, 0.001]
repoint = [0, 0, 0]
reangle = [0, 0, 0]
# # 等价于以下的旋转
# factor = [0, 0, 1, 2]
# coil = [10, 0.001]
# repoint = [0, 0, 0]
# reangle = [0, math.pi / 4, 0]
result_face = creat.coil_face(1, 1, factor, repoint=repoint, reangle=reangle, density=[0.5, 0.5], coil=coil)

# 画线/点图
# fig = plt.figure()
# ax = fig.add_subplot(311, projection='3d')
# ax.scatter(result_face[:, 0], result_face[:, 1], result_face[:, 2])
# # ax.set(xticklabels=['x'], yticklabels=['y'], zticklabels=['z'])
# plt.show()

# 画三维平面色图

# pos = [1, 2, 3]
# coil = [10, 0.001]
# result = creat.coil_point(1, 1, pos, coil=coil)

# print(result)
