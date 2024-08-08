import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rect2sphere(x, y, z):
    """
    输入直角坐标系位置，输出球坐标系位置
    :param x: 直角坐标系下x坐标 列表
    :param y: 直角坐标系下y坐标 列表
    :param z: 直角坐标系下z坐标 列表
    :return: 球坐标系下的 r矢径距离 , phi天顶角 弧度制,theta子午面转角 弧度制
    """
    if (x == 0) and (y == 0):
        r = math.fabs(z)
        phi = 0.0
        theta = 0.0
        result = numpy.array([r, phi, theta])
        return result
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    phi = math.acos(z / r)
    if y > 0:
        theta = math.acos(x / (r * math.sin(phi)))
    elif y == 0:
        theta = 0
    else:
        theta = 2 * math.pi - math.acos(x / (r * math.sin(phi)))
    result = numpy.array([r, phi, theta])
    return result


def line(r0, pos0, vector, density=0.1, wide=5):
    space = r0 * wide
    ev = vector / math.sqrt(vector[0][0] ** 2 + vector[0][1] ** 2 + vector[0][2] ** 2)
    n = (math.sqrt(pos0[0][0] ** 2 + pos0[0][1] ** 2 + pos0[0][2] ** 2) + space) / density + 1
    pos1 = pos0 + numpy.transpose(numpy.array([numpy.arange(1, n)])) * ev
    pos2 = pos0 + numpy.transpose(numpy.array([numpy.arange(-n, -1)])) * ev
    pos = numpy.concatenate((pos1, pos0, pos2), axis=0)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(431, projection='3d')
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    plt.show()
    return


def point_B(r0, I, pos, step=100, miu=4 * math.pi * 1e-7):
    """
    计算点的磁场强度
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param pos: 直角坐标系下坐标列表[x,y,z]
    :param step: 积分步数，未设则默认100步
    :param miu: 环境的真空磁导率，未设则默认真空

    :return: 点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（1,6） 单位T
    """
    pos = numpy.array([pos])
    result = measure(r0, I, pos, step=step, miu=miu)
    print('over')
    return result


def line_B(r0, I, pos0, vector, density=0.1, wide=5, step=100, miu=4 * math.pi * 1e-7):
    """
    计算直线上点的磁场强度，直线采用(x-x0)/a=(y-y0)/b=(z-z0)/c
    其中x0,y0,z0为经过的点；a,b,c为直线方向向量
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param pos0: 直线经过的点列表[x0,y0,z0]
    :param vector:直线的方向向量列表[a,b,c]
    :param density:所选点的密度(两点之间的距离)，默认0.1m
    :param wide:直线计算范围系数w，即以磁偶心为球心，w个磁偶极半径的球范围。若不设定则默认为5
    :param step: 积分步数，未设则默认100步
    :param miu: 环境的真空磁导率，未设则默认真空

    :return:点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（n,6） 单位T
    """
    pos0 = numpy.array([pos0])
    vector = numpy.array([vector])
    space = r0 * wide
    ev = vector / math.sqrt(vector[0][0] ** 2 + vector[0][1] ** 2 + vector[0][2] ** 2)
    n = (math.sqrt(pos0[0][0] ** 2 + pos0[0][1] ** 2 + pos0[0][2] ** 2) + space) / density + 1
    pos = pos0 + numpy.transpose(numpy.array([numpy.arange(-n + 1, n)])) * ev * density
    result = measure(r0, I, pos, step=step, miu=miu)
    print('over')
    return result


def face_B(r0, I, factor, density=[0.1, 0.1], wide=5, step=100, miu=4 * math.pi * 1e-7, pic=1):
    """
    计算平面上Ax+By+Cz+D=0点的磁场强度，先求一个点，再求一条线，再平移线得到面
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param factor:平面的系数列表[A,B,C,D]
    :param density:所选点的密度(两点之间的距离)，默认[0.1, 0.1]m 第一个系数为平行于YZ平面线的密度
    :param wide:面计算范围系数w，即以磁偶心为球心，w个磁偶极半径的球范围。若不设定则默认为5
    :param step: 积分步数，未设则默认100步
    :param miu: 环境的真空磁导率，未设则默认真空
    :param pic: 绘图选项，默认1为绘图，输入0为不绘图

    :return:点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（(2n-1)*(2m-1),6） 单位T
    """
    (pos0, vector0, vector) = face_factor(factor)  # 点[x0,y0,z0],面上线向量[a,b,c],面上另一向量[a1,b1,c1]
    origin_point = pos0
    space = r0 * wide
    ev0 = vector0 / math.sqrt(vector0[0][0] ** 2 + vector0[0][1] ** 2 + vector0[0][2] ** 2)
    ev = vector / math.sqrt(vector[0][0] ** 2 + vector[0][1] ** 2 + vector[0][2] ** 2)
    n = int((math.sqrt(pos0[0][0] ** 2 + pos0[0][1] ** 2 + pos0[0][2] ** 2) + space) / density[0] + 1)
    m = int((math.sqrt(pos0[0][0] ** 2 + pos0[0][1] ** 2 + pos0[0][2] ** 2) + space) / density[1] + 1)
    pos0 = pos0 + numpy.transpose(numpy.array([numpy.arange(-m + 1, m)])) * ev * density[1]
    pos = numpy.array([numpy.arange(0, 3)])  # [a,b,c]与[a1,b1,c1]确认的面，贮存
    for i in range(2 * m - 1):
        pos1 = pos0[i, :] + numpy.transpose(numpy.array([numpy.arange(-n + 1, n)])) * ev0 * density[0]
        pos = numpy.concatenate((pos, pos1), axis=0)
    pos = numpy.delete(pos, 0, axis=0)
    result = measure(r0, I, pos, step=step, miu=miu)

    plotBxyz(origin_point, factor, result, m, n, density)
    print('over')
    return result


print(rect2sphere(1, -0.5, 0))
