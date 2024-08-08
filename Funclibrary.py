import math
import numpy
import matplotlib.pyplot as plt

"""
此文件为计算磁偶极子在空间中磁场的分布，默认在磁偶极子坐标系中，Z轴为磁偶极子极轴，且电流从Z轴从上往下看为逆时针
"""

"————————————————以下为基础函数————————————————"


def rect2sphere(xyz):
    """
    输入直角坐标系位置，输出球坐标系位置
    :param xyz: 格式为numpy的n*3的矩阵[x,y,z],n为输入点的数量 单位m
    :return: [x,y,z  r,phi,theta]，shape=(n,6),n为输入点的数量 单位m,rad
    """
    ptsnew = numpy.hstack((xyz, numpy.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = numpy.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 4] = numpy.arctan2(numpy.sqrt(xy), xyz[:, 2])  # 用于从 Z 轴向下定义的天顶角
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy))        # 用于从 XY 平面向上定义的仰角
    ptsnew[:, 5] = numpy.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def angle2matrix(reangle):
    """
    输入两坐标系三个相对角度输出XYZ旋转的旋转矩阵
    :param reangle: 相对角度[dangle-x,dangle-y,dangle-z]，单位rad
    :return: XYZ旋转的旋转矩阵
    """
    a = reangle[0]
    b = reangle[1]
    c = reangle[2]
    X = numpy.array([[1, 0, 0], [0, numpy.cos(a), numpy.sin(a)], [0, - numpy.sin(a), numpy.cos(a)]])
    Y = numpy.array([[numpy.cos(b), 0, - numpy.sin(b)], [0, 1, 0], [numpy.sin(a), 0, numpy.cos(a)]])
    Z = numpy.array([[numpy.cos(c), numpy.sin(c), 0], [-numpy.sin(c), numpy.cos(c), 0], [0, 0, 1]])
    XY = numpy.dot(X, Y)
    XYZ = numpy.dot(XY, Z)
    return XYZ


def measure(r0, I, pos, step=100, miu=4 * math.pi * 1e-7):
    """
    计算输入点的磁场强度
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param pos: 直角坐标系下坐标，格式为numpy的n*3的矩阵[x,y,z],n为输入点的数量 单位m
    :param step: 积分步数，未设则默认100步
    :param miu: 环境的真空磁导率，未设则默认真空

    :return:点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（n,6）,n为输入点的数量  单位T
    """
    posr = rect2sphere(pos)  # [x,y,z  r,phi,theta]，shape=(n,6),n为输入点的数量
    l = 2 * math.pi / step
    r = numpy.array([posr[:, 3]])  # 矢径距离r矩阵，shape=（1,n)
    phi = numpy.array([posr[:, 4]])  # 天顶角phi矩阵，shape=（1,n)
    theta = numpy.array([posr[:, 5]])  # 子午面转角theta矩阵，shape=（1,n)
    A = -(2 * r0 * r * numpy.sin(phi)) / (r0 ** 2 + r ** 2)
    b = numpy.zeros((3, r.shape[1]))  # 创造贮存矩阵b=[bx;by;bz]，shape=(3,n)
    for i in range(step):
        dtheta = l * i
        temp = (1 + A * numpy.cos(dtheta - theta)) ** 1.5
        dbx = (numpy.cos(dtheta) * l / temp)
        dby = (numpy.sin(dtheta) * l / temp)
        dbz = ((r0 - r * numpy.sin(phi) * numpy.cos(dtheta - theta)) * l / temp)
        db = numpy.concatenate((dbx, dby, dbz), axis=0)  # 每步积分得到的值，shape=（3,n）,n为输入点的数量
        b = b + db  # 求和
    bx = (miu * I * r0 * r * numpy.cos(phi)) * b[0, :] / (4 * math.pi * (r0 ** 2 + r ** 2) ** 1.5)
    by = (miu * I * r0 * r * numpy.cos(phi)) * b[1, :] / (4 * math.pi * (r0 ** 2 + r ** 2) ** 1.5)
    bz = (miu * I * r0) * b[2, :] / (4 * math.pi * (r0 ** 2 + r ** 2) ** 1.5)
    result = numpy.transpose(numpy.concatenate((bx, by, bz), axis=0))  # 点的磁场三分量[bx,by,bz]，shape=（n,3）,n为输入点的数量
    result = numpy.concatenate((pos, result), axis=1)  # 点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（n,6）,n为输入点的数量
    return result


def face_factor(ABCD):
    """
    对平面Ax+By+Cz+D=0，输出：
    平面上点[x0,y0,z0]，关系为Ax0+By0+Cz0+D=0
    平面上的线向量为[a,b,c]，垂直关系为Aa+Bb+Cc=0
    平面上令一线向量[a1,b1,c1]，与[a,b,c]垂直
    分三种输入:
    1、A=B=C=0 为无解条件
    2、ABC中有两个值为0
    3、ABC中有一个值为0
    4、ABC全不为0，有点[x0,y0,z0]取[0,0,-D/C]、线向量[a,b,c]取[1,1,-(A+B)/C]
    :param ABCD: 平面的系数列表[A,B,C,D]
    :return: pos0[x0,y0,z0] 1*3 , vector0[a,b,c] 1*3 , vector[a1,b1,c1] 1*3
    """
    factor = numpy.array([ABCD])
    if numpy.sum(factor[0, 0:3] == 0) == 3:  # 1、A=B=C=0 为无解条件
        print("错误：输入平面法向量系数不能全为0")
        return
    if numpy.sum(factor[0, 0:3] == 0) == 2:  # 2、ABC中有两个值为0
        luck = numpy.flatnonzero(factor[0, 0:3])
        # 平面上一点[x0,y0,z0]
        pos0 = numpy.array([[0., 0., 0.]])
        pos0[0][luck] = -factor[0][3] / factor[0][luck]
        # 平面上一向量[a,b,c]
        luck = numpy.where(factor[0, 0:3] == 0)
        vector0 = numpy.array([[0., 0., 0.]])
        vector0[0][luck[0][0]] = 1
        # 平面上另一向量[a1,b1,c1]
        vector = numpy.cross(factor[0, 0:3], vector0)
        return pos0, vector0, vector
    if numpy.sum(factor[0, 0:3] == 0) == 1:  # 3、ABC中有一个值为0
        luck = numpy.flatnonzero(factor[0, 0:3])
        # 平面上一点[x0,y0,z0]
        pos0 = numpy.array([[0., 0., 0.]])
        pos0[0][luck[0]] = -factor[0][3] / factor[0][luck[0]]
        # 平面上一向量[a,b,c]
        vector0 = numpy.array([[0., 0., 0.]])
        vector0[0][luck[0]] = 1
        vector0[0][luck[1]] = -factor[0][luck[0]] / factor[0][luck[1]]
        # 平面上另一向量[a1,b1,c1]
        vector = numpy.cross(factor[0, 0:3], vector0)
        return pos0, vector0, vector
    if numpy.sum(factor[0, 0:3] == 0) == 0:  # 4、ABC全不为0
        # 平面上一点[x0,y0,z0]
        pos0 = numpy.array([[0, 0, 0]])
        pos0[0][2] = -factor[0][3] / factor[0][2]
        # 平面上一向量[a,b,c]
        vector0 = numpy.array([[0., 0., 0.]])
        vector0[0][1] = 1
        vector0[0][2] = -factor[0][1] / factor[0][2]
        # 平面上另一向量[a1,b1,c1]
        vector = numpy.cross(factor[0, 0:3], vector0)
        return pos0, vector0, vector


def plotfBxyz(origin_point, vector0, vector, result, all_origin_point, m, n, density):
    """
    绘制面上的磁场三分量图
    :param origin_point: 原点
    :param vector0: 平面横坐标单位矢量
    :param vector: 平面纵坐标单位矢量
    :param result: 磁场强度数据（经过坐标变化）
    :param all_origin_point: 原始位置点（不经过变化）
    :param m:
    :param n:
    :param density: 密度系数
    :return:
    """
    pos0 = numpy.transpose(numpy.array([numpy.arange(-m + 1, m)])) * density[1]  # 垂直
    pos1 = numpy.transpose(numpy.array([numpy.arange(-n + 1, n)])) * density[0]  # 平行
    Pos0, Pos1 = numpy.meshgrid(pos0, pos1)
    Bx = numpy.reshape(result[:, 3], Pos0.shape)
    By = numpy.reshape(result[:, 4], Pos0.shape)
    Bz = numpy.reshape(result[:, 5], Pos0.shape)
    levelsBx = numpy.linspace(Bx.min(), Bx.max(), 7)
    levelsBy = numpy.linspace(By.min(), By.max(), 7)
    levelsBz = numpy.linspace(Bz.min(), Bz.max(), 7)

    pv1 = vector0 * m / (math.sqrt((vector0[0][0]) ** 2 + (vector0[0][1]) ** 2 + (vector0[0][2]) ** 2) * 2)
    pv2 = vector * n / (math.sqrt((vector[0][0]) ** 2 + (vector[0][1]) ** 2 + (vector[0][2]) ** 2) * 2)

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    ax = fig.add_subplot(141, projection='3d')
    plt.plot(all_origin_point[:, 0], all_origin_point[:, 1], all_origin_point[:, 2])
    ax.quiver(origin_point[0][0], origin_point[0][1], origin_point[0][2], pv1[0][0], pv1[0][1], pv1[0][2],
              arrow_length_ratio=0.01, color=(1, 0, 0, 0.5))  # 红色
    ax.quiver(origin_point[0][0], origin_point[0][1], origin_point[0][2], pv2[0][0], pv2[0][1], pv2[0][2],
              arrow_length_ratio=0.01, color=(0, 0, 0, 0.5))  # 黑色
    temp = origin_point + pv1
    ax.text(temp[0][0], temp[0][1], temp[0][2], "横坐标", color='red')
    temp = origin_point + pv2
    ax.text(temp[0][0], temp[0][1], temp[0][2], "纵坐标", color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('平面在物体坐标系下示意图')

    ax = fig.add_subplot(142)
    cs = plt.contourf(Pos0, Pos1, Bx, cmap=plt.get_cmap('Spectral'), levels=levelsBx)
    ax.set_aspect(1)
    plt.colorbar(cs, shrink=0.5)
    # plt.xlabel('横坐标')
    # plt.ylabel('纵坐标')
    plt.title('物体坐标系下Bx [T]')

    ax = fig.add_subplot(143)
    cs = ax.contourf(Pos0, Pos1, By, cmap=plt.get_cmap('Spectral'), levels=levelsBy)
    ax.set_aspect(1)
    fig.colorbar(cs, shrink=0.5)
    # plt.xlabel('横坐标')
    # plt.ylabel('纵坐标')
    plt.title('物体坐标系下By [T]')

    ax = fig.add_subplot(144)
    cs = ax.contourf(Pos0, Pos1, Bz, cmap=plt.get_cmap('Spectral'), levels=levelsBz)
    ax.set_aspect(1)
    fig.colorbar(cs, shrink=0.5)
    # plt.xlabel('横坐标')
    # plt.ylabel('纵坐标')
    plt.title('物体坐标系下Bz [T]')
    plt.show()
    return


def plotlBxyz(all_point, density):
    """
    绘制线上的磁场三分量曲线
    :param all_point: 磁场强度数据（经过坐标系装换）
    :param density: 曲线密度
    :return:
    """
    n = numpy.shape(all_point)[0]
    site = (numpy.arange(0, n) - n / 2) * density

    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    fig.add_subplot(311)
    plt.plot(site, all_point[:, 3])
    plt.ylabel('惯性坐标系下x方向磁场B [T]')
    plt.title('线段上的磁场三分量')

    fig.add_subplot(312)
    plt.plot(site, all_point[:, 4])
    plt.ylabel('惯性坐标系下y方向磁场B [T]')

    fig.add_subplot(313)
    plt.plot(site, all_point[:, 5])
    plt.xlabel("线段上与pos0相对位置/m")
    plt.ylabel('惯性坐标系下z方向磁场B [T]')
    plt.show()
    return


"————————————————以下为多线圈计算函数————————————————"


def coil_point(r0, I, pos, step=100, repoint=[0, 0, 0], reangle=[0, 0, 0], coil=[1, 0.005], miu=4 * math.pi * 1e-7):
    """
    计算点的磁场强度，本质为单线圈的多次化
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param pos: 直角坐标系下坐标列表[x,y,z]
    :param step: 积分步数，未设则默认100步
    :param repoint:物体坐标系相对惯性坐标系中心点的相对位置[dx,dy,dz],默认为[0,0,0]
    :param reangle:物体坐标系相对惯性坐标系的欧拉角[dangle-x,dangle-y,dangle-z],默认为[0,0,0]
    :param miu: 环境的真空磁导率，未设则默认真空
    :param coil: 螺线圈系数[number, h] number为线圈数量，h为相邻线圈距离单位m，默认为[1, 0.005]

    :return: 点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（1,6） 单位T
    """
    pos = numpy.array([pos])
    # 以下为旋转操作
    XYZ = angle2matrix(reangle)
    pos = numpy.dot(pos, XYZ)
    # 以下为平移操作
    pos[:, 0] = pos[:, 0] + repoint[0]
    pos[:, 1] = pos[:, 1] + repoint[1]
    pos[:, 2] = pos[:, 2] + repoint[2]

    if coil[0] == 1:
        result = measure(r0, I, pos, step=step, miu=miu)
        print('over')
        return result
    else:
        n = coil[0]
        h = coil[1]
        result = measure(r0, I, pos, step=step, miu=miu)
        for i in range(n - 1):
            pos[0, 2] = pos[0, 2] + h
            temp_result = measure(r0, I, pos, step=step, miu=miu)
            result[0, 3:5] = result[0, 3:5] + temp_result[0, 3:5]
        print('over')
        return result


def coil_line(r0, I, pos0, vector, density=0.1, wide=5, step=100, repoint=[0, 0, 0], reangle=[0, 0, 0], coil=[1, 0.005],
              miu=4 * math.pi * 1e-7, pic=1):
    """
    计算直线上点的磁场强度，直线采用(x-x0)/a=(y-y0)/b=(z-z0)/c
    其中x0,y0,z0为经过的点；a,b,c为直线方向向量   本质为单线圈的多次化
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param pos0: 直线经过的点列表[x0,y0,z0]
    :param vector:直线的方向向量列表[a,b,c]
    :param density:所选点的密度(两点之间的距离)，默认0.1m
    :param wide:直线计算范围系数w，即以磁偶心为球心，w个磁偶极半径的球范围。若不设定则默认为5
    :param step: 积分步数，未设则默认100步
    :param repoint:物体坐标系相对惯性坐标系中心点的相对位置[dx,dy,dz],默认为[0,0,0]
    :param reangle:物体坐标系相对惯性坐标系的欧拉角[dangle-x,dangle-y,dangle-z],默认为[0,0,0]
    :param miu: 环境的真空磁导率，未设则默认真空
    :param coil: 螺线圈系数[number, h] number为线圈数量，h为相邻线圈距离单位m，默认为[1, 0.005]
    :param pic: 绘图选项，默认1为绘图，输入0为不绘图

    :return:点坐标与磁场三分量[x，y，z,bx,by,bz]，shape=（n,6） 单位T
    """
    pos0 = numpy.array([pos0])
    vector = numpy.array([vector])
    space = r0 * wide
    ev = vector / math.sqrt(vector[0][0] ** 2 + vector[0][1] ** 2 + vector[0][2] ** 2)
    n = (math.sqrt(pos0[0][0] ** 2 + pos0[0][1] ** 2 + pos0[0][2] ** 2) + space) / density + 1
    pos = pos0 + numpy.transpose(numpy.array([numpy.arange(-n + 1, n)])) * ev * density
    # 以下为旋转操作
    XYZ = angle2matrix(reangle)
    pos = numpy.dot(pos, XYZ)
    # 以下为平移操作
    pos[:, 0] = pos[:, 0] + repoint[0]
    pos[:, 1] = pos[:, 1] + repoint[1]
    pos[:, 2] = pos[:, 2] + repoint[2]
    if coil[0] == 1:
        result = measure(r0, I, pos, step=step, miu=miu)
        if pic == 1:
            plotlBxyz(result, density)
        print('over')
        return result
    else:
        n = coil[0]
        h = coil[1]
        result = measure(r0, I, pos, step=step, miu=miu)
        for i in range(n - 1):
            pos[:, 2] = pos[:, 2] + h
            temp_result = measure(r0, I, pos, step=step, miu=miu)
            result[:, 3:5] = result[:, 3:5] + temp_result[:, 3:5]
        if pic == 1:
            plotlBxyz(result, density)
        print('over')
        return result


def coil_face(r0, I, factor, density=[0.1, 0.1], wide=5, step=100, repoint=[0, 0, 0], reangle=[0, 0, 0],
              coil=[1, 0.005], miu=4 * math.pi * 1e-7, pic=1):
    """
    计算平面上Ax+By+Cz+D=0点的磁场强度，先求一个点，再求一条线，再平移线得到面  ，本质为单线圈的多次化
    :param r0: 磁偶极子半径 单位m
    :param I: 磁偶极子电流 单位A
    :param factor:平面的系数列表[A,B,C,D]
    :param density:所选点的密度(两点之间的距离)，默认[0.1, 0.1]m 第一个系数为平行于YZ平面线的密度
    :param wide:面计算范围系数w，即以磁偶心为球心，w个磁偶极半径的球范围。若不设定则默认为5
    :param step: 积分步数，未设则默认100步
    :param repoint:物体坐标系相对惯性坐标系中心点的相对位置[dx,dy,dz],默认为[0,0,0]
    :param reangle:物体坐标系相对惯性坐标系的欧拉角[dangle-x,dangle-y,dangle-z],默认为[0,0,0]
    :param miu: 环境的真空磁导率，未设则默认真空
    :param pic: 绘图选项，默认1为绘图，输入0为不绘图
    :param coil: 螺线圈系数[number, h] number为线圈数量，h为相邻线圈距离单位m，默认为[1, 0.005]

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
    # 以下为旋转操作
    XYZ = angle2matrix(reangle)
    all_origin_point = pos
    pos = numpy.dot(pos, XYZ)
    # 以下为平移操作
    pos[:, 0] = pos[:, 0] + repoint[0]
    pos[:, 1] = pos[:, 1] + repoint[1]
    pos[:, 2] = pos[:, 2] + repoint[2]

    if coil[0] == 1:
        result = measure(r0, I, pos, step=step, miu=miu)
        print('over')
        if pic == 1:
            plotfBxyz(origin_point, vector0, vector, result, all_origin_point, m, n, density)
        return result
    else:
        nt = coil[0]
        h = coil[1]
        result = measure(r0, I, pos, step=step, miu=miu)
        for i in range(nt - 1):
            pos[:, 2] = pos[:, 2] - h  # 线圈沿Z轴向上延伸
            temp_result = measure(r0, I, pos, step=step, miu=miu)
            result[:, 3:5] = result[:, 3:5] + temp_result[:, 3:5]
        print('over')
        if pic == 1:
            plotfBxyz(origin_point, vector0, vector, result, all_origin_point, m, n, density)
        return result
