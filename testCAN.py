import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import scipy.io as io
from sklearn.preprocessing import StandardScaler

# 定义一个小的容差
zero_tol = 1e-6


# 计算两组向量之间的欧几里得距离矩阵
def calc_Dist(X):
    n = X.shape[1]
    X_p1 = np.sum(X ** 2, axis=0).reshape(n, 1)  # 每列平方和，n×1
    Dist = X_p1 + X_p1.T - 2 * (X.T @ X)  # 欧几里得距离公式展开
    Dist[Dist < 0] = 0  # 修正可能的数值误差
    return Dist


# 构造邻接矩阵 S
def construct_S(Dist, k):
    n = Dist.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        # 获取第 i 个点的 k 个最近邻
        knn_idx = np.argsort(Dist[i])[:k + 1]  # 包括自身
        knn_dist = Dist[i, knn_idx]
        knn_dist = knn_dist[1:]  # 去除自身距离
        knn_idx = knn_idx[1:]

        # 计算权重
        sigma = np.mean(knn_dist) + zero_tol  # 防止分母为 0
        weights = np.exp(-knn_dist / (2 * sigma ** 2))

        if np.sum(weights) > zero_tol:  # 防止分母为 0
            weights /= np.sum(weights)  # 归一化
            S[i, knn_idx] = weights
        else:
            S[i, knn_idx] = 0  # 如果分母为 0，赋值为 0
    return S


# 计算拉普拉斯矩阵
def calc_Laplacian(S):
    W = (S + S.T) / 2  # 对称化
    D = np.diag(np.sum(W, axis=1))
    return D - W


# 数据标准化
def do_z_score(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


# CAN算法实现
def CAN(X, c, k=10, max_iter=100):
    n = X.shape[1]
    Dist = calc_Dist(X)  # 计算距离矩阵
    S = construct_S(Dist, k)  # 初始化邻接矩阵

    for iter in range(max_iter):
        # 计算拉普拉斯矩阵并特征分解
        L = calc_Laplacian(S)
        if np.isnan(L).any() or np.isinf(L).any():
            raise ValueError("Laplacian matrix contains NaN or Inf")

        eigenvalues, F = eigh(L, subset_by_index=[0, c - 1])  # 提取前 c 个特征向量
        F = F.real

        # 更新邻接矩阵 S
        Dist_F = calc_Dist(F.T)
        S_new = construct_S(Dist_F, k)

        # 检查收敛
        if np.linalg.norm(S - S_new, ord='fro') < zero_tol:
            break
        S = S_new

    # 最终分解
    n_components, labels = connected_components(csgraph=csr_matrix(S), directed=False)
    return S, labels, n_components


# 测试部分
if __name__ == "__main__":
    # 定义数据文件名
    filename = '_cmupie_32x32.mat'

    # 加载 .mat 文件的数据
    dic = io.loadmat(filename)  # 使用 scipy.io 加载文件
    X = dic['X']  # 获取数据矩阵 X
    Y = dic['Y']  # 获取对应的类别标签 Y

    # 获取数据矩阵的形状信息
    n, d = X.shape  # n 为样本数量，d 为特征维度
    c = len(np.unique(Y))  # 计算类别数

    # 对数据矩阵 X 进行标准化并转置
    X = do_z_score(X).T  # 标准化：每列均值为 0，方差为 1

    # 执行 CAN（Clustering with Adaptive Neighbors）算法
    S, labels, n_components = CAN(X, c, k=5, max_iter=200)

    # 调试输出
    print("Shape of X:", X.shape)
    print("Unique labels in Y:", np.unique(Y))
    print("Sparsity of S:", np.mean(S == 0))

    # 输出结果
    print("n_components=" + str(n_components) + ", c=" + str(c))

    # 验证结果
    if n_components == c:
        print("测试通过：CAN 算法成功聚类")
    else:
        print("测试失败：n_components 和 c 不匹配，检查算法或参数")
