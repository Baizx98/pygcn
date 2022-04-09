import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    """
    将标签转换为onehot编码
    用N为寄存器来对N个状态进行编码
    """
    # 使用set()函数去重
    classes = set(labels)
    # {fun(x):fun(y) for x,y in iterable} 利用for in遍历迭代器生成字典
    # enumrate()函数返回一个迭代器，每次迭代返回一个元组，第一个元素为索引，第二个元素为值
    # np.identity(len(classes))返回一个N*N的单位矩阵
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # 将labels中的每个元素转换为对应的onehot编码
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # np.genformtxt()从csv中生成数组
    # "{}{}.content".format(path, dataset) 生成数据集文件名
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 提取全部行和第二列到倒数第二列并转换为压缩稀疏行矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的类别标签并将其转换为onehot编码
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 提取样本的编号到列表中,[[],[],……[]] 实际上是n行1列的二维数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 构建{样本编号:样本索引}的字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 样本中的引用关系数组,也即图的边
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 用flatten()函数将二维数组展开成一维数组
    # 使用map()函数将edges_unordered中的每一个元素转换为idx_map中对应的索引
    # reshape()函数将其还原为二维数组
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 引用论文的索引作列,被引用论文的索引作行,在这个矩阵里填充1,其余为0
    # 实际上是在构建图的邻接矩阵,用三元组表示为稀疏矩阵,非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 将非对称矩阵中的非0元素填充到对称位置,得到对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 样本特征的压缩稀疏矩阵,行规范稀疏矩阵
    features = normalize(features)
    # 添加了自连接的邻接矩阵,并进行了归一化
    # adj=D^-1(A+I)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 分割训练集,验证集和测试集
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 将特征矩阵转换为Tensor
    features = torch.FloatTensor(np.array(features.todense()))
    # 将onehot编码转换为常规标签0,1,2,3,4...
    labels = torch.LongTensor(np.where(labels)[1])
    # 将scipy稀疏矩阵转换为tensor稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    # 转换为张量
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回 样本对称邻接矩阵的稀疏张量,样本特征张量,标签张量,训练集,验证集,测试集张量
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """
    Row-normalize sparse matrix
    计算D^-1A
    将矩阵每行进行归一化
    """
    # 将二维矩阵的每一行求和
    rowsum = np.array(mx.sum(1))
    # 对每行求和的值再求倒数并压缩为一维列表
    r_inv = np.power(rowsum, -1).flatten()
    # 将列表中无穷大的值置为零
    r_inv[np.isinf(r_inv)] = 0.
    # 根据r_inv的值创建对角矩阵
    r_mat_inv = sp.diags(r_inv)
    # 两个矩阵点乘
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    """计算准确率"""
    # .max(1)返回每一行最大值组成的一维数组和索引
    # .max(1)[1]返回索引
    # type_as()将张量转换为labels类型
    preds = output.max(1)[1].type_as(labels)
    # 判断labels和preds是否相等,相等置1,否则置0
    correct = preds.eq(labels).double()
    # 统计相等的数量
    correct = correct.sum()
    # 除以总数,得到准确率
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # tocoo()将矩阵转换为coo格式
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # vstack()将两个数组按垂直方向堆叠成一个新数组
    # coo的索引
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # coo的值
    values = torch.from_numpy(sparse_mx.data)
    # coo的形状大小
    shape = torch.Size(sparse_mx.shape)
    # 构造稀疏张量
    return torch.sparse.FloatTensor(indices, values, shape)
