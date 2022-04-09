import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        args:
            nfeat: number of input features
            nhid: number of hidden units
            nclass: number of classes
            dropout: dropout rate

        GCN由两个GraphConvolution层组成
        输出为输出层做log_softmax变换的结果
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # x是输入特征,adj是邻接矩阵
        # self.gc1(x,adj)是执行GraphConvolution层的前向传播函数
        # 结果为(随机初始化参数*输入特征*邻接矩阵+偏置量)
        # 经过激活函数
        x = F.relu(self.gc1(x, adj))
        # dropout层,随机把一些参数置为零
        x = F.dropout(x, self.dropout, training=self.training)
        # gc2层
        x = self.gc2(x, adj)
        # 输出层经过log_softmax变换的结果
        return F.log_softmax(x, dim=1)
