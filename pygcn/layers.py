import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        args:
            in_features: number of input features
            out_features: number of output features
            bias: boolean, whether to add bias

        attributes:
            weight: torch.Tensor, shape (out_features, in_features)
            bias: torch.Tensor, shape (out_features, )
        """
        super(GraphConvolution, self).__init__()
        # 成员变量初始化
        self.in_features = in_features
        self.out_features = out_features
        # Parameter将一个不可训练的类型Tensor转换为可训练的类型Tensor，使得其参数可以被更新
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 参数随机化
        self.reset_parameters()

    def reset_parameters(self):
        """参数随机化函数"""
        # size()返回out_features的大小
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 此处data为Tensor类型
        # uniform函数将参数初始化为（-stdv，stdv）区间的均匀分布
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """前向传播函数"""
        # 矩阵相乘
        support = torch.mm(input, self.weight)
        # spmm为稀疏矩阵相乘,减少了计算量
        output = torch.spmm(adj, support)
        # 为结果添加偏置量
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """类的字符串输出表示"""
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
