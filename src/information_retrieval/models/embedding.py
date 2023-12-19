import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import dgl
import dgl.nn.pytorch as dglnn
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

from src.config import hparams


# 字粒度编码
class SingleEmb(nn.Module):
    def __init__(self):
        super(SingleEmb, self).__init__()
        self.bert = AutoModel.from_pretrained(hparams.pretrained_model, output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.first_last_avg(output)
        return pooled, output.last_hidden_state

    def first_last_avg(self, output):
        hidden_states = output.hidden_states
        pooled = []
        for i in range(2):
            seq = hidden_states[-i]
            pooled += [torch.mean(seq, dim=1, keepdim=True)]
        pooled = torch.cat(pooled, dim=1)
        pooled = torch.sum(pooled, 1)
        return pooled


# 更新词节点特征
class RenewNodeFeat(nn.Module):
    def __init__(self):
        super(RenewNodeFeat, self).__init__()
        self.dim = hparams.input_dim
        self.head = hparams.head
        self.emb = SingleEmb()
        self.attention = ScaledDotProductAttention(d_model=self.dim, d_k=self.dim, d_v=self.dim, h=self.head)
        self.mix_p = nn.Parameter(torch.rand(1))

    def forward(self, input_ids, attention_mask, graph_num_nodes, mask, token_type_ids=None):
        # 每个节点都会生成对应的全句向量，按照mask掩码池化得到词向量表示
        if token_type_ids is not None:
            pooled, last_hidden_state = self.emb(input_ids=input_ids, attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids)
        else:
            pooled, last_hidden_state = self.emb(input_ids=input_ids, attention_mask=attention_mask)
        # 注意力机制
        att_output = self.attention(last_hidden_state, last_hidden_state, last_hidden_state)
        feat = att_output.repeat_interleave(graph_num_nodes, dim=0)
        # 混合池化
        masked_feat = mask.unsqueeze(-1) * feat  # 将无效值置零
        max_pooled, _ = torch.max(masked_feat, dim=1)
        non_zero_counts = mask.sum(dim=1, keepdim=True).float()  # 每个样本的非零项个数
        mean_pooled = masked_feat.sum(dim=1) / non_zero_counts
        mix_pooled = self.mix_p * max_pooled + (1 - self.mix_p) * mean_pooled
        return mix_pooled


# 异构图卷积提取句法信息
class HeteroGraph(nn.Module):
    def __init__(self, num_layers):
        super(HeteroGraph, self).__init__()
        self.dim = hparams.input_dim
        self.rel_names = ['loop', 'dep', '-dep']
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(self.dim, self.dim)
            for rel in self.rel_names}, aggregate='sum'))
        for _ in range(2, self.num_layers + 1):
            self.convs.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(self.dim, self.dim)
                for rel in self.rel_names}, aggregate='sum'))

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            conv = self.convs[i]
            feat = conv(graph, feat)
            feat = {k: F.relu(v) for k, v in feat.items()}
        pool_out = self.read_out(graph, feat['token'])
        return pool_out

    def read_out(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # 通过平均读出值来计算单图的表征
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
        return hg


class BinaryClass_Single(nn.Module):
    def __init__(self, dropout, norm_way, num_layers):
        super(BinaryClass_Single, self).__init__()
        self.dim = hparams.input_dim
        self.renew = RenewNodeFeat()
        self.gcn = HeteroGraph(num_layers)
        self.class_num = hparams.class_nums
        self.drop = nn.Dropout(dropout)
        if norm_way == "BatchNorm":
            self.norm = nn.BatchNorm1d
        elif norm_way == "LayerNorm":
            self.norm = nn.LayerNorm
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            self.norm(self.dim),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.dim, self.class_num)
        )

    def forward(self, input_ids1, attention_mask1, graph1, g1_num_nodes, mask1, input_ids2, attention_mask2, graph2,
                g2_num_nodes, mask2):
        # 先更新图与特征，再送图卷积层，最后读出向量拼接二分类
        word_vec1 = self.renew(input_ids1, attention_mask1, g1_num_nodes, mask1)
        word_vec2 = self.renew(input_ids2, attention_mask2, g2_num_nodes, mask2)
        feat1 = {'token': word_vec1}
        feat2 = {'token': word_vec2}
        # shape = batch_size*dim
        feat1 = self.gcn(graph1, feat1)
        feat2 = self.gcn(graph2, feat2)
        f1_f2 = torch.abs(feat1 - feat2)
        cat_feat = torch.cat((feat1, feat2, f1_f2), dim=-1)
        output = self.binary_classifier(cat_feat)
        return output

class BinaryClass_Double(nn.Module):
    def __init__(self, dropout, num_layers):
        super(BinaryClass_Double, self).__init__()
        self.dim = hparams.input_dim
        self.renew = RenewNodeFeat()
        self.gcn = HeteroGraph(num_layers)
        self.class_num = hparams.class_nums
        self.drop = nn.Dropout(dropout)
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            self.drop,
            nn.Linear(self.dim, self.class_num)
        )

    def forward(self, input_ids, token_type_ids, attention_mask, graph, g_num_nodes, mask):
        pooled, word_vec = self.renew(input_ids, attention_mask, g_num_nodes, mask, token_type_ids=token_type_ids)
        feat = {'token': word_vec}
        feat = self.gcn(graph, feat)
        cat_feat = torch.cat((pooled, feat), dim=-1)
        output = self.binary_classifier(cat_feat)
        return output