import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-headed attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        # q, k, v in linear projection's params
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        # output linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, q, k, v, mask, do_proj=True):
        """Calculate multiheaded attention
        Work for both self-/enc-dec attention
        Always requires mask:
        - self-att: src_len x src_len
        - enc-dec att: src_len x trg_len

        Args:
            q : bsz x src_len x embed_dim
            k : bsz x trg_len x embed_dim
            v : bsz x trg_len x embed_dim
            mask : src_len x trg_len
        """
        if do_proj:
            q, k, v = self.linear_projection(q, k, v)

        q, k, v = self.split_heads(q, k, v)
        output, att_weights = self.scaled_dot_attention(q, k, v, mask)
        output = self.concat_heads(output)
        return self.out_proj(output), att_weights

    def linear_projection(self, q, k, v):
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr()
        kv_same = k.data_ptr() == v.data_ptr()

        if qkv_same:
            q, k, v = self._in_proj(q).chunk(3, dim=-1)
        elif kv_same:
            q = self._in_proj(q, end=self.embed_dim)
            k, v = self._in_proj(k, start=self.embed_dim).chunk(2, dim=-1)
        else:
            q = self.in_proj_q(q)
            k = self.in_proj_k(k)
            v = self.in_proj_v(v)

        return q, k, v

    def split_heads(self, q, k, v):
        def _split_and_transpose(tensor):
            bsz, length, embed_dim = tensor.size()
            return tensor.reshape(bsz, length, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads, -1, self.head_dim)

        q = _split_and_transpose(q)
        k = _split_and_transpose(k)
        v = _split_and_transpose(v)
        return q, k, v

    def scaled_dot_attention(self, q, k, v, mask):
        att_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        bsz_x_num_heads, src_len, trg_len = att_weights.size()
        att_weights = att_weights.reshape(bsz_x_num_heads // self.num_heads, self.num_heads, src_len, trg_len)

        if mask is not None:
            att_weights.masked_fill_(mask, float('-inf'))

        att_weights = att_weights.reshape(bsz_x_num_heads, src_len, trg_len)
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = F.dropout(att_weights, p=self.dropout, training=self.training)

        return torch.bmm(att_weights, v), att_weights

    def concat_heads(self, output):
        bsz_x_num_heads = output.size()[0]
        return output.reshape(bsz_x_num_heads // self.num_heads, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz_x_num_heads // self.num_heads, -1, self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight[start:end, :]
        bias = self.in_proj_bias[start:end]
        return F.linear(input, weight, bias)

    def in_proj_q(self, q):
        return self._in_proj(q, end=self.embed_dim)

    def in_proj_k(self, k):
        return self._in_proj(k, start=self.embed_dim, end=self.embed_dim * 2)

    def in_proj_v(self, v):
        return self._in_proj(v, start=self.embed_dim * 2)


class PositionWiseFeedForward(nn.Module):
    """PositionWiseFeedForward"""
    def __init__(self, embed_dim, ff_dim, dropout=0.):
        super(PositionWiseFeedForward, self).__init__()
        self.in_proj = nn.Linear(embed_dim, ff_dim, bias=True)
        self.out_proj = nn.Linear(ff_dim, embed_dim, bias=True)
        self.dropout = dropout

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.dropout, training=self.training)
        return self.out_proj(y)
