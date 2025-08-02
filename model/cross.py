

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from .TS_Pos_Enc import *

# 傅里叶变换层
class FourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [bs x seq_len x d_model]
        # 傅里叶变换
        x_fft = torch.fft.fft(x, dim=1)

        # 返回实部和虚部
        return torch.real(x_fft), torch.imag(x_fft)

# 逆傅里叶变换层
class InverseFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_real, x_imag):
        # 合并实部和虚部
        x_complex = torch.complex(x_real, x_imag)

        # 逆傅里叶变换
        x_ifft = torch.fft.ifft(x_complex, dim=1)

        # 返回实部
        return torch.real(x_ifft)

# 修改后的 TSTEncoderLayer
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False,
                 fourier_ratio=0.5):  # 控制傅里叶分支的比例
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.LayerNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.LayerNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

        # 傅里叶变换相关层
        self.fourier_transform = FourierTransform()
        self.inverse_fourier_transform = InverseFourierTransform()
        self.fourier_linear = nn.Linear(d_model, d_model)  # 频率域特征提取的一个例子
        self.dropout_fourier = nn.Dropout(dropout)
        self.fourier_ratio = fourier_ratio

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        ## Multi-Head attention
        if self.res_attention:
            # print(q.shape,k.shape,v.shape)
            q2, attn, scores = self.self_attn(q, k, v, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        q = q + self.dropout_attn(q2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_attn(q)

        # 傅里叶变换分支
        q_real, q_imag = self.fourier_transform(q)
        q_fourier = self.fourier_linear(q_real)  # 频率域特征提取
        q_fourier = self.dropout_fourier(q_fourier)
        q_fourier_real = q_fourier
        q_fourier_imag = torch.zeros_like(q_fourier)  # 简单地将虚部置零

        q_fourier_output = self.inverse_fourier_transform(q_fourier_real, q_fourier_imag)

        # Feed-forward sublayer
        if self.pre_norm:
            q = self.norm_ffn(q)
        ## Position-wise Feed-Forward
        q2 = self.ff(q)
        ## Add & Norm
        q = q + self.dropout_ffn(q2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_ffn(q)

        # 融合
        q = (1 - self.fourier_ratio) * q + self.fourier_ratio * q_fourier_output

        if self.res_attention:
            return q, scores
        else:
            return q

# Cell
class CrossModal(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='LayerNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False, fourier_ratio=0.5): # 添加 fourier_ratio
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer( d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn,
                                                      fourier_ratio=fourier_ratio) for i in range(n_layers)])  # 传递 fourier_ratio
        self.res_attention = res_attention

    def forward(self, q:Tensor,k:Tensor,v:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        q  [bs * nvars x (text_num) x d_model]
        k  [bs * nvars x (text_num) x d_model]
        v  [bs * nvars x (text_num) x d_model]
        '''
        scores = None
        #print(q.shape,k.shape,v.shape)
        if self.res_attention:
            for mod in self.layers: output, scores = mod(q,k,v,  prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(q,k,v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa


    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

def get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()  # 返回 ReLU 模块的实例
    elif activation == "gelu":
        return nn.GELU()  # 返回 GELU 模块的实例
    else:
        raise ValueError("unknown activation function!")

class Transpose(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.transpose(*self.args)