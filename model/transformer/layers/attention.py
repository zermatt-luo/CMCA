
# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from torch import Tensor
from torch import nn

logger = logging.getLogger("dinov2")

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False

#
# class Attention(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int = 8,
#             qkv_bias: bool = False,
#             proj_bias: bool = True,
#             attn_drop: float = 0.0,
#             proj_drop: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#
# def forward(self, x: Tensor) -> Tensor:
#     B, N, C = x.shape  # B: 1, N: 257, C: 768
#     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # torch.Size([3, 1, 12, 257, 64])
#     q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
#     attn = q @ k.transpose(-2, -1)
#
#     attn = attn.softmax(dim=-1)
#     attn = self.attn_drop(attn)
#
#     x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#     x = self.proj(x)
#     x = self.proj_drop(x)
#     return x

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            layer_index: int,
            dep: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layer_index = layer_index
        self.dep = dep

        self.r = 64
        self.dim = dim

        if self.layer_index < dep - 1:
            self.w_a_linear_q = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_q = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_k = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_k = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_v = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_v = nn.Linear(self.r, self.dim, bias=False)

            self.w_a_linear_proj = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_proj = nn.Linear(self.r, self.dim, bias=False)

            nn.init.kaiming_uniform_(self.w_a_linear_q.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_k.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_v.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_q.weight)
            nn.init.zeros_(self.w_b_linear_k.weight)
            nn.init.zeros_(self.w_b_linear_v.weight)

            nn.init.kaiming_uniform_(self.w_a_linear_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_proj.weight)
        else:  # 12
            self.w_a_linear_q_sketch = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_q_sketch = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_k_sketch = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_k_sketch = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_v_sketch = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_v_sketch = nn.Linear(self.r, self.dim, bias=False)

            self.w_a_linear_q_photo = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_q_photo = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_k_photo = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_k_photo = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_v_photo = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_v_photo = nn.Linear(self.r, self.dim, bias=False)

            self.w_a_linear_proj_sketch = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_proj_sketch = nn.Linear(self.r, self.dim, bias=False)
            self.w_a_linear_proj_photo = nn.Linear(self.dim, self.r, bias=False)
            self.w_b_linear_proj_photo = nn.Linear(self.r, self.dim, bias=False)

            nn.init.kaiming_uniform_(self.w_a_linear_q_sketch.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_k_sketch.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_v_sketch.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_q_photo.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_k_photo.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_v_photo.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_q_sketch.weight)
            nn.init.zeros_(self.w_b_linear_k_sketch.weight)
            nn.init.zeros_(self.w_b_linear_v_sketch.weight)
            nn.init.zeros_(self.w_b_linear_q_photo.weight)
            nn.init.zeros_(self.w_b_linear_k_photo.weight)
            nn.init.zeros_(self.w_b_linear_v_photo.weight)

            nn.init.kaiming_uniform_(self.w_a_linear_proj_sketch.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_proj_photo.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_proj_sketch.weight)
            nn.init.zeros_(self.w_b_linear_proj_photo.weight)

    def forward(self, x: Tensor, data_type) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)

        if self.layer_index < self.dep - 1:
            new_q = self.w_b_linear_q(self.w_a_linear_q(x))
            new_k = self.w_b_linear_k(self.w_a_linear_k(x))
            new_v = self.w_b_linear_v(self.w_a_linear_v(x))
        elif data_type == 'sketch':
            new_q = self.w_b_linear_q_sketch(self.w_a_linear_q_sketch(x))
            new_k = self.w_b_linear_k_sketch(self.w_a_linear_k_sketch(x))
            new_v = self.w_b_linear_v_sketch(self.w_a_linear_v_sketch(x))
        else:
            new_q = self.w_b_linear_q_photo(self.w_a_linear_q_photo(x))
            new_k = self.w_b_linear_k_photo(self.w_a_linear_k_photo(x))
            new_v = self.w_b_linear_v_photo(self.w_a_linear_v_photo(x))

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, self.dim:2 * self.dim] += new_k
        qkv[:, :, -self.dim:] += new_v

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x_pretrained = self.proj(x)

        if self.layer_index < self.dep - 1:
            lora = self.w_b_linear_proj(self.w_a_linear_proj(x))
        elif data_type == 'sketch':
            lora = self.w_b_linear_proj_sketch(self.w_a_linear_proj_sketch(x))
        else:  # photo
            lora = self.w_b_linear_proj_photo(self.w_a_linear_proj_photo(x))
        x = x_pretrained + lora

        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, data_type: str, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x, data_type)  # Attention please, our method don't use xFormers

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
