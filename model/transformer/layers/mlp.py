# This source code is licensed under the license found in the References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional
import math
from torch import Tensor, nn


# class Mlp(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         hidden_features: Optional[int] = None,
#         out_features: Optional[int] = None,
#         act_layer: Callable[..., nn.Module] = nn.GELU,
#         drop: float = 0.0,
#         bias: bool = True,
#     ) -> None:
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class Mlp(nn.Module):
    def __init__(
            self,
            layer_index: int,
            dep: int,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable[..., nn.Module] = nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        self.layer_index = layer_index
        self.dep = dep
        self.r = 64

        if self.layer_index < dep - 1:
            self.w_a_linear_fc1 = nn.Linear(in_features, self.r, bias=False)
            self.w_b_linear_fc1 = nn.Linear(self.r, hidden_features, bias=False)
            self.w_a_linear_fc2 = nn.Linear(hidden_features, self.r, bias=False)
            self.w_b_linear_fc2 = nn.Linear(self.r, out_features, bias=False)

            nn.init.kaiming_uniform_(self.w_a_linear_fc1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_fc2.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_fc1.weight)
            nn.init.zeros_(self.w_b_linear_fc2.weight)
        else:
            self.w_a_linear_fc1_sketch = nn.Linear(in_features, self.r, bias=False)
            self.w_b_linear_fc1_sketch = nn.Linear(self.r, hidden_features, bias=False)
            self.w_a_linear_fc2_sketch = nn.Linear(hidden_features, self.r, bias=False)
            self.w_b_linear_fc2_sketch = nn.Linear(self.r, out_features, bias=False)

            nn.init.kaiming_uniform_(self.w_a_linear_fc1_sketch.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_fc2_sketch.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_fc1_sketch.weight)
            nn.init.zeros_(self.w_b_linear_fc2_sketch.weight)

            self.w_a_linear_fc1_photo = nn.Linear(in_features, self.r, bias=False)
            self.w_b_linear_fc1_photo = nn.Linear(self.r, hidden_features, bias=False)
            self.w_a_linear_fc2_photo = nn.Linear(hidden_features, self.r, bias=False)
            self.w_b_linear_fc2_photo = nn.Linear(self.r, out_features, bias=False)

            nn.init.kaiming_uniform_(self.w_a_linear_fc1_photo.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_a_linear_fc2_photo.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_fc1_photo.weight)
            nn.init.zeros_(self.w_b_linear_fc2_photo.weight)

    def forward(self, x: Tensor, datatype) -> Tensor:
        x_pretrained_fc1 = self.fc1(x)

        if self.layer_index < self.dep - 1:
            x1 = x_pretrained_fc1 + self.w_b_linear_fc1(self.w_a_linear_fc1(x))
        elif datatype == 'sketch':
            x1 = x_pretrained_fc1 + self.w_b_linear_fc1_sketch(self.w_a_linear_fc1_sketch(x))
        else:
            x1 = x_pretrained_fc1 + self.w_b_linear_fc1_photo(self.w_a_linear_fc1_photo(x))

        x1 = self.act(x1)
        x1 = self.drop(x1)

        x_pretrained_fc2 = self.fc2(x1)

        if self.layer_index < self.dep - 1:
            x2 = x_pretrained_fc2 + self.w_b_linear_fc2(self.w_a_linear_fc2(x1))
        elif datatype == 'sketch':
            x2 = x_pretrained_fc2 + self.w_b_linear_fc2_sketch(self.w_a_linear_fc2_sketch(x1))
        else:
            x2 = x_pretrained_fc2 + self.w_b_linear_fc2_photo(self.w_a_linear_fc2_photo(x1))
        x2 = self.drop(x2)
        return x2

