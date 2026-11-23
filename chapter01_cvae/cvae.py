"""第 1 章手写实战：ACT 的 CVAE 风格编码器（Style Encoder）。

本模块实现 ACT 仅在【训练期】启用的 CVAE 编码支路：
    输入:
        - state:         [B, state_dim]               当前关节状态 (ALOHA 为 14 维)
        - actions:       [B, chunk_size, action_dim]  未来 k 步真实动作块 (ALOHA 为 [B, 100, 14])
        - action_is_pad: [B, chunk_size] (可选 bool)  True 表示该帧是 episode 末尾的填充帧
    输出:
        - z:       [B, latent_dim]  经重参数化采样的风格隐变量 (默认 32 维)
        - mu:      [B, latent_dim]  后验高斯分布均值
        - log_var: [B, latent_dim]  后验高斯分布对数方差 log(sigma^2)

请按顺序完成文件中的 4 处 TODO（先不要看答案，直接对着 docstring 和张量 shape 写）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_1d_sinusoidal_pos_embed(seq_len: int, d_model: int) -> torch.Tensor:
    """生成固定 1D 正弦/余弦位置编码（现成工具函数，第 2 章再带你亲手写 1D/2D 版本）。

    Args:
        seq_len: 序列长度 L（本模块中为 2 + chunk_size，对应 [CLS], state, k 个 action）。
        d_model: 隐藏层维度 D（必须为偶数，默认 512）。

    Returns:
        shape 为 [1, seq_len, d_model] 的浮点张量，可直接与 [B, seq_len, d_model] 广播相加。
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model 必须为偶数，当前收到 {d_model}")

    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # [L, 1]
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )  # [D/2]

    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, L, D]


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """重参数化采样：z = mu + sigma * eps, 其中 eps ~ N(0, I)。

    为什么不能直接从 N(mu, sigma^2) 采样？
    因为随机采样节点无法反向传播梯度；把随机性外包给与参数无关的 eps 后，
    mu 和 log_var 仅参与确定性的指数/乘/加运算，梯度即可畅通无阻。

    Args:
        mu:      [B, latent_dim] 后验均值。
        log_var: [B, latent_dim] 后验对数方差 log(sigma^2)。

    Returns:
        z:       [B, latent_dim] 采样得到的风格隐变量。
    """
    # TODO 1: 实现重参数化采样（约 3 行）
    # 工具箱提示:
    #   1) sigma = exp(0.5 * log_var)  -> torch.exp(0.5 * log_var)
    #   2) eps ~ N(0, I)               -> torch.randn_like(std)
    #   3) return mu + std * eps
    raise NotImplementedError("请完成 TODO 1: reparameterize")


def compute_kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """计算后验分布 N(mu, sigma^2) 与标准正态先验 N(0, I) 之间的闭式 KL 散度。

    数学公式（对单个样本的所有 latent_dim 维度求和）：
        KL = -0.5 * sum_j (1 + log_var_j - mu_j^2 - exp(log_var_j))

    然后对 batch 维度取均值，返回一个标量（0 维张量）作为训练 Loss 的一项。

    自检直觉：当 mu 全 0、log_var 全 0（即 sigma^2 = 1）时，返回值应严格等于 0.0。

    Args:
        mu:      [B, latent_dim]
        log_var: [B, latent_dim]

    Returns:
        标量张量（shape 为 []），表示整个 batch 的平均 KL 散度。
    """
    # TODO 2: 实现闭式 KL 散度计算（约 2 行）
    # 工具箱提示:
    #   1) 逐元素项: -0.5 * (1.0 + log_var - mu.pow(2) - torch.exp(log_var))  -> [B, Z]
    #   2) 先对特征维 dim=-1 求和，再对 batch 维求均值 .mean() -> 标量 []
    raise NotImplementedError("请完成 TODO 2: compute_kl_divergence")


class CVAEStyleEncoder(nn.Module):
    """ACT 的 CVAE 风格编码器（仅在训练期启用）。

    数据流总览：
        [CLS]        : [B, 1, D]      (可学习参数)
        state        : [B, 14]  --> Linear --> [B, 1, D]
        actions      : [B, k, 14] --> Linear --> [B, k, D]
                                      │
                   沿 dim=1 拼接 → [B, 2 + k, D] + 1D 正弦位置编码
                                      ▼
                     TransformerEncoder (带 padding mask)
                                      │
                        取出索引 0 的 [CLS] → [B, D]
                                      ▼
                     Linear(D -> 2 * Z) → 拆分为 mu, log_var ([B, Z])
                                      ▼
                          reparameterize → z ([B, Z])
    """

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        chunk_size: int = 100,
        d_model: int = 512,
        latent_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        # 现成零件 1：可学习的 [CLS] token，shape [1, 1, d_model]
        self.cls_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_embed, std=0.02)

        # 现成零件 2：固定 1D 正弦位置编码，注册为 buffer（随模型搬到 GPU，但不参与梯度更新）
        # 序列总长度 = 1 ([CLS]) + 1 (state) + chunk_size (actions)
        pos_embed = build_1d_sinusoidal_pos_embed(2 + chunk_size, d_model)
        self.register_buffer("pos_embed", pos_embed)

        # TODO 3: 定义以下 4 个核心网络模块
        #   1) self.state_proj:  nn.Linear(state_dim, d_model)
        #   2) self.action_proj: nn.Linear(action_dim, d_model)
        #   3) self.encoder:     先构造 nn.TransformerEncoderLayer(
        #                            d_model=d_model, nhead=nhead,
        #                            dim_feedforward=dim_feedforward,
        #                            dropout=dropout, batch_first=True, norm_first=True
        #                        )
        #                        再包进 nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        #   4) self.latent_proj: nn.Linear(d_model, 2 * latent_dim)
        #                        (一次输出拼接在一起的 [mu, log_var])
        raise NotImplementedError("请完成 TODO 3: CVAEStyleEncoder.__init__")

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """将当前状态与真实未来动作块编码为风格隐变量 z 及分布参数。

        Args:
            state:         [B, state_dim]
            actions:       [B, chunk_size, action_dim]
            action_is_pad: 可选 bool 张量 [B, chunk_size]，True 表示该帧是填充帧（应被屏蔽）。

        Returns:
            (z, mu, log_var)，三者 shape 均为 [B, latent_dim]。
        """
        # TODO 4: 实现 CVAE 编码器前向传播
        # 推荐步骤（每步都在脑子里核对一遍张量 shape！）：
        #   1) batch_size = state.shape[0]
        #   2) 投影 state -> [B, 1, D]（提示：先过 self.state_proj，再 .unsqueeze(1)）
        #   3) 投影 actions -> [B, k, D]（提示：直接过 self.action_proj）
        #   4) 扩展 self.cls_embed -> [B, 1, D]（提示：.expand(batch_size, -1, -1)）
        #   5) 沿 dim=1 拼接 [cls_token, state_token, action_tokens] -> x: [B, 2 + k, D]
        #   6) 加上位置编码: x = x + self.pos_embed
        #   7) 构造 src_key_padding_mask（若 action_is_pad 不为 None）：
        #      [CLS] 和 state 这前 2 个 token 永远不是 pad，所以要在前面拼 2 列全 False：
        #      prefix_pad = torch.zeros((batch_size, 2), dtype=torch.bool, device=actions.device)
        #      key_padding_mask = torch.cat([prefix_pad, action_is_pad.bool()], dim=1)  # [B, 2 + k]
        #      若 action_is_pad 为 None，则 key_padding_mask = None
        #   8) 送入 self.encoder(x, src_key_padding_mask=key_padding_mask) -> encoded: [B, 2 + k, D]
        #   9) 取出第 0 个位置 [CLS] 的特征: cls_out = encoded[:, 0, :]  -> [B, D]
        #  10) 投影并拆分: mu, log_var = self.latent_proj(cls_out).chunk(2, dim=-1)  -> 各 [B, Z]
        #  11) 采样: z = reparameterize(mu, log_var)
        #  12) return z, mu, log_var
        raise NotImplementedError("请完成 TODO 4: CVAEStyleEncoder.forward")
