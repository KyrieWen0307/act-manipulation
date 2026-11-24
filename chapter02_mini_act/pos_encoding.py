"""第 2.1 关手写实战：1D 与 2D 正弦/余弦位置编码（Sinusoidal Positional Encoding）。

本模块实现给“序列”和“图像棋盘”盖坐标邮戳的两个核心函数：
    1) build_1d_sinusoidal_pos_embed(seq_len, d_model) -> [1, seq_len, d_model]
       给一排长度为 seq_len 的格子编上 1D 序号（0, 1, 2, ...）。
    2) build_2d_sinusoidal_pos_embed(height, width, d_model) -> [1, d_model, height, width]
       把通道数 d_model 一分为二：
         - 前 half_dim = d_model // 2 个通道：专门盖“第几行 (y)”的邮戳；
         - 后 half_dim = d_model // 2 个通道：专门盖“第几列 (x)”的邮戳。
       最终得到 [1, d_model, height, width]，可直接与 CNN 特征图 [B, d_model, H, W] 相加！

请按顺序完成文件中的 2 处 TODO（先对照注释里的变量名与张量 shape 自己手敲）。
"""

from __future__ import annotations

import math

import torch


def build_1d_sinusoidal_pos_embed(seq_len: int, d_model: int) -> torch.Tensor:
    """生成固定 1D 正弦/余弦位置编码。

    原理速记：
        对第 pos 个位置（0 <= pos < seq_len），偶数通道放 sin(pos * freq)，
        奇数通道放 cos(pos * freq)。不同通道使用不同频率，像时钟的“秒针/分针/时针”，
        保证每个位置都有独一无二的坐标编码。

    Args:
        seq_len: 序列长度 L（正整数）。
        d_model: 特征维度 D（必须为偶数，因为 sin/cos 两两配对）。

    Returns:
        shape 为 [1, seq_len, d_model] 的 float32 张量。
    """
    if d_model % 2 != 0:
        raise ValueError(f"1D 位置编码要求 d_model 必须为偶数，当前收到 {d_model}")

    # TODO 1: 实现 1D 正弦/余弦位置编码（约 7 行）
    # 推荐变量名与 shape 速查：
    #   1) position: [seq_len, 1]
    #      torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    #   2) div_term: [d_model // 2]
    #      torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    #   3) pe:       [seq_len, d_model]  全 0 初始画布
    #      torch.zeros(seq_len, d_model, dtype=torch.float32)
    #   4) 偶数位填 sin: pe[:, 0::2] = torch.sin(position * div_term)
    #      奇数位填 cos: pe[:, 1::2] = torch.cos(position * div_term)
    #   5) 返回前补上 batch 维 [1, seq_len, d_model]: return pe.unsqueeze(0)
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def build_2d_sinusoidal_pos_embed(
    height: int,
    width: int,
    d_model: int,
) -> torch.Tensor:
    """生成固定 2D 正弦/余弦位置编码，用于给 CNN 特征图 [B, D, H, W] 盖行/列坐标戳。

    核心思想（复用上面写好的 1D 函数！）：
        把通道维 D 劈成两半（half_dim = d_model // 2）：
        - 前半段 [1, half_dim, H, W]：同一行的所有格子共享同一个“行坐标 y 编码”；
        - 后半段 [1, half_dim, H, W]：同一列的所有格子共享同一个“列坐标 x 编码”；
        - 沿通道维 dim=1 拼接 -> [1, D, H, W]！

    Args:
        height:  特征图行数 H（如 ALOHA 480/32 = 15）。
        width:   特征图列数 W（如 ALOHA 640/32 = 20）。
        d_model: 特征维度 D（必须能被 4 整除，因为劈成两半后每一半还要成对放 sin/cos）。

    Returns:
        shape 为 [1, d_model, height, width] 的 float32 张量，可直接与 [B, D, H, W] 广播相加。
    """
    if d_model % 4 != 0:
        raise ValueError(f"2D 位置编码要求 d_model 必须能被 4 整除，当前收到 {d_model}")

    # TODO 2: 复用 build_1d_sinusoidal_pos_embed 实现 2D 位置编码（约 7 行）
    # 推荐变量名与 shape 速查（每写一行都在脑子里核对一遍 4 个数字！）：
    #   1) half_dim = d_model // 2
    #   2) 先生成行坐标 1D 编码:
    #      y_embed = build_1d_sinusoidal_pos_embed(height, half_dim)         -> [1, H, D/2]
    #      变形并沿“列方向”铺满整个棋盘:
    #      y_grid = y_embed.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, width)
    #      (慢动作: [1, H, D/2] -transpose-> [1, D/2, H] -unsqueeze(3)-> [1, D/2, H, 1] -expand-> [1, D/2, H, W])
    #   3) 再生成列坐标 1D 编码:
    #      x_embed = build_1d_sinusoidal_pos_embed(width, half_dim)          -> [1, W, D/2]
    #      变形并沿“行方向”铺满整个棋盘:
    #      x_grid = x_embed.transpose(1, 2).unsqueeze(2).expand(-1, -1, height, -1)
    #      (慢动作: [1, W, D/2] -transpose-> [1, D/2, W] -unsqueeze(2)-> [1, D/2, 1, W] -expand-> [1, D/2, H, W])
    #   4) 沿通道维 dim=1 拼接（其余 3 个维度 1, H, W 完全相同，正好拼出 D/2 + D/2 = D！）:
    #      pos_2d = torch.cat([y_grid, x_grid], dim=1)                       -> [1, D, H, W]
    #   5) return pos_2d
    half_dim = d_model // 2
    y_embed = build_1d_sinusoidal_pos_embed(height, half_dim)
    y_grid = y_embed.transpose(1, 2).unsqueeze(3).expand(-1, -1, -1, width)
    x_embed = build_1d_sinusoidal_pos_embed(width, half_dim)
    x_grid = x_embed.transpose(1, 2).unsqueeze(2).expand(-1, -1, height, -1)
    pos_2d = torch.cat([y_grid, x_grid], dim=1)
    return pos_2d