"""第 2.1 关离线单元测试：验证 1D 与 2D 正弦位置编码的 shape、行列语义与唯一性。

等你亲手完成 chapter02_mini_act/pos_encoding.py 中的 2 处 TODO 后，在项目根目录运行：
    .venv\\Scripts\\python.exe -m pytest chapter02_mini_act/test_pos_encoding.py -v
"""

import pytest
import torch

from chapter01_cvae.cvae import build_1d_sinusoidal_pos_embed as ref_1d_pos_embed
from chapter02_mini_act.pos_encoding import (
    build_1d_sinusoidal_pos_embed,
    build_2d_sinusoidal_pos_embed,
)


def test_1d_pos_embed_shape_values_and_match_ch1() -> None:
    """验证 1D 位置编码的 shape、数值范围，以及与第 1 章参考实现完全一致。"""
    pe = build_1d_sinusoidal_pos_embed(seq_len=102, d_model=64)

    assert pe.shape == (1, 102, 64), f"期望 shape (1, 102, 64)，实际得到 {pe.shape}"
    assert pe.dtype == torch.float32
    assert pe.min().item() >= -1.0 and pe.max().item() <= 1.0

    # 位置 0 的偶数通道应为 sin(0)=0，奇数通道应为 cos(0)=1
    assert torch.allclose(pe[0, 0, 0::2], torch.zeros(32), atol=1e-6)
    assert torch.allclose(pe[0, 0, 1::2], torch.ones(32), atol=1e-6)

    # 与第 1 章现成的参考函数结果严格对齐
    ref = ref_1d_pos_embed(seq_len=102, d_model=64)
    assert torch.allclose(pe, ref, atol=1e-6)

    with pytest.raises(ValueError):
        build_1d_sinusoidal_pos_embed(seq_len=10, d_model=63)


def test_2d_pos_embed_shape_and_validation() -> None:
    """用 ALOHA 真实特征图尺寸 (H=15, W=20, D=512) 验证输出 shape 与合法性检查。"""
    pos_2d = build_2d_sinusoidal_pos_embed(height=15, width=20, d_model=512)

    assert pos_2d.shape == (1, 512, 15, 20), f"期望 shape (1, 512, 15, 20)，实际得到 {pos_2d.shape}"
    assert pos_2d.dtype == torch.float32

    # 不能被 4 整除时应报错
    with pytest.raises(ValueError):
        build_2d_sinusoidal_pos_embed(height=15, width=20, d_model=62)


def test_2d_pos_embed_row_and_col_semantics() -> None:
    """核心语义测试：前半段通道只认行坐标 y，后半段通道只认列坐标 x。"""
    d_model = 64
    half = d_model // 2
    pos_2d = build_2d_sinusoidal_pos_embed(height=6, width=8, d_model=d_model)

    # 1) 同一行 (y=2)，不同列 (x=1 vs x=5)：
    #    行编码 (前 half 维) 必须完全相同；列编码 (后 half 维) 必须不同！
    assert torch.allclose(pos_2d[0, :half, 2, 1], pos_2d[0, :half, 2, 5]), (
        "同一行不同列的格子，前半段行编码应当完全相同！请检查 y_grid 的 unsqueeze/expand 维度。"
    )
    assert not torch.allclose(pos_2d[0, half:, 2, 1], pos_2d[0, half:, 2, 5]), (
        "同一行不同列的格子，后半段列编码应当不同！"
    )

    # 2) 不同行 (y=1 vs y=4)，同一列 (x=3)：
    #    列编码 (后 half 维) 必须完全相同；行编码 (前 half 维) 必须不同！
    assert torch.allclose(pos_2d[0, half:, 1, 3], pos_2d[0, half:, 4, 3]), (
        "不同行同一列的格子，后半段列编码应当完全相同！请检查 x_grid 的 unsqueeze/expand 维度。"
    )
    assert not torch.allclose(pos_2d[0, :half, 1, 3], pos_2d[0, :half, 4, 3]), (
        "不同行同一列的格子，前半段行编码应当不同！"
    )


def test_2d_pos_embed_all_grid_cells_unique() -> None:
    """验证 15x20 棋盘上全部 300 个格子的 2D 坐标邮戳两两互不重复。"""
    pos_2d = build_2d_sinusoidal_pos_embed(height=15, width=20, d_model=64)
    # 拆成 [300, 64]
    tokens = pos_2d.squeeze(0).flatten(1).transpose(0, 1)  # [300, 64]
    assert tokens.shape == (300, 64)

    # 计算两两欧氏距离，对角线以外最小距离必须严格大于 0
    dists = torch.cdist(tokens, tokens)
    dists.fill_diagonal_(float("inf"))
    assert dists.min().item() > 1e-3, "棋盘上存在两个不同格子拥有相同的 2D 位置编码！"


def test_2d_pos_embed_add_and_flatten_to_img_tokens() -> None:
    """端到端模拟积木 2：特征图 [B, D, H, W] 加上 2D 邮戳后，拉直转置为 [B, H*W, D]。"""
    batch_size, d_model, height, width = 4, 128, 15, 20
    feat_map = torch.randn(batch_size, d_model, height, width)
    pos_2d = build_2d_sinusoidal_pos_embed(height=height, width=width, d_model=d_model)

    feat_with_pos = feat_map + pos_2d  # 广播相加: [4, 128, 15, 20]
    img_tokens = feat_with_pos.flatten(2).transpose(1, 2)  # [4, 300, 128]

    assert feat_with_pos.shape == (batch_size, d_model, height, width)
    assert img_tokens.shape == (batch_size, height * width, d_model)
