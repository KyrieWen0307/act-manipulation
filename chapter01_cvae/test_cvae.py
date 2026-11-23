"""第 1 章离线单元测试：验证 CVAEStyleEncoder 的 shape、梯度流、KL 性质与 Padding 隔离。

等你亲手完成 chapter01_cvae/cvae.py 中的 4 处 TODO 后，在项目根目录运行：
    .venv\\Scripts\\python.exe -m pytest chapter01_cvae/test_cvae.py -v
"""

import torch

from chapter01_cvae.cvae import (
    CVAEStyleEncoder,
    compute_kl_divergence,
    reparameterize,
)


def test_reparameterize_shape_and_grad() -> None:
    """验证重参数化输出的 shape、随机性以及梯度能否畅通流回 mu 与 log_var。"""
    torch.manual_seed(42)
    mu = torch.randn(4, 32, requires_grad=True)
    log_var = torch.randn(4, 32, requires_grad=True)

    z1 = reparameterize(mu, log_var)
    z2 = reparameterize(mu, log_var)

    assert z1.shape == (4, 32), f"期望 shape 为 (4, 32)，实际得到 {z1.shape}"
    assert not torch.allclose(z1, z2), "两次随机采样的结果不应完全相同，请确认是否加入了 eps 噪声"

    loss = z1.sum()
    loss.backward()
    assert mu.grad is not None and torch.allclose(mu.grad, torch.ones_like(mu)), "mu 的梯度应全为 1"
    assert log_var.grad is not None and log_var.grad.abs().sum() > 0, "log_var 必须收到非零梯度"


def test_kl_divergence_standard_normal_and_positive() -> None:
    """验证标准正态分布时 KL 严格为 0，偏离标准正态时 KL 严格大于 0。"""
    mu_zero = torch.zeros(8, 32)
    log_var_zero = torch.zeros(8, 32)
    kl_zero = compute_kl_divergence(mu_zero, log_var_zero)

    assert kl_zero.ndim == 0, f"KL 返回值应为标量 (0 维张量)，实际维度为 {kl_zero.ndim}"
    assert torch.isclose(kl_zero, torch.tensor(0.0), atol=1e-6), f"标准正态时 KL 应为 0，实际为 {kl_zero.item()}"

    mu_shifted = torch.ones(8, 32, requires_grad=True) * 2.0
    log_var_shifted = torch.ones(8, 32, requires_grad=True) * (-1.0)
    kl_pos = compute_kl_divergence(mu_shifted, log_var_shifted)
    assert kl_pos.item() > 0.0, "偏离标准正态分布时，KL 散度必须严格大于 0"

    kl_pos.backward()
    assert mu_shifted.grad is not None and mu_shifted.grad.abs().sum() > 0
    assert log_var_shifted.grad is not None and log_var_shifted.grad.abs().sum() > 0


def test_cvae_encoder_forward_shapes() -> None:
    """用真实 ALOHA 维度 (state=14, action=14, chunk=100, latent=32) 验证前向输出 shape。"""
    model = CVAEStyleEncoder(
        state_dim=14,
        action_dim=14,
        chunk_size=100,
        d_model=128,  # 单测用 128 维跑得飞快，结构与 512 完全等价
        latent_dim=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    )
    state = torch.randn(4, 14)
    actions = torch.randn(4, 100, 14)

    z, mu, log_var = model(state, actions)
    assert z.shape == (4, 32)
    assert mu.shape == (4, 32)
    assert log_var.shape == (4, 32)


def test_cvae_encoder_padding_mask_isolation() -> None:
    """硬核防踩坑测试：尾部被 action_is_pad=True 标记的帧即使填入巨大垃圾值，也不应污染 mu 和 log_var。"""
    torch.manual_seed(0)
    model = CVAEStyleEncoder(
        state_dim=14,
        action_dim=14,
        chunk_size=20,
        d_model=64,
        latent_dim=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.0,  # 关掉 dropout 确保两次前向确定性可比
    )
    model.eval()

    state = torch.randn(2, 14)
    actions_clean = torch.randn(2, 20, 14)
    actions_corrupted = actions_clean.clone()
    # 假设最后 5 帧是 episode 末尾的 padding 帧，故意塞进极大垃圾数值
    actions_corrupted[:, 15:, :] = 1e5

    action_is_pad = torch.zeros(2, 20, dtype=torch.bool)
    action_is_pad[:, 15:] = True

    with torch.no_grad():
        _, mu_clean, log_var_clean = model(state, actions_clean, action_is_pad=action_is_pad)
        _, mu_corrupted, log_var_corrupted = model(state, actions_corrupted, action_is_pad=action_is_pad)

    assert torch.allclose(mu_clean, mu_corrupted, atol=1e-5), (
        "传入 action_is_pad 后，padding 帧的数值污染了 mu！请检查 src_key_padding_mask 拼接顺序。"
    )
    assert torch.allclose(log_var_clean, log_var_corrupted, atol=1e-5), (
        "传入 action_is_pad 后，padding 帧的数值污染了 log_var！"
    )


def test_cvae_encoder_backward_end_to_end() -> None:
    """验证整个 CVAEStyleEncoder 端到端反向传播无断流、无 NaN。"""
    model = CVAEStyleEncoder(
        state_dim=14,
        action_dim=14,
        chunk_size=20,
        d_model=64,
        latent_dim=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    )
    state = torch.randn(3, 14)
    actions = torch.randn(3, 20, 14)
    action_is_pad = torch.zeros(3, 20, dtype=torch.bool)
    action_is_pad[:, 17:] = True

    z, mu, log_var = model(state, actions, action_is_pad=action_is_pad)
    loss = z.pow(2).mean() + compute_kl_divergence(mu, log_var)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有收到梯度，计算图存在断流！"
        assert torch.isfinite(param.grad).all(), f"参数 {name} 出现 NaN 或 Inf 梯度！"
