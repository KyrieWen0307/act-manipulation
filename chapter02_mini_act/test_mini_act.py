"""第 2.2 关离线单元测试：验证 Mini-ACT 主干的图像编码、训练/推理双模式、CVAE 隔离与端到端梯度流。

等你亲手完成 chapter02_mini_act/mini_act.py 中的 4 处 TODO 后，在项目根目录运行：
    .venv\\Scripts\\python.exe -m pytest chapter02_mini_act/test_mini_act.py -v
"""

import torch

from chapter01_cvae.cvae import compute_kl_divergence
from chapter02_mini_act.mini_act import MiniACT


def _build_tiny_mini_act(chunk_size: int = 20, dropout: float = 0.0) -> MiniACT:
    """构造轻量小尺寸 MiniACT，保证 CPU 单测几秒内跑完，结构与 512 维完整版完全等价。"""
    return MiniACT(
        state_dim=14,
        action_dim=14,
        chunk_size=chunk_size,
        d_model=64,
        latent_dim=16,
        nhead=4,
        num_cvae_layers=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        backbone_channels=32,
        dropout=dropout,
    )


def test_encode_image_aloha_resolution() -> None:
    """用 ALOHA 真实图像分辨率 [B, 3, 480, 640] 验证视觉编码输出 [B, 300, D]。"""
    model = _build_tiny_mini_act()
    image = torch.randn(2, 3, 480, 640)

    img_tokens = model.encode_image(image)
    assert img_tokens.shape == (2, 300, 64), (
        f"480x640 经 32 倍下采样应为 15x20=300 个视觉页，期望 (2, 300, 64)，实际得到 {img_tokens.shape}"
    )


def test_mini_act_training_mode_shapes() -> None:
    """验证训练期（传入 actions）输出动作块、mu、log_var 的全部维度。"""
    model = _build_tiny_mini_act(chunk_size=100)
    image = torch.randn(2, 3, 128, 128)
    state = torch.randn(2, 14)
    actions = torch.randn(2, 100, 14)
    action_is_pad = torch.zeros(2, 100, dtype=torch.bool)
    action_is_pad[:, 90:] = True

    pred_actions, mu, log_var = model(image, state, actions=actions, action_is_pad=action_is_pad)

    assert pred_actions.shape == (2, 100, 14)
    assert mu is not None and mu.shape == (2, 16)
    assert log_var is not None and log_var.shape == (2, 16)


def test_mini_act_inference_mode_skips_cvae_and_is_deterministic() -> None:
    """核心铁律测试：推理期 (actions=None) 必须彻底跳过 CVAEStyleEncoder，且固定 z=0 确定性输出。"""
    model = _build_tiny_mini_act(chunk_size=20, dropout=0.0)
    model.eval()

    # 给 CVAE 装上“绊线”：如果推理期不小心调用了它，立刻报错！
    def _forbid_cvae(*args, **kwargs):
        raise AssertionError("推理期 (actions=None) 绝对不能调用 self.cvae_encoder！")

    model.cvae_encoder.forward = _forbid_cvae  # type: ignore[method-assign]

    image = torch.randn(3, 3, 128, 128)
    state = torch.randn(3, 14)

    with torch.no_grad():
        pred1, mu1, log_var1 = model(image, state)
        pred2, mu2, log_var2 = model(image, state)

    assert pred1.shape == (3, 20, 14)
    assert mu1 is None and log_var1 is None
    assert mu2 is None and log_var2 is None
    assert torch.allclose(pred1, pred2, atol=1e-6), "推理期固定 z=0，eval 模式下两次输出必须完全一致！"


def test_mini_act_forwards_action_is_pad_to_cvae() -> None:
    """验证训练期传入的 action_is_pad 确实传递给了 CVAEStyleEncoder，屏蔽了尾部垃圾填充值。"""
    torch.manual_seed(7)
    model = _build_tiny_mini_act(chunk_size=20, dropout=0.0)
    model.eval()

    image = torch.randn(2, 3, 128, 128)
    state = torch.randn(2, 14)
    actions_clean = torch.randn(2, 20, 14)
    actions_corrupted = actions_clean.clone()
    actions_corrupted[:, 15:, :] = 1e5

    action_is_pad = torch.zeros(2, 20, dtype=torch.bool)
    action_is_pad[:, 15:] = True

    with torch.no_grad():
        _, mu_clean, log_var_clean = model(image, state, actions=actions_clean, action_is_pad=action_is_pad)
        _, mu_corrupted, log_var_corrupted = model(image, state, actions=actions_corrupted, action_is_pad=action_is_pad)

    assert mu_clean is not None and mu_corrupted is not None
    assert log_var_clean is not None and log_var_corrupted is not None
    assert torch.allclose(mu_clean, mu_corrupted, atol=1e-5), "action_is_pad 未正确传入 cvae_encoder！"
    assert torch.allclose(log_var_clean, log_var_corrupted, atol=1e-5)


def test_mini_act_end_to_end_backward_all_params() -> None:
    """验证训练期端到端前向 + L1 + KL 反向传播，所有子模块参数都能收到非空且有限的梯度。"""
    model = _build_tiny_mini_act(chunk_size=20, dropout=0.1)
    image = torch.randn(2, 3, 128, 128)
    state = torch.randn(2, 14)
    actions = torch.randn(2, 20, 14)

    pred_actions, mu, log_var = model(image, state, actions=actions)
    assert mu is not None and log_var is not None

    l1_loss = torch.nn.functional.l1_loss(pred_actions, actions)
    kl_loss = compute_kl_divergence(mu, log_var)
    loss = l1_loss + 10.0 * kl_loss
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有收到梯度，计算图存在断流！"
        assert torch.isfinite(param.grad).all(), f"参数 {name} 出现 NaN 或 Inf 梯度！"


def test_mini_act_responds_to_image_and_state() -> None:
    """防哑巴测试：改变输入照片或关节状态时，预测出的动作必须随之变化。"""
    model = _build_tiny_mini_act(chunk_size=20, dropout=0.0)
    model.eval()

    img1 = torch.randn(1, 3, 128, 128)
    img2 = torch.randn(1, 3, 128, 128)
    state1 = torch.randn(1, 14)
    state2 = torch.randn(1, 14)

    with torch.no_grad():
        pred_base, _, _ = model(img1, state1)
        pred_diff_img, _, _ = model(img2, state1)
        pred_diff_state, _, _ = model(img1, state2)

    assert not torch.allclose(pred_base, pred_diff_img), "换了图片但预测动作没变化，请检查 img_tokens 是否接入 Encoder！"
    assert not torch.allclose(pred_base, pred_diff_state), "换了关节状态但预测动作没变化，请检查 state_token 是否接入 Encoder！"
