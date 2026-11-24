"""第 2.2 关手写实战：组装完整的 Mini-ACT 主干架构。

本模块把前面学过的所有积木拼成一台完整的 Mini-ACT：
    - 第 1 章成果: CVAEStyleEncoder（仅训练期启用，提炼动作风格 z）
    - 第 2.1 关成果: build_2d_sinusoidal_pos_embed（给图像特征棋盘盖 2D 坐标邮戳）
    - 本关新零件:
        1) SimpleVisionBackbone: 5 层步长为 2 的卷积（共 32 倍下采样，与 ResNet-18 尺寸完全一致）
        2) Policy Transformer Encoder（情报分析部）: 融合 [z, state, img_tokens] -> memory
        3) Policy Transformer Decoder（行动规划部）: 100 张 Action Queries 查 memory -> [B, 100, D]
        4) Action Head: Linear(D -> 14) 一次并行吐出未来 100 步双臂动作 [B, 100, 14]

请按顺序完成文件中的 4 处 TODO（每步都在脑子里核对一遍张量 shape！）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from chapter01_cvae.cvae import CVAEStyleEncoder
from chapter02_mini_act.pos_encoding import build_2d_sinusoidal_pos_embed


class SimpleVisionBackbone(nn.Module):
    """现成的轻量级视觉骨干网络（纯 CPU 友好，无需联网下载权重）。

    功能等价于去掉末尾池化与分类头的 ResNet-18（同样做 32 倍空间下采样）：
        - 输入 ALOHA 照片: [B, 3, 480, 640]
        - 输出特征棋盘:    [B, out_channels, 15, 20]
    第 4 章去 Colab GPU 正式训练时，可直接无缝替换为 torchvision 的预训练 ResNet-18。
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        c1 = max(16, out_channels // 4)
        c2 = max(16, out_channels // 2)
        c3 = out_channels
        # 连续 5 次 stride=2 卷积 -> 长宽各缩小 2^5 = 32 倍
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MiniACT(nn.Module):
    """手写教学版 ACT（Action Chunking with Transformers）完整策略网络。"""

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        chunk_size: int = 100,
        d_model: int = 512,
        latent_dim: int = 32,
        nhead: int = 8,
        num_cvae_layers: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        dim_feedforward: int = 3200,
        backbone_channels: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"MiniACT 内部使用 2D 位置编码，d_model 必须能被 4 整除，当前收到 {d_model}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.latent_dim = latent_dim

        # ==================== 现成零件（已为你装配好） ====================
        # 现成零件 1：第 1 章亲手写的 CVAE 风格编码器（仅训练期启用）
        self.cvae_encoder = CVAEStyleEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
            num_layers=num_cvae_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 现成零件 2：视觉骨干网络，把 [B, 3, H_in, W_in] 浓缩成 [B, backbone_channels, H, W]
        self.vision_backbone = SimpleVisionBackbone(out_channels=backbone_channels)

        # 现成零件 3：给前 2 页非图像材料 [z_token, state_token] 贴的可学习身份标签 [1, 2, D]
        self.non_visual_pos_embed = nn.Parameter(torch.zeros(1, 2, d_model))
        nn.init.normal_(self.non_visual_pos_embed, std=0.02)

        # 现成零件 4：行动部 (Decoder) 的 chunk_size 张专属空白工单 [1, chunk_size, D]
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)

        # ==================== TODO 1：定义核心连接层与两部门 ====================
        # 请定义以下 6 个核心属性（推荐名称与规格如下）：
        #   1) self.img_proj:
        #      nn.Conv2d(backbone_channels, d_model, kernel_size=1)
        #      （1x1 卷积：不改棋盘 HxW 大小，只把每格通道数从 backbone_channels 变成 d_model）
        #   2) self.z_proj:
        #      nn.Linear(latent_dim, d_model)
        #   3) self.state_proj:
        #      nn.Linear(state_dim, d_model)
        #   4) self.policy_encoder:（情报分析部）
        #      先建 encoder_layer = nn.TransformerEncoderLayer(
        #          d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        #          dropout=dropout, batch_first=True, norm_first=True,
        #      )
        #      再包进 nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        #   5) self.policy_decoder:（行动规划部）
        #      先建 decoder_layer = nn.TransformerDecoderLayer(
        #          d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
        #          dropout=dropout, batch_first=True, norm_first=True,
        #      )
        #      再包进 nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        #   6) self.action_head:
        #      nn.Linear(d_model, action_dim)
        #      （把行动部每张工单的 d_model 个数字，翻译回 action_dim=14 个关节动作）
        self.img_proj = nn.Conv2d(backbone_channels, d_model, kernel_size=1)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.state_proj = nn.Linear(self.state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True)
        self.policy_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True)
        self.policy_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.action_head = nn.Linear(d_model, action_dim)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """把原始图像 [B, 3, H_in, W_in] 变成带 2D 坐标邮戳的视觉页 [B, H*W, D]。

        Args:
            image: [B, 3, H_in, W_in] 相机画面张量。

        Returns:
            img_tokens: [B, H*W, d_model] 拆好棋盘且带 2D 坐标的视觉 token 序列。
        """
        # ==================== TODO 2：图像 -> 棋盘 -> 盖 2D 邮戳 -> 拆页 ====================
        # 推荐步骤与变量名（约 6 行）：
        #   1) 过视觉骨干: feat = self.vision_backbone(image)           -> [B, C_vis, H, W]
        #   2) 1x1 卷积对齐通道: feat = self.img_proj(feat)             -> [B, D, H, W]
        #   3) 读取棋盘高宽: _, _, height, width = feat.shape
        #   4) 生成 2D 坐标底图并搬到和 feat 同样的设备/精度:
        #      pos_2d = build_2d_sinusoidal_pos_embed(height, width, self.d_model).to(
        #          device=feat.device, dtype=feat.dtype
        #      )                                                        -> [1, D, H, W]
        #   5) 盖上坐标邮戳: feat = feat + pos_2d                       -> [B, D, H, W]
        #   6) 拆棋盘并转身: img_tokens = feat.flatten(2).transpose(1, 2) -> [B, H*W, D]
        #   7) return img_tokens
        feat = self.vision_backbone(image)
        feat = self.img_proj(feat) 
        _, _, height, width = feat.shape
        pos_2d = build_2d_sinusoidal_pos_embed(height, width, self.d_model).to(device=feat.device, dtype=feat.dtype)
        feat = feat + pos_2d
        img_tokens = feat.flatten(2).transpose(1, 2)
        return img_tokens
    
    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Mini-ACT 前向传播（自动支持训练期 / 推理期双模式切换）。

        Args:
            image:         [B, 3, H_in, W_in] 相机图像。
            state:         [B, state_dim] 当前关节状态。
            actions:       [B, chunk_size, action_dim] 可选；传入则进入【训练期】，
                           为 None 则进入【推理期】（跳过 CVAE，强制 z = 0）。
            action_is_pad: [B, chunk_size] 可选 bool 掩码，仅训练期传给 CVAE 屏蔽填充帧。

        Returns:
            (pred_actions, mu, log_var):
                - pred_actions: [B, chunk_size, action_dim] 预测的未来动作块。
                - mu, log_var:  训练期各为 [B, latent_dim]；推理期均为 None。
        """
        batch_size = state.shape[0]

        # ==================== TODO 3：训练 / 推理双模式切换获取 z ====================
        # 推荐逻辑（约 6 行）：
        #   if actions is not None:
        #       # 训练期：调用第 1 章的 self.cvae_encoder(state, actions, action_is_pad=action_is_pad)
        #       z, mu, log_var = ...
        #   else:
        #       # 推理期：彻底跳过 CVAE，直接构造全 0 的 z（形状 [batch_size, self.latent_dim]，
        #       # 注意指定 dtype=state.dtype, device=state.device），并设 mu = None, log_var = None
        #       z = ...
        #       mu = None
        #       log_var = None
        if actions is not None:
            z, mu, log_var = self.cvae_encoder(state, actions, action_is_pad=action_is_pad)
        else:
            z = torch.zeros((batch_size, self.latent_dim), dtype=state.dtype, device=state.device)
            mu = None
            log_var = None

        # ==================== TODO 4：拼大书 -> 情报部 -> 行动部 -> 输出动作 ====================
        # 推荐步骤与变量名（约 9 行）：
        #   1) z_token = self.z_proj(z).unsqueeze(1)                    -> [B, 1, D]
        #   2) state_token = self.state_proj(state).unsqueeze(1)        -> [B, 1, D]
        #   3) 前 2 页拼起来并贴上身份标签:
        #      non_visual_tokens = torch.cat([z_token, state_token], dim=1) + self.non_visual_pos_embed
        #                                                               -> [B, 2, D]
        #   4) 提取带 2D 坐标的图像页:
        #      img_tokens = self.encode_image(image)                    -> [B, H*W, D]
        #   5) 堆叠成情报部完整大书:
        #      enc_input = torch.cat([non_visual_tokens, img_tokens], dim=1)
        #                                                               -> [B, 2 + H*W, D]
        #   6) 情报分析部开会，整理出全局记忆库:
        #      memory = self.policy_encoder(enc_input)                  -> [B, 2 + H*W, D]
        #   7) 行动规划部准备 B 份专属工单，并查 memory:
        #      queries = self.action_queries.expand(batch_size, -1, -1) -> [B, chunk_size, D]
        #      decoded = self.policy_decoder(tgt=queries, memory=memory)-> [B, chunk_size, D]
        #   8) 翻译回动作空间:
        #      pred_actions = self.action_head(decoded)                 -> [B, chunk_size, action_dim]
        #   9) return pred_actions, mu, log_var
        z_token = self.z_proj(z).unsqueeze(1)
        state_token = self.state_proj(state).unsqueeze(1)
        non_visual_tokens = torch.cat([z_token, state_token], dim=1) + self.non_visual_pos_embed
        img_tokens = self.encode_image(image)
        enc_input = torch.cat([non_visual_tokens, img_tokens], dim=1)
        memory = self.policy_encoder(enc_input) 
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.policy_decoder(tgt=queries, memory=memory)
        pred_actions = self.action_head(decoded)
        return pred_actions, mu, log_var