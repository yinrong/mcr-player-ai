from common import nn, torch


import torch
import torch.nn as nn


class DiscardModel(nn.Module):
    def __init__(
        self,
        num_inputs=10,          # 输入特征维度（N）
        num_rows=4,             # 输入的行数
        num_cols=9,             # 输入的列数
        embed_dim=8,           # 通用卷积嵌入维度
        mlp_hidden_dim=256,     # MLP 隐藏层维度
        output_dim=34           # 输出类别数
    ):
        super(DiscardModel, self).__init__()

        # ===== 每个 N 维度的独立 CNN =====
        self.individual_cnns = nn.ModuleList()
        for _ in range(num_inputs):
            self.individual_cnns.append(
                nn.Sequential(
                    nn.Conv2d(1, embed_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(),
                    nn.Dropout2d(p=0.8),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
            )

        # ===== 每个 N 维度的独立 MLP =====
        self.individual_mlps = nn.ModuleList()
        for _ in range(num_inputs):
            self.individual_mlps.append(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_rows * num_cols, mlp_hidden_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.8),
                    nn.Linear(mlp_hidden_dim, embed_dim),
                )
            )

        # ===== 最终输出层 =====
        total_embed_dim = embed_dim * 2 * num_inputs
        self.output_layer = nn.Sequential(
            nn.Linear(total_embed_dim, output_dim),
            nn.LeakyReLU(),
        )

    @staticmethod
    def create_hand_mask(hands, max_tile_index=34):
        """
        根据手牌生成掩码，确保输出合法。
        hands: Tensor, shape=[B, 4, 9]
            当前批量的手牌张数。
        max_tile_index: int
            最大牌索引。
        """
        batch_size = hands.size(0)
        mask = torch.zeros((batch_size, max_tile_index), dtype=torch.float32, device=hands.device)
        for i in range(batch_size):
            for r in range(hands.size(1)):  # 遍历万/条/饼/字
                for c in range(hands.size(2)):  # 遍历列
                    tile_index = r * 9 + c
                    if tile_index < max_tile_index:
                        mask[i, tile_index] += hands[i, r, c]
        return mask

    def forward(self, x):
        """
        x: Tensor, shape=[B, N, 4, 9]
        """
        batch_size, num_inputs, _, _ = x.shape

        # ===== 每个 N 维度的独立 CNN 分支 =====
        cnn_features = []
        for i in range(num_inputs):
            x_i = x[:, i:i+1, :, :]  # 取出第 i 个输入维度，形状 [B, 1, 4, 9]
            cnn_features.append(self.individual_cnns[i](x_i).view(batch_size, -1))
        cnn_features = torch.cat(cnn_features, dim=1)  # 形状 [B, num_inputs * embed_dim]

        # ===== 每个 N 维度的独立 MLP 分支 =====
        mlp_features = []
        for i in range(num_inputs):
            x_i = x[:, i:i+1, :, :]  # 取出第 i 个输入维度，形状 [B, 1, 4, 9]
            mlp_features.append(self.individual_mlps[i](x_i))
        mlp_features = torch.cat(mlp_features, dim=1)  # 形状 [B, num_inputs * embed_dim]

        # ===== 特征融合 =====
        fused_features = torch.cat([cnn_features, mlp_features], dim=1)

        # ===== 输出层 =====
        logits = self.output_layer(fused_features)  # 形状 [B, output_dim]

        # ===== 应用掩码，限制输出范围 =====
        hands = x[:, 0]  # 假设第一个维度代表手牌
        #print('hands:    ', hands[0])
        hand_mask = self.create_hand_mask(hands)
        #print('hand_mask:')
        #print(hand_mask)
        masked_logits = torch.where(hand_mask < 1, torch.tensor(1e-6, device=logits.device), logits)
        #print('masked:')
        #print(masked_logits)
        return masked_logits