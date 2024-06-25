import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# CNN 模型
class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=256, dropout_prob=0.1):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=9, stride=9)
        self.fc1 = nn.Linear(hidden_size * (input_shape // 9), hidden_size)  # 确保输出维度正确
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度，形状变为 [batch_size, 1, seq_len]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformer 模型
class TransformerQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=256, num_heads=8, num_layers=3, dropout_prob=0.1):
        super(TransformerQNetwork, self).__init__()
        self.embedding = nn.Linear(input_shape, hidden_size)
        
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_shape, hidden_size))

        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, num_actions)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)  # [batch_size, seq_len, hidden_size]
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_size]
        x += self.positional_encoding[:, :seq_len, :]  # 添加位置编码
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 全连接网络模型
class FCQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_size=256):
        super(FCQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_actions)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        #x = self.dropout(x)
        x = self.fc4(x)
        return x

# 设置固定种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 定义组合模型
class EnsembleQNetwork(nn.Module):
    def __init__(self, cnn_model, transformer_model, fc_model, num_actions):
        super(EnsembleQNetwork, self).__init__()
        self.cnn_model = cnn_model
        self.transformer_model = transformer_model
        self.fc_model = fc_model
        self.fc = nn.Linear(num_actions * 3, num_actions)  # 将三个模型的输出组合在一起

    def forward(self, x):
        cnn_output = self.cnn_model(x)
        transformer_output = self.transformer_model(x)
        fc_output = self.fc_model(x)
        combined_output = torch.cat((cnn_output, transformer_output, fc_output), dim=1)
        final_output = self.fc(combined_output)
        return final_output

