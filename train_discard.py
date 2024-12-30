import pickle
from common import *
from quezha_parser_1 import getData
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Model(nn.Module):

    """
    同时使用1D-CNN对 x_tiles 提取"静态局面"特征,
    用RNN对 history_seq 提取"时序弃牌"特征,
    融合后输出对 [0..33] 每种牌的打牌倾向.
    """
    def __init__(
        self,
        in_channels=3,          # 0:我的手牌,1:我的副露,2:他人副露
        cnn_hidden=32,
        cnn_layers=2,
        
        num_tile_types=34,      # 牌种类(输出分类数)
        
        rnn_embed_dim=32,       # history_seq中每张牌embedding维度
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        
        hidden_dim=128
    ):
        super().__init__()
        
        # ---- 1D-CNN for x_tiles ----
        cnn_modules = []
        ch_in = in_channels
        for i in range(cnn_layers):
            ch_out = cnn_hidden * (2**i)
            cnn_modules.append(nn.Conv1d(ch_in, ch_out, kernel_size=3, padding=1))
            cnn_modules.append(nn.ReLU())
            ch_in = ch_out
        self.cnn_1d = nn.Sequential(*cnn_modules)
        self.cnn_out_dim = ch_out
        
        # 全局池化
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)  # -> shape [B, cnn_out_dim, 1]
        
        # ---- RNN for history_seq ----
        self.tile_embedding = nn.Embedding(num_tile_types, rnn_embed_dim)
        self.rnn = nn.GRU(
            input_size=rnn_embed_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_out_dim = rnn_hidden_dim * 2  # 双向
        
        # ---- Fusion & Output ----
        # 融合: [cnn_out_dim + rnn_out_dim] -> hidden_dim -> num_tile_types
        fusion_dim = self.cnn_out_dim + self.rnn_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tile_types)
        )
        
    def forward(self, x_tiles, history_seq):
        """
        x_tiles: [B, in_channels, 34]
        history_seq: [B, seq_len] (各值 in [0..num_tile_types-1])
        
        return: logits, shape=[B, num_tile_types]
        """
        B = x_tiles.size(0)
        
        # 1) CNN for x_tiles
        x_cnn = self.cnn_1d(x_tiles)      # -> [B, out_ch, 34]
        x_cnn = self.cnn_pool(x_cnn)      # -> [B, out_ch, 1]
        x_cnn = x_cnn.squeeze(-1)         # -> [B, out_ch]
        
        # 2) RNN for history_seq
        emb = self.tile_embedding(history_seq)  # [B, seq_len, rnn_embed_dim]
        rnn_out, _ = self.rnn(emb)              # [B, seq_len, rnn_hidden_dim*2]
        # 做平均池化(或取最后时刻)
        x_rnn = rnn_out.mean(dim=1)             # -> [B, rnn_out_dim]
        
        # 3) Fusion
        fused = torch.cat([x_cnn, x_rnn], dim=-1)  # [B, fusion_dim]
        logits = self.fusion(fused)               # [B, num_tile_types]
        
        return logits
    
    def predict(self, input_data):
        """
        推理(预测)接口:
        :param input_data: (N,1,10,10) 的torch.Tensor
        :return: softmax后的预测结果 (N,136)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_data)
            probs = F.softmax(outputs, dim=1)
        return probs

    def _prepare_tensors(self, data_list):
        """
        将类似 train_data 或 val_data (list of (input, label, weight)) 转为 X, Y, W 的torch.Tensor
        
        :param data_list: [(np.ndarray(10,10,1)), np.ndarray(136), weight), ...]
        :return: (X, Y, W), 分别是:
                 X => (N,1,10,10) 的浮点Tensor
                 Y => (N,136)      的浮点Tensor (one-hot)
                 W => (N,)         的浮点Tensor
        """
        X_list = []
        Y_list = []
        W_list = []
        for (input_encoded, label_encoded, weight) in data_list:
            X_list.append(input_encoded)  # (10,10,1)
            Y_list.append(label_encoded)  # (136,)
            W_list.append(weight)         # float

        X = torch.tensor(X_list, dtype=torch.float32)  # => (N,10,10,1)
        X = X.permute(0, 3, 1, 2)                      # => (N,1,10,10)
        Y = torch.tensor(Y_list, dtype=torch.float32)  # => (N,136)
        W = torch.tensor(W_list, dtype=torch.float32)  # => (N,)
        return X, Y, W

    def _eval_on_dataset(self, X, Y, W, criterion, batch_size=32):
        """
        在给定数据集(X, Y, W)上评估模型的平均loss (用来做验证集指标或也可做训练集评估)
        :param X: (N,1,10,10)
        :param Y: (N,136)  one-hot
        :param W: (N,)     sample_weight
        :param criterion:  nn.CrossEntropyLoss(reduction='none')
        :return: 平均loss (float)
        """
        self.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[0][i:i+batch_size], X[1][i:i+batch_size]
                batch_Y = Y[i:i+batch_size]
                batch_W = W[i:i+batch_size]

                outputs = self.forward(*batch_X)          # => (batch_size,136)

                loss_per_sample = criterion(outputs, batch_Y)  # => (batch_size,)
                loss_weighted = (loss_per_sample * batch_W).sum()     # 小技巧：先 sum，不取 mean
                total_loss += loss_weighted.item()
                total_samples += len(batch_Y)

        return total_loss / total_samples if total_samples > 0 else 0.0
    
    def train_model(self, 
                    train_data, 
                    val_data,
                    epochs=1, 
                    batch_size=32,
                    early_stopping_patience=3,
                    min_delta=1e-4):
        """
        使用训练集 + 验证集进行训练和早停 (Early Stopping).
        
        :param train_data: 列表，每个元素是 (input_encoded, label_encoded, weight)
                           - input_encoded: 形状 (10,10,1) 的 numpy 数组(或可转Tensor的结构)
                           - label_encoded: 形状 (136,) 的 one-hot 向量
                           - weight: 当前样本的训练权重
        :param val_data:   验证集数据，同上 (也可拆分为别的格式)
        :param epochs:     训练轮数上限
        :param batch_size: mini-batch 大小
        :param early_stopping_patience: 若连续多少个epoch在验证集上无显著改进则停止
        :param min_delta:  判断是否改进的阈值
        """

        TrainDataset = getDatasetClass('discard_model_train_data.pkl')
        train_dataset = TrainDataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        ValidDataset = getDatasetClass('discard_model_valid_data.pkl')
        val_dataset = ValidDataset()
        val_loader = DataLoader(val_dataset, batch_size=len(valid_data), shuffle=False, num_workers=8)

        for (batch_X, batch_Y, batch_W) in val_loader:
            X_val = batch_X
            Y_val = batch_Y
            W_val = batch_W

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # 注意: 使用 reduction='none'，以便后面计算 sample_weight
        criterion = nn.CrossEntropyLoss(reduction='none')

        best_val_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(epochs):

            self.train()
            epoch_loss_sum = 0.0
            sample_count = 0

            max_iter_per_epoch = 100
            iter = 0
            for (batch_X, batch_Y, batch_W) in train_loader:

                optimizer.zero_grad()
                outputs = self.forward(*batch_X)  # => (batch_size,136)

                loss_per_sample = criterion(outputs, batch_Y)  # => (batch_size,)
                loss = (loss_per_sample * batch_W).mean()

                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * batch_size
                sample_count    += batch_size

                iter += 1
                if iter >= max_iter_per_epoch:
                    break

            # 训练集平均loss
            train_avg_loss = epoch_loss_sum / sample_count if sample_count>0 else 0.0

            # ========== 验证阶段 ==========
            val_avg_loss = self._eval_on_dataset(
                X_val, Y_val, W_val, criterion
            )

            print(f"Epoch {epoch+1}/{epochs} => "
                  f"Train Loss={train_avg_loss:.6f}, Val Loss={val_avg_loss:.6f}")

            # ========== 早停判断 ==========
            if best_val_loss - val_avg_loss > min_delta:
                # 有显著改进 => 重置计数
                best_val_loss = val_avg_loss
                no_improvement_count = 0
                torch.save(self, 'best_model_discard.pt')
            else:
                # 没有显著改进
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val_loss={best_val_loss:.6f}")
                break


def split_dataset(games, val_ratio=0.2):
    """
    将actions列表随机打乱，然后按照val_ratio比例拆分成train_data和val_data
    :param actions: 完整数据(list)
    :param val_ratio: 验证集所占比例
    :return: (train_data, val_data)
    """
    # 先随机打乱
    random.shuffle(games)
    # 计算验证集大小
    n_val = int(len(games) * val_ratio)
    # 切片拆分
    val_data = games[:n_val]
    train_data = games[n_val:]
    return train_data, val_data



import torch
from torch.utils.data import Dataset, DataLoader
import random

def getDatasetClass (fn):
    with open(fn, 'rb') as f:
        X, y, W = pickle.load(f)
    class MyDataset(Dataset):
        """
        使用已有的 convertX 函数对 action_samples 做一次性转换，然后
        在 __getitem__ 中返回 (X, Y, W)。

        参数
        ----
        action_samples : list of dict
            原始数据，每个元素包含 'paihe', 'fulu', 'hands', 'discard', 'weight'(可选)等键。
        convertX : function
            已经写好的函数, 接收 (action_samples, max_tile_index, seq_len) 返回 (x_tiles, history_seq, y)。
            注意: 这里不重复写 convertX, 直接调用即可。
        max_tile_index : int
            牌面最大索引, 和 convertX 保持一致。
        seq_len : int
            历史记录序列长度, 和 convertX 保持一致。
        sample_ratio : float
            若 <1.0，则在初始化时只随机保留 sample_ratio * len(action_samples) 个样本。
        """

        def __init__(self):
            super().__init__()
            
            
            # 2) 一次性调用 convertX 做数据预处理
            #    convertX 返回: x_tiles, history_seq, y
            #    这里假设 x_tiles: [B, in_channels, max_tile_index+1]
            #            history_seq: [B, seq_len]
            #            y: [B]
            
            # 4) 存储到 self.xxx，方便 __getitem__ 访问
            self.X = X
            self.y = y
            self.w = W

        def __len__(self):
            return int('inf')

        def __getitem__(self, index):
            index = random.randint(len(W))
            # X 可以是一个元组，也可以拼成dict，看你模型的 forward() 需要什么格式
            X = self.X[0][index], self.X[1][index]
            Y = self.y[index]
            W = self.w[index]
            return X, Y, W
    return MyDataset



if __name__ == '__main__':
    valid_size = 10000
    train_data, valid_data = d[:-valid_size], d[-valid_size:]
    print('data size:', len(train_data), len(valid_data))
    # 2) 初始化模型
    my_model = Model()
    
    # 3) 训练
    #    假设我们想训练2个epoch，每次batch大小32
    my_model.train_model(train_data, valid_data, epochs=10000, batch_size=64)
    