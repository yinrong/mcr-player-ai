from common import *
from quezha_parser_1 import getAllGames, rankPlayers

import torch
import torch.nn as nn
import torch.optim as optim


def convertTrainData(rank_info, games):
    """
    将原始动作数据(含paihe, fulu, hands, discard等)转换为可训练的输入、标签、权重。
    
    :param actions: list or iterable of dict，每个dict包含与训练相关的字段，例如：
        {
            'player_name': 'xxxx',
            'player_id': 111111,
            "action_type": "discard",
            "player": 1,
            "paihe": { "0": [120, 124, 132, ...], "1": [...], ... },
            "fulu": { "0": [(83, 72, 79)], ... },
            "is_hand_discard": { "0": [true, ...], ... },
            "hands": [51, 2, 48, ...],
            "discard": 59,
            "rank_info": {
                '10014': {'win_rate': 0.7777, 'rank': 1}, 
                '10056': {'win_rate': 0.2222, 'rank': 2}, 
                ...
            }
        }
    :return: list of tuples: [ (input_encoded, label_encoded, weight), ... ]
             或根据需求也可以拆分成 X, y, sample_weights
    """
    data_for_training = []
    for game in games:
        for action in game['actions']:
            # 1) 获取当前玩家的 rank_info
            #    注意：示例中 player_id 形如 111111，需要和 rank_info 中的 key(形如 '10014')对应。
            #    本示例假设action中包含rank_info，且player_id可以在rank_info字典里找到（真实情况需自己映射）。
            player_id_str = str(action['player_id'])
            rank_info = action.get('rank_info', {})
            player_rank = rank_info.get(player_id_str, {}).get('rank', 4)  # 缺省rank=4

            # 2) 计算权重：排名越靠前，权重越高(此处仅示例)
            #    下面的例子假定 rank=1 => weight=1.0, rank=2 => weight=0.75, rank=3 => weight=0.5, rank=4 => weight=0.25
            weight = (5 - player_rank) * 0.25
            
            # 3) 编码输入：将paihe、fulu、hands进行整合
            paihe = action.get('paihe', {})
            fulu = action.get('fulu', {})
            hands = action.get('hands', [])
            
            # 3.1) 外部函数，将paihe, fulu, hands转成统一格式(二维/三维张量等)
            #      比如可以对每一张牌调用getTileTypeId(...)再汇总编码
            #      下面仅作示例，使用假函数 encodeTableState(...)
            # 外部函数，未来扩展
            input_encoded = encodeTableState(paihe, fulu, hands)
            
            # 4) 编码标签：discard那张牌
            discard_tile = action.get('discard', None)
            # 外部函数，未来扩展
            label_encoded = encodeDiscardTile(discard_tile)
            
            # 5) 将(input, label, weight)打包
            data_for_training.append((input_encoded, label_encoded, weight))
    
    return data_for_training


# 如果已经存在getTileTypeId(tile_id)这样可复用的外部函数，可在此直接调用；
# 这里示例一个本地简易版本进行演示。
def _getSuitAndRank(tile_id):
    """
    临时示例函数: 将0~135的TileId转换为(suit, rank)。
    suit 范围: 0~(理论最多)；rank 范围: 0~(理论最多)
    具体分配规则、花色/牌数等可根据实际麻将规则调整。
    """
    tile_type = tile_id // 4  # tile_type范围 0~33，共34种牌
    suit = tile_type // 9     # suit范围示例: 0~3+ (风/箭要自己划分)
    rank = tile_type % 9      # rank范围示例: 0~8（适用于万/条/饼9张）
    return suit, rank


def encodeTableState(paihe, fulu, hands):
    """
    将当前牌局信息(paihe、fulu、hands等)编码为一个固定形状的张量(示例: (10, 10, 1)).
    
    :param paihe: dict, e.g. { "0":[tileId, ...], "1":[...], ... } ，各玩家弃牌
    :param fulu: dict, e.g. { "0":[(tileId1, tileId2, tileId3), ...], ... } ，副露(吃/碰/杠)
    :param hands: list, e.g. [tileId, tileId, ...] ，自己的手牌
    :return: np.ndarray, shape=(10, 10, 1) 的编码结果(示例)
    """
    # 这里只是一个示例，实际可根据麻将牌数量及CNN需求设计更合适的维度。
    # 例如 (5, 9, 4) 或 (34, 4) 或 (10,10) 等。
    # 这里做一个10x10的二维数组，再扩维至(10,10,1).
    
    state = np.zeros((10, 10), dtype=np.float32)
    
    # 1) 编码自己的手牌
    for tile_id in hands:
        suit, rank = _getSuitAndRank(tile_id)
        if suit < 10 and rank < 10:
            state[suit, rank] += 1.0  # 自己手牌可计为+1
    
    # 2) 编码所有玩家的弃牌(paihe)
    #    示例：把每张弃牌记为+2，用于和手牌做区分
    for player_idx, discard_list in paihe.items():
        for tile_id in discard_list:
            suit, rank = _getSuitAndRank(tile_id)
            if suit < 10 and rank < 10:
                state[suit, rank] += 2.0
    
    # 3) 编码副露(fulu)，示例记为+1.5
    #    每个元素可能是一个tuple，例如 (83, 72, 79) 代表吃/碰/杠的三张牌
    for player_idx, fulu_combos in fulu.items():
        for combo in fulu_combos:
            for tile_id in combo:
                suit, rank = _getSuitAndRank(tile_id)
                if suit < 10 and rank < 10:
                    state[suit, rank] += 1.5
    
    # 扩展到(10,10,1)
    return state.reshape((10, 10, 1))


def encodeDiscardTile(tile_id):
    """
    将需要丢弃的那张牌编码为模型可训练/预测的标签格式。
    示例：使用 one-hot 编码(长度136)，对应所有TileId范围 [0, 135]。
    
    :param tile_id: int or None，需要丢弃的牌ID (0~135)
    :return: np.ndarray, shape=(136,) 的one-hot向量
    """
    one_hot = np.zeros(136, dtype=np.float32)
    if tile_id is not None and 0 <= tile_id < 136:
        one_hot[tile_id] = 1.0
    return one_hot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)  # => (N,32,8,8)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # => (N,64,6,6)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, 136)  # 对应牌ID(0~135)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # => (N,32,8,8)
        x = F.relu(self.conv2(x))   # => (N,64,6,6)

        # 如果遇到非连续报错，可先 x = x.contiguous()
        x = x.reshape(x.shape[0], -1)  # flatten => (N,2304)

        x = F.relu(self.fc1(x))     # => (N,128)
        x = self.fc2(x)             # => (N,136)
        return x
    
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
                batch_X = X[i:i+batch_size]
                batch_Y = Y[i:i+batch_size]
                batch_W = W[i:i+batch_size]

                outputs = self.forward(batch_X)          # => (batch_size,136)
                batch_Y_class = torch.argmax(batch_Y, 1) # => (batch_size,)

                loss_per_sample = criterion(outputs, batch_Y_class)  # => (batch_size,)
                loss_weighted = (loss_per_sample * batch_W).sum()     # 小技巧：先 sum，不取 mean
                total_loss += loss_weighted.item()
                total_samples += batch_X.size(0)

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

        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # 注意: 使用 reduction='none'，以便后面计算 sample_weight
        criterion = nn.CrossEntropyLoss(reduction='none')

        # 1) 准备训练集 Tensor
        X_train, Y_train, W_train = self._prepare_tensors(train_data)
        # 2) 准备验证集 Tensor
        X_val, Y_val, W_val = self._prepare_tensors(val_data)

        best_val_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(epochs):
            # ========== 训练阶段 ========== 
            # 打乱训练集
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            W_train = W_train[indices]

            self.train()
            epoch_loss_sum = 0.0
            sample_count = 0

            # mini-batch
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_Y = Y_train[i:i+batch_size]
                batch_W = W_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.forward(batch_X)  # => (batch_size,136)

                # batch_Y 是 one-hot => 转为 class index
                batch_Y_class = torch.argmax(batch_Y, dim=1)  # => (batch_size,)

                loss_per_sample = criterion(outputs, batch_Y_class)  # => (batch_size,)
                loss = (loss_per_sample * batch_W).mean()

                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * batch_X.size(0)
                sample_count    += batch_X.size(0)

            # 训练集平均loss
            train_avg_loss = epoch_loss_sum / sample_count if sample_count>0 else 0.0

            # ========== 验证阶段 ==========
            val_avg_loss = self._eval_on_dataset(
                X_val, Y_val, W_val, criterion, batch_size=batch_size
            )

            print(f"Epoch {epoch+1}/{epochs} => "
                  f"Train Loss={train_avg_loss:.6f}, Val Loss={val_avg_loss:.6f}")

            # ========== 早停判断 ==========
            if best_val_loss - val_avg_loss > min_delta:
                # 有显著改进 => 重置计数
                best_val_loss = val_avg_loss
                no_improvement_count = 0
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



if __name__ == '__main__':
    #games = getAllGames()
    #games = getAllGames(to=10010)
    games = getAllGames()
    train_games, test_games = split_dataset(games)
    rank_info = rankPlayers(games)

    # 1) 转换训练数据
    train_data = convertTrainData(rank_info, train_games)
    valid_data = convertTrainData(rank_info, test_games)
    
    # 2) 初始化模型
    my_model = Model()
    
    # 3) 训练
    #    假设我们想训练2个epoch，每次batch大小32
    my_model.train_model(train_data, valid_data, epochs=10000, batch_size=32)
    
    # （根据需要，训练结束后可保存模型）
    # my_model.model.save('xxx.h5')