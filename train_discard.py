import glob
import os
import pickle
import random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from discard_model import DiscardModel
from common import *
from data_server import MyLocalDataset

class Validator:
    def __init__(self, batch_size=5000):
        """
        验证器类，用于一次性加载所有验证集数据并快速进行评估。
        验证集文件路径模式是固定的：".discard_model/valid_*.pkl"
        Args:
            batch_size (int): 用于评估的批量大小。
        """
        self.file_pattern = ".discard_model/valid_*.pkl"
        self.batch_size = batch_size
        self.batched_data = self._load_and_batch_data()

    def _load_all_data(self):
        """
        加载所有验证数据到内存中。
        Returns:
            tuple: 包括输入 (X)、标签 (Y)、权重 (W) 的张量。
        """
        all_X, all_Y, all_W = [], [], []
        files = sorted(glob.glob(self.file_pattern))
        for file in files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                all_X.extend(data[0].unsqueeze(0))
                all_Y.extend(data[1])
                all_W.extend(data[2])
        all_Y = all_Y[:30000]
        all_W = all_W[:30000]
        all_X = torch.cat(all_X, dim=0)  # 假设每个数据的第0维是样本
        all_X = all_X[:30000]
        all_Y = torch.tensor(all_Y)
        all_W = torch.tensor(all_W)
        return all_X, all_Y, all_W

    def _load_and_batch_data(self):
        """
        加载并预先分批验证数据。
        Returns:
            list: 预分批的数据，每个批次是 (batch_X, batch_Y, batch_W)。
        """
        X, Y, W = self._load_all_data()
        batched_data = []
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i:i + self.batch_size]
            batch_Y = Y[i:i + self.batch_size]
            batch_W = W[i:i + self.batch_size]
            batched_data.append((batch_X, batch_Y, batch_W))
        return batched_data

    def evaluate(self, model, criterion):
        """
        在验证数据上评估模型的平均损失和准确率。
        Args:
            model: 需要评估的模型。
            criterion: nn.CrossEntropyLoss(reduction='none')
        Returns:
            tuple: 平均损失 (float), 准确率 (float)
        """
        model.eval()
        total_loss = 0.0
        total_weight = 0
        total_sample = 0
        total_correct = 0

        with torch.no_grad():
            for batch_X, batch_Y, batch_W in self.batched_data:
                outputs = model(batch_X)
                loss_per_sample = criterion(outputs, batch_Y)  # => (batch_size,)
                normalized_batch_W = batch_W / batch_W.sum()  # 权重归一化
                loss = (loss_per_sample * normalized_batch_W).sum()

                total_loss += loss.item() * batch_W.sum().item()  # 恢复到总损失的数值量级
                total_weight += batch_W.sum().item()  # 样本总权重累积

                preds = outputs.argmax(dim=-1)  # 取每行最大值的索引，即预测的类别
                total_correct += (preds == batch_Y).sum().item()
                total_sample += batch_X.size(0)

        avg_loss = total_loss / total_weight if total_weight > 0 else 0.0
        accuracy = total_correct / total_sample if total_sample > 0 else 0.0
        return avg_loss, accuracy

def train_model(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if world_size > 1:
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    """
    使用训练集 + 验证集进行训练和早停 (Early Stopping).
    
    :param train_data: 列表，每个元素是 (input_encoded, label_encoded, weight)
                        - input_encoded: 形状 (10,10,1) 的 numpy 数组(或可转Tensor的结构)
                        - label_encoded: 形状 (136,) 的 one-hot 向量
                        - weight: 当前样本的训练权重
    :param val_data:   验证集数据，同上 (也可拆分为别的格式)
    :param batch_size: mini-batch 大小
    :param early_stopping_patience: 若连续多少个epoch在验证集上无显著改进则停止
    """
    model = DiscardModel()
    if world_size > 1:
        model = DDP(model)

    train_dataset = MyLocalDataset('train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    print('load train dataset')

    validator = Validator()
    print('load valid dataset')

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.004,
        weight_decay=1e-6,
    )
    # 注意: 使用 reduction='none'，以便后面计算 sample_weight
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val_corr = 0.20
    no_improvement_count = 0
    for epoch in range(99999):

        model.train()
        epoch_loss_sum = 0.0
        sample_count = 0

        max_n_trained = 240
        n_trained = 0
        for (batch_X, batch_Y, batch_W) in train_loader:

            optimizer.zero_grad()
            outputs = model.forward(batch_X)  # => (batch_size,136)

            loss_per_sample = criterion(outputs, batch_Y)  # => (batch_size,)
            normalized_batch_W = batch_W / batch_W.sum()  # 权重归一化
            loss = (loss_per_sample * normalized_batch_W).sum()

            loss.backward()
            #print(f'train_loss={loss:.4f}')

            if world_size > 1:
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

            optimizer.step()

            epoch_loss_sum += loss.item() * batch_W.sum().item()  # 恢复到总损失的数值量级
            sample_count += batch_W.sum().item()  # 样本总权重累积

            n_trained += len(batch_W)
            if epoch ==0 or n_trained >= max_n_trained:
                break

        # 训练集平均loss
        train_avg_loss = epoch_loss_sum / sample_count if sample_count>0 else 0.0

        # ========== 早停判断 ==========
        save = 0
        if rank == 0:
            # ========== 验证阶段 ==========
            val_avg_loss, val_corr = validator.evaluate(model, criterion)
            if val_corr > best_val_corr:
                best_val_corr = val_corr
                no_improvement_count = 0
                if val_corr > 0.3:
                    torch.save(model, 'best_model_discard.pt')
                    save = 1
            else:
                no_improvement_count += 1

            print(f"epoch={epoch}, train_loss={train_avg_loss:.2f}, val_loss={val_avg_loss:.2f}, val_corr={val_corr:.4f}, no_imp={no_improvement_count}, save={save}")

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. Best val_corr={best_val_corr:.6f}")
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
from torch.utils.data import DataLoader
import random

world_size = 1
batch_size=8
early_stopping_patience=20

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"

    torch.set_printoptions(precision=4, sci_mode=False)
    train_model(0, world_size)
    #mp.spawn(train_model, args=(world_size), nprocs=world_size, join=True)
    #mp.spawn(train_model, args=(world_size), nprocs=world_size, join=True)
    #mp.spawn(train_model, args=(world_size), nprocs=world_size, join=True)
    #mp.spawn(train_model, args=(world_size), nprocs=world_size, join=True)