import torch
import multiprocessing
import os
import pickle
from multiprocessing import Pool
import subprocess
import sys
from dask import delayed, compute
from quezha_parser_1 import getData
PARALLEL = 16

def convertX(action_samples, max_tile_index=33, seq_len=32):
    """
    将多个 action_sample 转化为 (x_tiles, history_seq, y) 用于模型输入和训练。
    
    参数:
    -------
    action_samples: List[dict]
        每个元素形如:
        {
        'paihe': {0: [...], 1: [...], 2: [...], 3: [...]},  # 各玩家历史弃牌
        'fulu':  {0: [...], 1: [...], 2: [...], 3: [...]},  # 各玩家副露信息
        'hands': [...],                                     # 当前玩家(0)手牌
        'discard': int,                                     # 当前玩家本步真正打出的牌(标签)
        ... (其他可有可无)
        }
    max_tile_index: int
        牌的最大索引(示例设为 33 表示可能万/条/饼/风箭共34种). 
        如果只测试1..9可设为9.
    seq_len: int
        记录多少步的历史弃牌做时序输入. 若实际多于seq_len, 保留后seq_len个; 不足则补0
    
    返回:
    -------
    x_tiles:  Tensor, shape=[B, 3, max_tile_index+1]
    history_seq: Tensor, shape=[B, seq_len]
    y:        Tensor, shape=[B], 每条记录的弃牌标签 (0..max_tile_index)
    """
    batch_size = len(action_samples)
    
    # in_channels=3: (0)我的手牌计数, (1)我的副露计数, (2)其他人副露计数
    x_tiles = torch.zeros(batch_size, 3, max_tile_index+1, dtype=torch.float32)
    
    # 时序: 所有玩家的弃牌历史(整合后), 再截取或补0到 seq_len
    history_seq = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # 训练标签
    y = torch.zeros(batch_size, dtype=torch.long)
    W = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(action_samples):
        # ========== A) 构建 x_tiles ==========
        
        # 1) 我的手牌 (channel=0)
        my_hand_counts = torch.zeros(max_tile_index+1, dtype=torch.float32)
        for tile in sample['hands']:
            if tile <= max_tile_index:
                my_hand_counts[tile] += 1
        x_tiles[i, 0] = my_hand_counts
        
        # 2) 我的副露 (channel=1)
        my_fulu_counts = torch.zeros(max_tile_index+1, dtype=torch.float32)
        if 0 in sample['fulu']:
            # sample['fulu'][0] 可能是一个 列表, 里面每个元素是(或可能是) tuple/list
            # 如: [(6,5,6)] 或 [ [3,3,3] ] -- 需要自己约定解析
            for meld in sample['fulu'][0]:
                # meld 可能是一个 tuple 或 list, 把它flatten出来
                if isinstance(meld, (list, tuple)):
                    for t in meld:
                        if t <= max_tile_index:
                            my_fulu_counts[t] += 1
                else:
                    # 如果是单个数
                    if meld <= max_tile_index:
                        my_fulu_counts[meld] += 1
        x_tiles[i, 1] = my_fulu_counts
        
        # 3) 其他人副露 (channel=2)
        others_fulu_counts = torch.zeros(max_tile_index+1, dtype=torch.float32)
        for pid in [1, 2, 3]:
            if pid in sample['fulu']:
                for meld in sample['fulu'][pid]:
                    if isinstance(meld, (list, tuple)):
                        for t in meld:
                            if t <= max_tile_index:
                                others_fulu_counts[t] += 1
                    else:
                        if meld <= max_tile_index:
                            others_fulu_counts[meld] += 1
        x_tiles[i, 2] = others_fulu_counts
        
        # ========== B) 构建 history_seq(所有人的弃牌) ==========
        all_discards = []
        for pid in [0, 1, 2, 3]:
            # paihe[pid] 是从早到晚, 直接extend
            discard_list = sample['paihe'].get(pid, [])
            all_discards.extend(discard_list)
        
        # 若总弃牌数>seq_len, 留后 seq_len; 若不足, 前面补0
        # 这里演示 "保留最新" 的 seq_len
        total_len = len(all_discards)
        start_idx = max(0, total_len - seq_len)
        needed = seq_len - (total_len - start_idx)
        
        # 补0(写在 history_seq[i, 0:needed]) 之后再写 discard
        # 先全部填0
        seq_data = [0]*needed + all_discards[start_idx:]
        
        # 截断到 seq_len
        seq_data = seq_data[-seq_len:]
        
        # clamp一下，以防 tile超出 max_tile_index
        seq_data = [min(x, max_tile_index) for x in seq_data]
        
        # 写入
        history_seq[i] = torch.tensor(seq_data, dtype=torch.long)
        
        # ========== C) 训练标签 y = 当前玩家实际打出的牌 ==========
        discard_tile = sample['discard']
        discard_tile = min(discard_tile, max_tile_index)
        y[i] = discard_tile
        W[i] = sample['weight']
    
    return (x_tiles, history_seq), y, W

def process_and_save(task_type, begin, end):
    """
    数据加载、处理和保存函数。由每个子进程调用。
    """
    print(f"Worker processing {task_type} range: {begin}-{end}")
    # 1. 加载数据
    data = getData(begin, end)
    
    # 2. 处理数据
    converted_data = convertX(data, max_tile_index=33, seq_len=32)
    
    # 3. 保存结果
    filename = f'.discard_model/{task_type}_{begin}_{end}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(converted_data, f)
    print(f"Processed and saved: {filename}")

def chunk_data(data, num_chunks):
    """Split data into `num_chunks` approximately equal parts."""
    chunk_size = len(data) // num_chunks
    return [data[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

def load_data_with_index(file_pattern):
    """
    Load data from multiple files matching a given pattern and provide a method to retrieve data using a global index.

    Args:
        file_pattern (str): File pattern to match, e.g., 'discard_model_train_data_*.pkl'.

    Returns:
        dict: A structure containing the loaded data blocks and a retrieval method by global index.
    """
    data_blocks = []
    file_offsets = []
    total_entries = 0

    # Get all matching files
    files = [f for f in os.listdir('.') if f.startswith(file_pattern)]

    # Sort files to ensure consistent order
    files.sort()

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)  # Assume data is a list or iterable of items
            data_blocks.append(data)
            file_offsets.append(total_entries)
            total_entries += len(data)

    def get_data_by_index(global_index):
        """
        Retrieve data using a global index.

        Args:
            global_index (int): The global index of the desired data.

        Returns:
            The data entry corresponding to the global index.
        """
        # Find the block that contains the global index
        for i in range(len(file_offsets)):
            if global_index < file_offsets[i]:
                block_index = i - 1
                break
        else:
            block_index = len(file_offsets) - 1

        # Calculate the local index within the block
        local_index = global_index - file_offsets[block_index]
        return data_blocks[block_index][local_index]

    return {
        'data_blocks': data_blocks,
        'file_offsets': file_offsets,
        'total_entries': total_entries,
        'get_data_by_index': get_data_by_index
    }



def main():
    """
    主进程入口。分配任务范围并启动子进程。
    """
    # 定义数据范围
    train_begin, train_end = 10001, 110000
    valid_begin, valid_end = 110001, 114466

    # 子进程任务列表
    train_step = (train_end - train_begin) // PARALLEL
    valid_step = (valid_end - valid_begin) // PARALLEL

    tasks = []

    # 添加 train 的任务
    for i in range(PARALLEL):
        begin = train_begin + i * train_step
        end = train_begin + (i + 1) * train_step if i < PARALLEL - 1 else train_end
        tasks.append(('train', begin, end))

    # 添加 valid 的任务
    for i in range(PARALLEL):
        begin = valid_begin + i * valid_step
        end = valid_begin + (i + 1) * valid_step if i < PARALLEL - 1 else valid_end
        tasks.append(('valid', begin, end))

    # 使用 multiprocessing 启动进程
    processes = []
    for task_type, begin, end in tasks:
        p = multiprocessing.Process(target=process_and_save, args=(task_type, begin, end))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All tasks completed.")

if __name__ == "__main__":
    main()  # 主进程入口