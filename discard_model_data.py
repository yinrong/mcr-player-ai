import glob
import os
import torch
import multiprocessing
import pickle
from quezha_parser_1 import TILE_TYPE_NUM, getData
PARALLEL = 16
def convertX(action_samples):
    """
    将多个 action_sample 转化为 (x, y, w) 用于模型输入和训练。
    
    参数:
    -------
    action_samples: List[dict]
        每个元素形如:
        {
        'paihe': {0: [...], 1: [...], 2: [...], 3: [...]},  # 各玩家历史弃牌
        'fulu':  {0: [...], 1: [...], 2: [...], 3: [...]},  # 各玩家副露信息
        'hands': [...],                                     # 当前玩家(0)手牌
        'discard': int,                                     # 当前玩家本步真正打出的牌(标签)
        'weight': int,                                      # 权重
        ... (其他可有可无)
        }
    seq_len: int
        时序弃牌历史的衰减系数长度.

    返回:
    -------
    x: Tensor, shape=[B, N, 4, 9]
        N 是特征维度，每一项表示主观视角的不同信息。
    y: Tensor, shape=[B]
    w: Tensor, shape=[B]
        每条记录的权重。
    """
    batch_size = len(action_samples)
    N = 10  # 特征数量
    num_rows = 4
    num_cols = 9  # 最大长度（字牌行有效长度为 7，后面补 0）

    # 初始化 x, y, w
    x = torch.zeros(batch_size, N, num_rows, num_cols, dtype=torch.float32)
    y = torch.zeros(batch_size, dtype=torch.long)
    w = torch.zeros(batch_size, dtype=torch.float32)

    for i, sample in enumerate(action_samples):
        skip = False
        hands = sample['hands']
        for tile in hands:
            if tile >= TILE_TYPE_NUM: # 花牌
                skip = True
                break
        if skip: continue

        # 提取手牌信息
        my_hand_counts = [0] * TILE_TYPE_NUM
        for tile in hands:
            if tile >= TILE_TYPE_NUM: continue # 花牌
            my_hand_counts[tile] += 1
        
        # 提取副露信息
        fulu_counts = [[0] * TILE_TYPE_NUM for _ in range(4)]  # 0:自己, 1:下家, 2:对家, 3:上家
        for pid, fulu in sample['fulu'].items():
            for meld in fulu:
                if isinstance(meld, (list, tuple)):
                    for tile in meld:
                        fulu_counts[pid][tile] += 1
                else:
                    fulu_counts[pid][meld] += 1
        
        # 计算剩余牌
        remaining_counts = [4] * TILE_TYPE_NUM
        for j in range(TILE_TYPE_NUM):
            remaining_counts[j] -= my_hand_counts[j]
            remaining_counts[j] -= sum(fulu_counts[pid][j] for pid in range(4))
        
        # 时序编码弃牌历史
        decay_factor = 0.8  # 每一步衰减系数
        discard_history = [[0] * TILE_TYPE_NUM for _ in range(4)]  # 0:自己, 1:下家, 2:对家, 3:上家
        for pid, discards in sample['paihe'].items():
            weight = 1.0
            for tile in reversed(discards):
                if tile >= TILE_TYPE_NUM: continue # 花牌
                discard_history[pid][tile] += weight
                weight *= decay_factor

        # 填入 x
        x[i, 0] = encode_to_4x9(my_hand_counts)           # 自己的手牌
        x[i, 1] = encode_to_4x9(my_hand_counts, fulu_counts[0])  # 自己的, 手牌+副露
        x[i, 2] = encode_to_4x9(remaining_counts)         # 剩余牌
        x[i, 3] = encode_to_4x9(fulu_counts[0])           # 自己的副露
        x[i, 4] = encode_to_4x9(fulu_counts[1])           # 下家的副露
        x[i, 5] = encode_to_4x9(fulu_counts[2])           # 对家的副露
        x[i, 6] = encode_to_4x9(fulu_counts[3])           # 上家的副露
        x[i, 7] = encode_to_4x9(discard_history[0])       # 自己的弃牌
        x[i, 8] = encode_to_4x9(discard_history[1])       # 下家的弃牌
        x[i, 9] = encode_to_4x9(discard_history[2])       # 对家的弃牌

        y[i] = sample['discard']
        w[i] = sample.get('weight', 1.0)  # 默认权重为 1.0

    return x, y, w


def encode_to_4x9(counts, additional_counts=None):
    """
    将 counts 转换为 4x9 格式（万/条/饼/字）。
    如果 additional_counts 不为 None，则将其累加。
    """
    rows = [counts[0:9], counts[9:18], counts[18:27], counts[27:34]]  # 补 0 到 9
    if additional_counts:
        rows = [
            [a + b for a, b in zip(row, additional_counts[start_idx:start_idx + len(row)])]
            for row, start_idx in zip(rows, [0, 9, 18, 27])
        ]
    rows[3] += [0, 0]
    return torch.tensor(rows, dtype=torch.float32)



def process_and_save(save_path, task_type, begin, end):
    """
    数据加载、处理和保存函数。由每个子进程调用。
    """
    print(f"Worker processing {task_type} range: {begin}-{end}")
    # 1. 加载数据
    data = getData(begin, end)
    
    # 2. 处理数据
    converted_data = convertX(data)
    
    # 3. 保存结果
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    filename = f'{save_path}/{task_type}_{begin}_{end}.pkl'
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
    files = [f for f in glob.glob(file_pattern)]

    # Sort files to ensure consistent order
    files.sort()

    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)  # Assume data is a list or iterable of items
            data_blocks.append(data)
            file_offsets.append(total_entries)
            total_entries += len(data[2])

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
        i = global_index - file_offsets[block_index]
        d = data_blocks[block_index]
        X = d[0][0][i], d[0][1][i]
        Y = d[1][i]
        W = d[2][i]
        return X, Y, W

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
    save_path = '.discard_model'

    # 子进程任务列表
    train_step = (train_end - train_begin) // PARALLEL
    valid_step = (valid_end - valid_begin) // PARALLEL

    if DEBUG:
        save_path = '.test_save'
        process_and_save(save_path, 'train', 110000, 114466)
        process_and_save(save_path, 'valid', 10001, 10500)
        process_and_save(save_path, 'train', 114000, 114466)
    else:
        tasks = []

        # 添加 train 的任务
        for i in range(PARALLEL):
            begin = train_begin + i * train_step
            end = train_begin + (i + 1) * train_step if i < PARALLEL - 1 else train_end
            tasks.append((save_path, 'train', begin, end))

        # 添加 valid 的任务
        for i in range(PARALLEL):
            begin = valid_begin + i * valid_step
            end = valid_begin + (i + 1) * valid_step if i < PARALLEL - 1 else valid_end
            tasks.append((save_path, 'valid', begin, end))

        with multiprocessing.Pool(processes=PARALLEL) as pool:
            pool.starmap(process_and_save, tasks)
        print("All tasks completed.")

if __name__ == "__main__":
    DEBUG=True
    #DEBUG=False
    main()  # 主进程入口