from common import *
import os
from record import read_record
from quezha_parser import get_action
from tqdm import tqdm
from quezha_parser import get_action
import pickle

# 更新手牌编码函数
def encode_hand(hands):
    hand_array = np.zeros(34)
    for tile in hands:
        index = (tile - 1) % 34  # 将牌值转换为0-33之间的索引
        hand_array[index] += 1
    return hand_array

# 更新副露编码函数
def encode_melds(fulu):
    meld_array = np.zeros(4, dtype=int)  # 最多4个副露
    for i, meld in enumerate(fulu):
        if meld:  # 检查副露是否存在
            base_index = (meld[0] - 1) % 34
            meld_array[i] = base_index * 3 + meld[1]  # meld[1] 表示排列形式：顺刻杠
    return meld_array

# 数据解析函数
def parse_action_list(action_list):
    states = []
    actions = []
    for action in action_list:
        player = action['player']
        fulu = action['fulu'][player]
        hands = action['hands']
        discard = action['discard']

        # 编码手牌和副露
        hand_array = encode_hand(hands)
        meld_array = encode_melds(fulu)

        # 组合手牌和副露编码
        state = (hand_array, meld_array)
        states.append(state)
        actions.append((discard - 1) % 34)  # 将牌编号转为索引

    return states, actions

# 数据准备函数
def prepare_data(action_list):
    states, actions = parse_action_list(action_list)
    hand_inputs = []
    meld_inputs = []
    targets = []

    for state, action in zip(states, actions):
        hand_array, meld_array = state
        hand_inputs.append(hand_array)
        meld_inputs.append(meld_array)
        targets.append(action)

    hand_inputs = torch.tensor(hand_inputs, dtype=torch.float32)
    meld_inputs = torch.tensor(meld_inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return meld_inputs, hand_inputs, targets

# 保存数据到文件
def save_data(data, id):
    with open(f'data_checkpoint_{id}.pkl', 'wb') as f:
        pickle.dump(data, f)

# 加载上次保存的进度
def load_checkpoint():
    checkpoints = [f for f in os.listdir() if f.startswith('data_checkpoint_') and f.endswith('.pkl')]
    if not checkpoints:
        return [], id_start
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    with open(latest_checkpoint, 'rb') as f:
        data = pickle.load(f)
    last_id = int(latest_checkpoint.split('_')[2].split('.')[0])
    return data, last_id + 1

if __name__ == '__main__':
    # 尝试加载上次的进度
    data, id = load_checkpoint()
    print(id)
    action_list = []
    checkpoint_interval = 5000

    with tqdm(total=id_end - id_start, initial=id - id_start) as pbar:
        while id <= id_end:
            while True: #  as a block
                r = get_action(id)
                if r is None:
                    break
                d = prepare_data(r)
                data.append(d)
                if len(data) % checkpoint_interval == 0:
                    save_data(data, id)
                    data = []  # 清空已保存的数据
                break
            id += 1
            pbar.update(1)
        
        # 保存最后剩余的数据
        if data:
            save_data(data, id)
