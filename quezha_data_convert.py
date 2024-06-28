import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from record import read_record
from quezha_parser import get_action

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

    return hand_inputs, meld_inputs, targets

if __name__ == '__main__':
    # 使用示例
    action_list = [
        {
            'action_type': 'discard',
            'player': 0,
            'paihe': {0: [], 1: [], 2: [], 3: []},
            'fulu': {0: [], 1: [], 2: [], 3: []},
            'is_hand_discard': {0: [True, True, True, True, True, True, True, False, False], 1: [True, False, False, True, False, False, False, True, False], 2: [True, True, True, True, True, False, False], 3: [True, True, True, True, True, True, False, False]},
            'hands': [66, 26, 86, 120, 20, 132, 72, 12, 110, 8, 79, 111, 124, 28],
            'discard': 120
        },
        # 添加更多的动作数据
    ]
    action_lists = get_action(id=21415)
    hand_inputs, meld_inputs, targets = prepare_data(action_lists)
    print(hand_inputs)
    print(meld_inputs)
    print(targets)
