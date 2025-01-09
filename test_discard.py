import random
import torch
from common import debugTensor
from discard_model_data import convertX
from mjutil import renderSimple
from discard_model import DiscardModel  # 替换为你的模型定义文件路径

# 假设牌的索引到可读名称的映射
tile_index_to_str = {
    0: "万1", 1: "万2", 2: "万3", 3: "万4", 4: "万5",
    5: "万6", 6: "万7", 7: "万8", 8: "万9",
    9: "条1", 10: "条2", 11: "条3", 12: "条4", 13: "条5",
    14: "条6", 15: "条7", 16: "条8", 17: "条9",
    18: "饼1", 19: "饼2", 20: "饼3", 21: "饼4", 22: "饼5",
    23: "饼6", 24: "饼7", 25: "饼8", 26: "饼9",
    27: "东", 28: "南", 29: "西", 30: "北", 31: "中", 32: "发", 33: "白"
}
tile_str_to_index = {v: k for k, v in tile_index_to_str.items()}

# 模拟摸牌-弃牌的过程
def simulate_game(model_path, seq_len=32, num_rounds=30):
    # 加载模型
    model = torch.load(model_path)
    model.eval()

    # 初始化手牌与牌堆
    deck = list(tile_index_to_str.keys()) * 4  # 每种牌4张
    random.shuffle(deck)  # 随机洗牌
    hand = deck[:13]  # 初始手牌
    deck = deck[13:]  # 剩余牌堆
    history = []  # 弃牌历史

    renderSimple(0, hand, None)

    for i in range(num_rounds):
        if not deck:
            print("牌山已空")
            break

        # 摸牌
        new_tile = deck.pop(0)
        hand = sorted(hand)
        hand.append(new_tile)

        # 准备输入样本
        action_sample = {
            'paihe': {player: history for player in range(4)},  # 所有玩家的弃牌历史
            'fulu': {player: [] for player in range(4)},  # 假设没有副露
            'hands': hand,
            'weight': 1,
            'discard': 1,
        }

        #debugTensor(hand, 'hand:')
        x,y,w = convertX([action_sample])
        with torch.no_grad():
            logits = model(x)
        suggested_discard = logits.argmax(dim=-1).item()
        #debugTensor(suggested_discard, 'argmax:')

        renderSimple(i + 1, hand, suggested_discard)

        history.append(suggested_discard)


# 调用模拟游戏
if __name__ == "__main__":
    model_path = "discard_best.pt"  # 替换为实际模型路径
    torch.set_printoptions(precision=4, sci_mode=False)
    simulate_game(model_path)
