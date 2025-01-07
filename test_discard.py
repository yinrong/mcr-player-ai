import random
import torch
from train_discard import Model  # 替换为你的模型定义文件路径

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

    # 修复: 确保 hand 是整数索引并正确打印
    print("初始手牌:", [tile_index_to_str[t] for t in hand])

    for i in range(num_rounds):
        if not deck:
            print("牌堆已空，结束游戏！")
            break

        # 摸牌
        new_tile = deck.pop(0)
        hand.append(new_tile)
        print(f"\n第 {i + 1} 轮摸牌: {tile_index_to_str[new_tile]}")
        print("摸牌后手牌:", [tile_index_to_str[t] for t in hand])

        # 转换输入
        x_tiles = torch.zeros(1, 3, 34, dtype=torch.float32)  # 假设 0:手牌, 1:自己的副露, 2:他人副露
        for tile in hand:
            x_tiles[0, 0, tile] += 1

        history_seq = torch.zeros(1, seq_len, dtype=torch.long)
        if history:  # 只有在 history 不为空时，才进行赋值
            history_indices = history[-seq_len:]
            history_seq[0, -len(history_indices):] = torch.tensor(history_indices)

        # 模型预测弃牌
        with torch.no_grad():
            logits = model(x_tiles, history_seq)
        suggested_discard = logits.argmax(dim=-1).item()
        print(f"建议弃牌: {tile_index_to_str[suggested_discard]}")

        # 执行弃牌
        hand.remove(suggested_discard)
        history.append(suggested_discard)
        print("弃牌后手牌:", [tile_index_to_str[t] for t in hand])
        print("弃牌历史:", [tile_index_to_str[t] for t in history])

    print("\n游戏结束！最终手牌:", [tile_index_to_str[t] for t in hand])
    print("最终弃牌历史:", [tile_index_to_str[t] for t in history])

# 调用模拟游戏
if __name__ == "__main__":
    model_path = "best_model_discard.pt"  # 替换为实际模型路径
    simulate_game(model_path)
