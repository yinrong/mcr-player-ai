from common import *
from model import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from ai.env import SimplifiedMahjongEnv, num_actions
import mj
import os
import pickle
learning_rate = 0.04
input_shape = num_actions
hidden_size = 256
dropout_prob = 0.1
num_heads = 8
num_layers = 3
estimate_win_tile = 18
fan_keep = 5 * estimate_win_tile
fan_save = 20 * estimate_win_tile

if __name__ == '__main__':
    set_seed(1)



    cnn_model = CNNQNetwork(input_shape, num_actions, hidden_size, dropout_prob)
    transformer_model = TransformerQNetwork(input_shape, num_actions, hidden_size, num_heads, num_layers, dropout_prob)
    fc_model = FCQNetwork(input_shape, num_actions, hidden_size)

    model = FCQNetwork(input_shape, num_actions, hidden_size)
    target_model = FCQNetwork(input_shape, num_actions, hidden_size)
    if os.path.exists("model.pth"):
        model = torch.load("model.pth")
        target_model = torch.load("model.pth")
    else:
        target_model.load_state_dict(model.state_dict())

    #model = EnsembleQNetwork(cnn_model, transformer_model, fc_model, num_actions)
    #target_model = EnsembleQNetwork(cnn_model, transformer_model, fc_model, num_actions)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_function = nn.MSELoss()

    if os.path.exists("buffer_history.pkl"):
        with open("buffer_history.pkl", "rb") as f:
            fan_buffers = pickle.load(f)
    else:
        fan_buffers = {}
    batch_size = 64
    gamma = 0.99
    explore = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1 - 1e-3

    def train_step():
        if len(buffer) < batch_size:
            return

        minibatch = random.sample(buffer, batch_size)
        states, discard_tiles, rewards, next_states, dones, _ = zip(*minibatch)

        states = torch.stack(states)
        discard_tiles = torch.tensor(discard_tiles, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 计算当前状态的 Q 值
        q_values = model(states)
        # 提取实际执行动作的 Q 值
        q_values = q_values.gather(1, discard_tiles.unsqueeze(1)).squeeze(1)

        # 计算下一个状态的 Q 值
        next_q_values = target_model(next_states)

        max_next_q_values = next_q_values.max(1)[0]

        # 计算目标 Q 值：使用下一个状态的所有 Q 值的最大值（除弃掉的牌外的总和）
        # max_next_q_values = next_q_values.sum(1) - next_q_values.gather(1, discard_tiles.unsqueeze(1)).squeeze(1)

        # 如果下一个状态是终止状态，目标 Q 值只等于即时奖励
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        # 计算损失
        loss = loss_function(q_values, target_q_values.detach())

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global explore
        if explore > epsilon_min:
            explore *= epsilon_decay
    
    env = SimplifiedMahjongEnv(model)
    last_hist_fan = 0
    max_len = 3000
    sum_step = 0
    sum_env = 0
    for episode in range(20000):
        state = env.reset()
        total_reward = 0

        while not env.done:
            if np.random.rand() < explore:
                #env.step(discard_by=env.discard_rand)
                env.step(discard_by=env.discard_strategy1)
                #env.step(discard_by=env.discard_by_model_reverse)
            else:
                env.step()
        sum_step += env.step_count
        sum_env += 1

        # Add fan to fan_buffers
        if len(env.this_fans) == 0:
            current_fans = ['F']
        else:
            current_fans = env.this_fans
        for current_fan in current_fans:
            if current_fan not in fan_buffers:
                fan_buffers[current_fan] = deque(maxlen=fan_save)
            if len(fan_buffers[current_fan]) > fan_save:
                fan_buffers[current_fan] = deque(fan_buffers[current_fan], maxlen=fan_save)
            fan_buffers[current_fan].extend(env.buffer)
        
        # Ensure diverse training set
        buffer = []
        for fan, buf in fan_buffers.items():
            if len(buf) == 0: continue
            if len(buf) > fan_keep:
                buf = random.sample(buf, fan_keep)
            buffer.extend(buf)

        for n in range(50):
            train_step()
            s = (seq(fan_buffers.items())
                .map(lambda e: f"{e[0]}.{len(e[1])//estimate_win_tile}")
                .make_string('_')
            )

        print(f"event=found, episode={episode}, explore={explore:.2f}, step={env.step_count}, reward={env.buffer[-1][2]:.2f}"
                +f" buffer={len(buffer)}, found=sum.{len(env.hist_fan)}_{s}")
        if len(env.hist_fan) != last_hist_fan or episode % 50 == 49:
            reward_sum = 0
            reward_count = 0
            for fan, buf in fan_buffers.items():
                if len(buf) == 0: continue
                reward_sum += seq(buf).map(lambda e: e[-1]).sum()
                reward_count += len(buf)
            reward_avg = reward_sum / reward_count
            for fan, buf in fan_buffers.items():
                buf = (seq(buf)
                    .filter(lambda e: e[-1] > reward_avg)
                    .to_list()
                )
            print(f"event=summary, episode={episode}, explore={explore:.2f}, step={env.step_count}, reward={env.buffer[-1][2]:.2f}/{reward_avg:.1f}"
                    +f" buffer={len(buffer)}, found=sum.{len(env.hist_fan)}_{s}")
            sum_step = 0
            sum_env = 0
            last_hist_fan = len(env.hist_fan)
            torch.save(model, 'model.pth')
            target_model.load_state_dict(model.state_dict())

        if episode % 300 == 299:
            with open("buffer_history.pkl", "wb") as f:
                pickle.dump(fan_buffers, f)
            print(f'event=save, episode={episode}')

    print("Training finished.")
