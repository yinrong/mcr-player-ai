from common import *
from model import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from ai.env import SimplifiedMahjongEnv, num_actions
import mj

if __name__ == '__main__':
    set_seed(1)

    learning_rate = 0.04
    input_shape = num_actions
    hidden_size = 128
    dropout_prob = 0.1
    num_heads = 8
    num_layers = 3

    cnn_model = CNNQNetwork(input_shape, num_actions, hidden_size, dropout_prob)
    transformer_model = TransformerQNetwork(input_shape, num_actions, hidden_size, num_heads, num_layers, dropout_prob)
    fc_model = FCQNetwork(input_shape, num_actions, hidden_size)

    model = FCQNetwork(input_shape, num_actions, hidden_size)
    target_model = FCQNetwork(input_shape, num_actions, hidden_size)

    #model = EnsembleQNetwork(cnn_model, transformer_model, fc_model, num_actions)
    #target_model = EnsembleQNetwork(cnn_model, transformer_model, fc_model, num_actions)

    target_model.load_state_dict(model.state_dict())

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_function = nn.MSELoss()

    buffer = []
    unique_buffer = []
    batch_size = 64
    gamma = 0.99
    explore = 1.0
    epsilon_min = 0.5
    epsilon_decay = 1 - 1e-5

    def train_step():
        if len(buffer) < batch_size:
            return

        minibatch = random.sample(buffer, batch_size)
        states, discard_tiles, rewards, next_states, dones = zip(*minibatch)

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

        # 计算目标 Q 值：使用下一个状态的所有 Q 值的最大值（除弃掉的牌外的总和）
        max_next_q_values = next_q_values.sum(1) - next_q_values.gather(1, discard_tiles.unsqueeze(1)).squeeze(1)

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
    for episode in range(20000):
        state = env.reset()
        total_reward = 0

        while not env.done:
            if np.random.rand() < explore:
                #env.step(discard_by=env.discard_rand)
                env.step(discard_by=env.discard_strategy1)
            else:
                env.step()

        buffer += env.buffer
        if len(buffer) > max_len:
            random.shuffle(buffer)
            buffer = buffer[-max_len:] + unique_buffer
        for n in range(len(buffer) // 1000 + 1):
            train_step()
        
        if len(env.hist_fan) != last_hist_fan:
            unique_buffer += env.buffer
            print(f"episode={episode}, explore={explore:.2f}, data={len(buffer)}.{len(unique_buffer)}, found={len(env.hist_fan)}.{seq(env.hist_fan).make_string('.')}")
            last_hist_fan = len(env.hist_fan)

        if episode % 300 == 1:
            target_model.load_state_dict(model.state_dict())

    print("Training finished.")
