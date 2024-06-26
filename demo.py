from common import *
from model import FCQNetwork
from ai.env import SimplifiedMahjongEnv
from mjutil import renderSimple
from train import input_shape, num_actions, hidden_size

# 加载模型
model = torch.load("model.pth")
model.eval()

# 创建环境
env = SimplifiedMahjongEnv(model)

# 进行游戏步骤并显示每次弃牌后的手牌和已弃牌列表
state = env.reset()

while True:
    hand_array = np.zeros(num_actions)
    for tile in state['hand']:
        hand_array[tile - 1] += 1

    last_state = env.state['hand'].copy()
    env.step()
    if env.done:
        break
    renderSimple(len(env.buffer), last_state, env.state['discarded'][-1])

renderSimple(len(env.buffer), env.state['hand'], None)
print(env.this_fan, seq(env.this_fans).make_string('.'))