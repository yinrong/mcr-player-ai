import mj
import random

class SimplifiedMahjongEnv:
    def __init__(self):
        self.state = None
        self.done = False
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()

    def reset(self):
        """
        重置环境，返回初始状态。
        """
        self.state = self.initialize_state()
        self.done = False
        return self.state

    def step(self, action):
        """
        执行动作，返回（下一状态, 奖励, 是否结束, 额外信息）。
        """
        if self.done:
            raise ValueError("Episode has ended, please reset the environment.")

        next_state = self.state_transition(action)
        self.done, reward = self.calculate_reward(next_state)
        
        self.state = next_state
        return next_state, reward, self.done

    def render(self):
        """
        渲染环境状态。
        """
        print(len(self.state['pool']), self.state['hand'])

    def initialize_state(self):
        """
        初始化状态。
        """
        # 初始化手牌、牌池等，具体实现根据麻将规则确定
        pool = [i for i in range(1, 35) for _ in range(4)]
        random.shuffle(pool)
        return {'hand': [], 'pool': pool, 'discarded': []}

    def state_transition(self, action):
        """
        状态转移函数，根据动作更新状态。
        """
        # 根据动作更新状态，具体实现根据麻将规则确定
        next_state = self.state.copy()
        # 示例：假设action是摸牌
        if action == 'draw':
            next_state['hand'].append(self.draw_tile())
            next_state['hand'].sort()
        elif action == 'discard':
            tile = random.choice(next_state['hand'])
            next_state['hand'].remove(tile)
            next_state['discarded'].append(tile)
            next_state['hand'].sort()
        # 添加其他动作的处理
        return next_state

    def calculate_reward(self, state):
        """
        根据动作和当前状态计算奖励。
        """
        p = state['hand']
        if len(p) < 14:
            n = 0
        else:
            n = mj.quick_calc(p)
        done = n >= 8 or self.is_deck_empty()
        return done, n


    def draw_tile(self):
        x = self.state['pool'].pop()
        return x

    def is_deck_empty(self):
        """
        判断牌堆是否为空。
        """
        # 示例：判断牌池是否空了
        return len(self.state['pool']) == 0


class ActionSpace:

    def sample(self, state):
        """
        根据当前状态选择一个合法的动作。
        
        参数：
        state (dict): 当前状态。
        
        返回：
        action (str): 合法的动作。
        """
        available_actions = []

        # 当手牌达到14张时，可以选择discard动作
        if len(state['hand']) == 14:
            available_actions.append('win')
            available_actions.append('discard')

        # 当手牌少于14张且牌池不为空时，可以选择draw动作
        if len(state['hand']) < 14 and state['pool']:
            available_actions.append('draw')

        ## 根据具体游戏规则，判断chow、pong、kong、win动作是否可行
        #if self.can_chow(state):
        #    available_actions.append('chow')
        #if self.can_pong(state):
        #    available_actions.append('pong')
        #if self.can_kong(state):

        return random.choice(available_actions) if available_actions else None

    @property
    def n(self):
        """
        返回动作的数量。
        """
        return len(self.actions)

class ObservationSpace:
    """
    状态空间类，定义状态的维度。
    """
    def __init__(self):
        self.shape = (1, )  # 状态的维度，根据具体实现确定




if __name__ == '__main__':
    # 使用示例
    env = SimplifiedMahjongEnv()
    state = env.reset()
    done = False
    best_reward = 0
    while best_reward == 0:
        while not done:
            action = env.action_space.sample(state)
            next_state, reward, done = env.step(action)
            #env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        best_reward = max(reward, best_reward)