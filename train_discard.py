from common import *
from quezha_data_convert import prepare_data

# 模型定义
class MahjongDiscardModel(nn.Module):
    def __init__(self, num_tile_types=34, hidden_size=128, num_actions=34):
        super(MahjongDiscardModel, self).__init__()
        self.hand_embedding = nn.Linear(num_tile_types, hidden_size)
        self.meld_embedding = nn.Embedding(num_tile_types * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, hand, melds):
        hand_emb = self.hand_embedding(hand)  # shape: (batch_size, hidden_size)
        melds_emb = self.meld_embedding(melds)  # shape: (batch_size, 4, hidden_size)
        melds_emb = torch.sum(melds_emb, dim=1)  # shape: (batch_size, hidden_size)

        combined = torch.cat([hand_emb, melds_emb], dim=1)  # shape: (batch_size, hidden_size * 2)
        x = F.leaky_relu(self.fc1(combined))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型训练示例
def train_model(model, action_list, epochs=100, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    buffer = deque(maxlen=5000)

    # 准备数据
    hand_inputs, meld_inputs, targets = prepare_data(action_list)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(hand_inputs.size()[0])

        for i in range(0, hand_inputs.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_hand_inputs, batch_meld_inputs, batch_targets = hand_inputs[indices], meld_inputs[indices], targets[indices]

            optimizer.zero_grad()
            outputs = model(batch_hand_inputs, batch_meld_inputs)
            loss = loss_function(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
