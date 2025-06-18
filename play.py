import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gomoku_env import GomokuEnv
import math
from pathlib import Path
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

name = "DQN"
env = GomokuEnv(board_size = 15)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


abs_dir = Path(__file__).parent.absolute()
models_dir = Path.joinpath(abs_dir, "models", name)

    
class CNNModel(nn.Module):
    def __init__(self, n_hidden=8):
        super().__init__()
        self.cnn_block_1 = nn.Sequential(
            nn.Conv2d(2, n_hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(n_hidden*2, n_hidden*2, 5, padding=2),
            # nn.ReLU(),
        )

        self.cnn_block_2 = nn.Sequential(
            nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(n_hidden*4, n_hidden*4, 3, padding=1),
            # nn.ReLU(),
        )
        
        self.pool = nn.MaxPool2d(2, stride=1, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=1)

        self.cnn_t_block_2 = nn.Sequential(
            nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            # nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            # nn.ReLU(),
        )

        self.cnn_t_block_1 = nn.Sequential(
            nn.Conv2d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden*2, n_hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden, 1, 3, padding=1),
        )
        

    def forward(self, x):
        x = self.cnn_block_1(x)
        # x, indices_0 = self.pool(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_2(x) #has or not
        # x, indices_1 = self.pool(x)
        x = self.cnn_block_2(x)
        x = self.cnn_block_2(x) #has or not not
        # x = self.unpool(x, indices_1)
        x = self.cnn_block_2(x)
        x = self.cnn_block_2(x) #has or not not
        
        x = self.cnn_t_block_2(x)
        # x = self.unpool(x, indices_0)
        x = self.cnn_block_2(x) #has or not not

        x = self.cnn_t_block_1(x)

        return x



num_hidden = 32

policy_net_0 = CNNModel(num_hidden).to(device)
target_net_0 = CNNModel(num_hidden).to(device)

policy_net_1 = CNNModel(num_hidden).to(device)
target_net_1 = CNNModel(num_hidden).to(device)



episode_start = 10000
policy_net_0.load_state_dict(
    torch.load(Path.joinpath(models_dir, "policy_net_0", f"{episode_start}_epis.pt"), map_location=device)
)

target_net_0.load_state_dict(policy_net_0.state_dict())



def select_action(state, policy_net):
    with torch.no_grad():
        mask = (state[0][0] + state[0][1]).to(bool)
        prob = policy_net(state)[0, 0]
        prob_masked = torch.masked_select(prob, ~mask)

        return (prob==torch.max(prob_masked)).nonzero()[0].unsqueeze(0)


while True:
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state, policy_net_0)

        observation, reward, terminated, _ = env.step(action.tolist()[0])
        reward = torch.tensor([reward], device=device)
        done = terminated

        if done and reward == 1:
            print("ai win");

        env.render()
        

        user_input = list(map(int, input().split()))
        observation, reward, terminated, _ = env.step(user_input)
        reward = torch.tensor([reward], device=device)
        done = terminated

        if done and reward == 1:
            print("u win");
        
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        state = next_state

        env.render()