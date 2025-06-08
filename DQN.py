import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gomoku_env import GomokuEnv
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

env = GomokuEnv()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class LinearModel(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, 256)
        self.layer6 = nn.Linear(256, n_actions)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.drop(x)
        x = F.relu(self.layer3(x))
        x = self.drop(x)
        x = F.relu(self.layer4(x))
        x = self.drop(x)
        # x = F.relu(self.layer5(x))
        return self.layer6(x)
    
    
class CNNModel(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=8):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=n_hidden,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_hidden,
                      out_channels=n_hidden*2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(n_hidden*2, n_hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_hidden, n_hidden, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_hidden*(n_hidden+1),
                      out_features=n_actions)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, _ = env.reset()
n_observations = 15 * 15


select_model = 0

if select_model == 0:
    policy_net_0 = LinearModel(n_observations, n_actions).to(device)
    target_net_0 = LinearModel(n_observations, n_actions).to(device)

    policy_net_1 = LinearModel(n_observations, n_actions).to(device)
    target_net_1 = LinearModel(n_observations, n_actions).to(device)
elif select_model == 1:
    policy_net_0 = CNNModel(n_observations, n_actions).to(device)
    target_net_0 = CNNModel(n_observations, n_actions).to(device)

    policy_net_1 = CNNModel(n_observations, n_actions).to(device)
    target_net_1 = CNNModel(n_observations, n_actions).to(device)


target_net_0.load_state_dict(policy_net_0.state_dict())
optimizer_0 = optim.AdamW(policy_net_0.parameters(), lr=LR, amsgrad=True)
memory_0 = ReplayMemory(10000)

target_net_1.load_state_dict(policy_net_1.state_dict())
optimizer_1 = optim.AdamW(policy_net_1.parameters(), lr=LR, amsgrad=True)
memory_1 = ReplayMemory(10000)


steps_done = 0


def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * int(steps_done / 2) / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

        
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



num_episodes = 2000

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        if i_episode + 1 == num_episodes:
            env.render()

        action = select_action(state, policy_net_0)
        
        observation, reward, terminated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated

        # if done:
        #     next_state = None
        # else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory_0.push(state, action, next_state, reward)

        state = next_state

        optimize_model(memory_0, policy_net_0, target_net_0, optimizer_0)

        target_net_state_dict = target_net_0.state_dict()
        policy_net_state_dict = policy_net_0.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_0.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

        

        if i_episode + 1 == num_episodes:
            env.render()

        action = select_action(state, policy_net_1)
        observation, reward, terminated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated

        # if done:
        #     next_state = None
        # else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory_1.push(state, action, next_state, reward)

        state = next_state

        optimize_model(memory_1, policy_net_1, target_net_1, optimizer_1)

        target_net_state_dict = target_net_1.state_dict()
        policy_net_state_dict = policy_net_1.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_1.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()