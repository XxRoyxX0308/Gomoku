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
models_dir.mkdir(parents=True, exist_ok=True)

Path.joinpath(models_dir, "policy_net_0").mkdir(parents=True, exist_ok=True)
Path.joinpath(models_dir, "target_net_0").mkdir(parents=True, exist_ok=True)
Path.joinpath(models_dir, "policy_net_1").mkdir(parents=True, exist_ok=True)
Path.joinpath(models_dir, "target_net_1").mkdir(parents=True, exist_ok=True)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


win = None

class ReplayMemory(object):

    def __init__(self, capacity, number):
        self.number = number
        self.memory = deque([], maxlen=capacity)
        self.past_args = None

    def push(self, *args):
        global win

        if self.past_args != None:
            if args[3] == 1: win = self.number

            if win == int(not self.number):
                self.past_args[3] -= 1
                win = None

            self.memory.append(Transition(*self.past_args))
        
        self.past_args = list(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

    
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


BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.05
LR = 1e-4  ## 4
episode_start = 0
num_episodes = 100000
save_epis = 500


state_load = True



num_hidden = 32

policy_net_0 = CNNModel(num_hidden).to(device)
target_net_0 = CNNModel(num_hidden).to(device)

policy_net_1 = CNNModel(num_hidden).to(device)
target_net_1 = CNNModel(num_hidden).to(device)


if state_load:
    episode_start = 5000
    policy_net_0.load_state_dict(
        torch.load(Path.joinpath(models_dir, "policy_net_0", f"{episode_start}_epis.pt"), map_location=device)
    )
    policy_net_1.load_state_dict(
        torch.load(Path.joinpath(models_dir, "policy_net_1", f"{episode_start}_epis.pt"), map_location=device)
    )
    EPS_START = EPS_END


target_net_0.load_state_dict(policy_net_0.state_dict())
optimizer_0 = optim.AdamW(policy_net_0.parameters(), lr=LR, amsgrad=True)
memory_0 = ReplayMemory(10000, 0)

target_net_1.load_state_dict(policy_net_1.state_dict())
optimizer_1 = optim.AdamW(policy_net_1.parameters(), lr=LR, amsgrad=True)
memory_1 = ReplayMemory(10000, 1)


steps_done = 0


def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * int(steps_done / 2) / EPS_DECAY)
    steps_done += 1
    # if sample > eps_threshold:
    if True:
        with torch.no_grad():
            mask = (state[0][0] + state[0][1]).to(bool)
            prob = policy_net(state)[0, 0]
            prob_masked = torch.masked_select(prob, ~mask)

            return (prob==torch.max(prob_masked)).nonzero()[0].unsqueeze(0)
    else:
        zero_indices = (state[0][0] + state[0][1] == 0).nonzero(as_tuple=False)
        chosen_index = zero_indices[random.randint(0, zero_indices.size(0) - 1)]
        
        return chosen_index.unsqueeze(0)


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
    
    prob = policy_net(state_batch).squeeze(1)
    state_action_values = prob[action_batch[:, 0], action_batch[:, 1]]

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        pred = target_net(non_final_next_states)
        next_state_values[non_final_mask] = pred.view(pred.size(0), -1).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state, policy_net_0)

        observation, reward, terminated, _ = env.step(action.tolist()[0])
        reward = torch.tensor([reward], device=device)
        done = terminated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory_0.push(state, action, next_state, reward)

        state = next_state

        optimize_model(memory_0, policy_net_0, target_net_0, optimizer_0)

        target_net_state_dict = target_net_0.state_dict()
        policy_net_state_dict = policy_net_0.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_0.load_state_dict(target_net_state_dict)

        if i_episode + 1 == num_episodes:
            print(action)
            env.render()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

        

        action = select_action(state, policy_net_1)
        observation, reward, terminated, _ = env.step(action.tolist()[0])
        reward = torch.tensor([reward], device=device)
        done = terminated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory_1.push(state, action, next_state, reward)

        state = next_state

        optimize_model(memory_1, policy_net_1, target_net_1, optimizer_1)

        target_net_state_dict = target_net_1.state_dict()
        policy_net_state_dict = policy_net_1.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net_1.load_state_dict(target_net_state_dict)

        if i_episode + 1 == num_episodes:
            print(action)
            env.render()

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


    if (i_episode+1) % save_epis == 0:
        torch.save(
            policy_net_0.state_dict(),
            Path.joinpath(models_dir, "policy_net_0", f"{episode_start+i_episode+1}_epis.pt")
        )
        torch.save(
            target_net_0.state_dict(),
            Path.joinpath(models_dir, "target_net_0", f"{episode_start+i_episode+1}_epis.pt")
        )
        torch.save(
            policy_net_1.state_dict(),
            Path.joinpath(models_dir, "policy_net_1", f"{episode_start+i_episode+1}_epis.pt")
        )
        torch.save(
            target_net_1.state_dict(),
            Path.joinpath(models_dir, "target_net_1", f"{episode_start+i_episode+1}_epis.pt")
        )


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()