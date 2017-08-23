# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import env2048

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 4*4*16 -> 4*4*32
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        # 4*4*32 -> 5*5*64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # 5*5*64 -> 4*4*64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, padding=0)
        # self.bn1 = nn.BatchNorm2d(64)
        # 4*4*64 -> 4
        self.head = nn.Linear(4*4*64, 4)

    def forward(self, x):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))

def convert_state2net(state):
    n = torch.FloatTensor(16,16).fill_(0)
    for i,c in enumerate(state.flatten()):
        n[c,i]=1
    n.resize_(16,4,4)
    return n

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
def select_action(state):
    n=convert_state2net(state)
    n.resize_(1,16,4,4)
    net.forward(Variable(n))

class DQNPlayer():
    def __init__(self, name = 'DQN Player', eps = 0.8):
        self.name_ = name
        self.eps_ = eps
        self.dqn_ = DQN()
    def genmove(self, state, greedy=False):
        if greedy or np.random.random()<self.eps_:
            n = convert_state2net(state)
            n.resize_(1,16,4,4)
            return torch.max(self.dqn_.forward(Variable(n)).data,1)[1][0]
        else:
            return np.random.randint(4)
    def training(num_episodes):
        env = env2048.Env2048()
        for i_episode in range(num_episodes):
            # Initialisze the environment and state
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()
            state = env.get_state()
            for t in count():
                # Select and perform an action
                action = select_action(state)
                _, reward, done, _ = env.step(action[0, 0])
                reward = Tensor([reward])
    
                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None
    
                # Store the transition in memory
                memory.push(state, action, next_state, reward)
    
                # Move to the next state
                state = next_state
    
                # Perform one step of the optimization (on the target network)
                optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
def __test__():
    net = DQN()
    s=np.array([[0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]], dtype=np.int8)
    n=convert_state2net(s)
    n.resize_(1,16,4,4)
    print(net.forward(Variable(n)))
    
if __name__ == '__main__':
    n_episodes = 10
    env2048.test_player(DQNPlayer(), n_episodes)