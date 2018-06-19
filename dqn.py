# -*- coding: utf-8 -*-

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from collections import namedtuple
import env2048

# if gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 4*4*16 -> 4*4*32
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 4*4*32 -> 5*5*64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 5*5*64 -> 4*4*64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        # 4*4*64 -> 4
        self.head = nn.Linear(4 * 4 * 64, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNPlayer():
    def __init__(self, name='DQN Player', eps=0.9):
        self.name_ = name
        self.eps_ = eps
        self.eps_end_ = 0.05
        self.batch_size_ = 128
        self.gamma_ = 0.9
        self.policy_net_ = DQN().to(device)
        self.target_net_ = DQN().to(device)
        self.target_net_.load_state_dict(self.policy_net_.state_dict())
        self.target_net_.eval()
        self.optimizer_ = optim.RMSprop(self.policy_net_.parameters())
        self.memory_ = ReplayMemory(10000)
        self.average_ = 0
        self.trained_ = 0
        self.TARGET_UPDATE = 10

    def select_action(self, state, greedy=False):
        return_item = False
        if type(state) is np.ndarray:
            return_item = True
            state = torch.tensor(env2048.state2tensor(state), device=device, dtype=torch.float32)
        if greedy or np.random.random() > self.eps_:
            with torch.no_grad():
                ret = self.policy_net_(state).max(1)[1].view(1, 1)
        else:
            ret = torch.tensor(
                [[random.randrange(4)]], device=device, dtype=torch.long)
        if return_item:
            return ret.item()
        else:
            return ret


    def optimize_model(self):
        if len(self.memory_) < self.batch_size_:
            return
        transitions = self.memory_.sample(self.batch_size_)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.uint8)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net_(state_batch).gather(
            1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size_, device=device)
        next_state_values[non_final_mask] = self.target_net_(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma_) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer_.zero_grad()
        loss.backward()
        for param in self.policy_net_.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_.step()

    def training(self, num_episodes):
        last10 = []
        env = env2048.Env2048()
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env.reset()
            state = torch.tensor(env.to_tensor(), device=device, dtype=torch.float32)
            for i_step in range(100000):
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor(
                    [reward], device=device, dtype=torch.float32)
#                print('Reward %f'%(reward))
                if not done:
                    next_state = torch.tensor(env.to_tensor(), device=device, dtype=torch.float32)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory_.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net_.load_state_dict(self.policy_net_.state_dict())


#            self.average_ = (env.get_return() + self.average_
#                             * self.trained_) / (self.trained_ + 1)
            last10.append(env.get_return())
            if len(last10) > 10:
                last10.pop(0)
            self.average_ = np.average(last10)
            self.trained_ += 1
            self.eps_ = self.eps_end_ + np.exp(
                -self.trained_ / 200) * (0.9 - 0.05)
            print('Episode %3d, steps %4d, score %5d, avg %5.0f, eps %.4f' %
                  (self.trained_, i_step, env.get_return(), self.average_, self.eps_))


def __test__():
    net = DQN().to(device)
    s = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
        dtype=np.int8)
    print(net.forward(torch.tensor(env2048.state2tensor(s), device=device, dtype=torch.float32)))
    player = DQNPlayer()
    print(player.policy_net_.forward(torch.tensor(env2048.state2tensor(s), device=device, dtype=torch.float32)))
    print(player.select_action(s))
    env2048.test_player(player, 10)


if __name__ == '__main__':
    n_episodes = 2000
    player = DQNPlayer()
    player.training(n_episodes)
    player.eps_=0.05
    env2048.test_player(player, 100)

