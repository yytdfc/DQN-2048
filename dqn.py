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

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'ret'))

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
        self.head = nn.Linear(4*4*64, 4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def convert_state2net(state):
    n = Tensor(16,16).fill_(0)
    for i,c in enumerate(state.flatten()):
        n[c,i]=1 #2.0**c
    n.resize_(1,16,4,4)
    return n

class DQNPlayer():
    def __init__(self, name = 'DQN Player', eps = 0.9):
        self.name_ = name
        self.eps_ = eps
        self.eps_end_ = 0.05
        self.batch_size_ = 128
        self.gamma_ = 0.999
        self.dqn_ = DQN()
        if use_cuda:
            self.dqn_.cuda()
        self.optimizer_ = optim.RMSprop(self.dqn_.parameters())
        self.memory_ = ReplayMemory(1000)
        self.average_ = 0
        self.trained_ = 0
    def select_action(self, state, greedy=False):
        if greedy or np.random.random()>self.eps_:
            n = convert_state2net(state)
#            n.resize_(1,16,4,4)
            return torch.max(self.dqn_(Variable(n)).data,1)[1][0]
        else:
            return np.random.randint(4)
    def optimize_model(self):
        if len(self.memory_) < self.batch_size_:
            return
        transitions = self.memory_.sample(self.batch_size_)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))
#        final_mask = ByteTensor(tuple(map(lambda s: s is None,
#                                              batch.next_state)))
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        return_batch = Variable(torch.cat(batch.ret))
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.dqn_(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size_).type(Tensor))
        next_state_values[non_final_mask] = self.dqn_(non_final_next_states).max(1)[0]
#        next_state_values[final_mask] = reward_batch[final_mask]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma_) \
                                        - reward_batch + return_batch
#        print(batch.state[0],batch.next_state[0])
#        print(state_action_values.data[0][0],expected_state_action_values.data[0])
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
#        print(loss.data[0])
        # Optimize the model
        self.optimizer_.zero_grad()
        loss.backward()
        for param in self.dqn_.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_.step()
#        input()
    def training(self, num_episodes):
        env = env2048.Env2048()
        last10=[]
        for i_episode in range(num_episodes):
            # Initialisze the environment and state
            env.reset()
            state = env.get_state()
            while 1:
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done = env.step(action)
                
                # Observe new state
                if not done:
                    next_state = env.get_state()
                    if reward>=0:
                        # Store the transition in memory
                        self.memory_.push(convert_state2net(state),
                                          LongTensor([[action]]),
                                          convert_state2net(next_state),
                                          Tensor(1).fill_(float(reward)),
                                          Tensor(1).fill_(0))
                else:
                    next_state = None
                    # Store the transition in memory
                    self.memory_.push(convert_state2net(state),
                                      LongTensor([[action]]),
                                      None,
                                      Tensor(1).fill_(float(reward)),
                                      Tensor(1).fill_(float(env.get_return())))
                # Move to the next state
                state = next_state
    
                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                
                if done:
                    
                    break
#            self.average_ = (env.get_return() + self.average_ 
#                             * self.trained_) / (self.trained_ + 1)
            last10.append(env.get_return())
            if len(last10)>10:
                last10.pop(0)
            self.average_ = np.average(last10)
            self.trained_ += 1
            self.eps_ = self.eps_end_ + np.exp(-self.trained_/200) * (0.9-0.05)
            print('epi %3d, score %5d, avg %5.0f, eps %.4f'%(self.trained_,
                    env.get_return(), self.average_, self.eps_))
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
    n_episodes = 200
    player = DQNPlayer()
    player.training(n_episodes)
    env2048.test_player(player, 10)
    s=np.array([[0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]], dtype=np.int8)
    print(player.dqn_.forward(Variable(convert_state2net(s))))
    s=np.array([[0, 0, 0, 0],
                [3, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 3, 0]], dtype=np.int8)
    print(player.dqn_.forward(Variable(convert_state2net(s))))
    s=np.array([[0, 0, 0, 0],
                [1, 2, 3, 0],
                [1, 2, 3, 0],
                [0, 0, 1, 0]], dtype=np.int8)
    print(player.dqn_.forward(Variable(convert_state2net(s))))