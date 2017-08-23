# -*- coding: utf-8 -*-

import numpy as np
ACTION_MAP = [ '↑','↓','←','→']
class Env2048(object):
    def __init__(self, dim=4, base=2, state=None):
        self.dim_ = dim
        self.base_ = base
        self.start_tiles_ = 2
        self.score_ = 0
        if state is None:
            self.state_ = np.zeros((self.dim_,self.dim_), dtype=np.int8)
            self.add_start_tile()
        else:
            self.state_ = state.copy()
    def __str__(self):
        conver2char = lambda num :'%5d'%(self.base_**num) if num>0 else ' '*5
        demarcation = ( '+' + '-'*5 ) * self.dim_ + '+\n'
        ret = demarcation
        ret += demarcation.join(['|'+'|'.join([conver2char(num) 
                        for num in row])+'|\n' for row in self.state_])
        ret += demarcation
        return ret
    def __repr__(self):
        return self.__str__(self)
    def set_state(self, state):
        self.state_ = state.copy()
        self.score_ = 0
    def get_state(self):
        return self.state_.copy()
    def reset(self):
        self.state_ = np.zeros((self.dim_,self.dim_), dtype=np.int8)
        self.add_start_tile()
        self.score_ = 0
    def act(self, action):
        score = self.score_
        if self.move(action):
            self.add_random_tile()
            return self.state_.copy(), self.score_-score, self.is_terminate()
        else:
            return self.state_.copy(), -8, self.is_terminate()
    def get_return(self):
        return self.score_
    ## functions according to 2048/js/game_manager.js
    def add_start_tile(self):
        for i in range(self.start_tiles_):
            self.add_random_tile()
    def add_random_tile(self):
        avilable = self.available_cell()
        if avilable.size:
            self.state_.itemset(np.random.choice(avilable),
                    1 if np.random.random() < 0.9 else 2)
            return False
        else:
            return True
    def available_cell(self):
        return np.argwhere(self.state_.flatten()==0).flatten()
    def is_terminate(self):
        if self.available_cell().size==0:
            if (self.state_[:,:-1]==self.state_[:,1:]).any() \
                or (self.state_[:-1,:]==self.state_[1:,:]).any():
                return False
            else:
                return True
        return False
    
    def emit_line(self, line):
        changed = False
        # [ i, _ , j , _ ]
        i,j = 0,1
        while j<self.dim_:
            if line[j]==0:
                j+=1
            elif line[i]==0:
                changed = True
                line[i]=line[j]
                line[j]=0
                j+=1
            elif line[i]==line[j]:
                changed = True
                line[i]+=1
                line[j]=0
                self.score_ += self.base_**line[i]
                i+=1
                j+=1
            else:
                i+=1
            if j==i:
                j+=1
        return line, changed
    def move(self, direct):
        changed = False
        if direct==0: # up
            for i in range(4):
                self.state_[:,i], c = self.emit_line(self.state_[:,i])
                changed = changed or c
        elif direct==1: # down
            for i in range(4):
                self.state_[-1::-1,i], c = self.emit_line(self.state_[-1::-1,i])
                changed = changed or c
        elif direct==2: # left
            for i in range(4):
                self.state_[i,:], c = self.emit_line(self.state_[i,:])
                changed = changed or c
        elif direct==3: # right
            for i in range(4):
                self.state_[i,-1::-1], c = self.emit_line(self.state_[i,-1::-1])
                changed = changed or c
        return changed
    
class RandomPlayer():
    """
    A player which will take random move
    on 100 simulations: average score 1093, max score 2736
    """
    def __init__(self, name = 'Random Player'):
        self.name_ = name
    def genmove(self, state):
        return np.random.randint(4)
    
class OneStepPlayer():
    """
    A player which will search one step forward.
    on 100 simulations: average score 1811, max score 6192
    """
    def __init__(self, name = 'One Step Player'):
        self.name_ = name
    def genmove(self, state):
        rewards = [Env2048(state=state).act(i)[1] for i in range(4)]
        if np.max(rewards)>0:
            return np.argmax(rewards)
        else:
            return np.random.randint(4)
        
class MultiStepPlayer():
    """
    A player which will search steps depth.
    on 100 simulations:
        2 steps depth: average score 7648, max score 16132
        3 steps depth: average score 8609, max score 16248
    """
    def __init__(self, name = 'Mutil Step Player', steps=2):
        self.name_ = name
        self.n_steps_ = steps
    def genmove(self, state):
        rewards = np.zeros(4, dtype=np.int32)
        for i in range(4):
            env = Env2048(state=state)
            new_state, rewards[i], _ = env.act(i)
            for _ in range(self.n_steps_-1):
                new_state, reward, _ = env.act(OneStepPlayer().genmove(new_state))
                rewards[i] += reward
        if np.max(rewards)>0:
            return np.argmax(rewards)
        else:
            return np.random.randint(4)

def play_once(env, player):
    epoch=1
    while 1:
        a = player.genmove(env.get_state())
        _, r, t = env.act(a)
        if t:
            break
        epoch+=1
        # print('Ehoch %d: act %s'%(epoch, a))
    ret = env.get_return()
#    print('Ehoch %d: score %d'%(epoch, ret))
#    print(env)
    return ret

def test_player(player, n_episodes):
    env = Env2048()
    sum_ret = 0
    max_ret = 0
    for episode in range(n_episodes):
        env.reset()
        ret = play_once(env, player)
        sum_ret += ret
        if ret>max_ret:
#            print(env)
            max_ret = ret
    print('%s average score %d, max score %d'%(player.name_, sum_ret/n_episodes, max_ret))

def __test__():
    s=np.array([[0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]], dtype=np.int8)
    env = Env2048(state=s)
    print(env)
    env.act(1)
    print(env)

if __name__ == '__main__':
    n_episodes = 10
#    test_player(RandomPlayer(), n_episodes)
    test_player(MultiStepPlayer(steps=2), n_episodes)
    
