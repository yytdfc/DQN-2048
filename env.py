# -*- coding: utf-8 -*-

import numpy as np
class env_2048(object):
    def __init__(self, dim=4, base=2, state=None):
        if not state:
            self.state_ = np.zeros((dim,dim), dtype=np.int8)
        self.dim_ = dim
        self.base_ = 2
        self.start_tiles_ = 2
        self.add_start_tile()
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
        pass
    def action(self, act):
        new_state = self.state
        reward = 0
        return new_state, reward
    def score(self):
        return np.sum(self.base_**self.state_)
    
    ## functions according to 2048/js/game_manager.js
    def add_start_tile(self):
        for i in range(self.start_tiles_):
            self.add_random_tile()
    def add_random_tile(self):
        self.state_.itemset(self.random_available_cell(),
                    1 if np.random.random() < 0.9 else 2)
    def random_available_cell(self):
        return np.random.choice(self.available_cell())
    def available_cell(self):
        return np.argwhere(self.state_.flatten()==0).flatten()
    
if __name__ == '__main__':
    env = env_2048()
    print(env)