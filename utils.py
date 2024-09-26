import numpy as np 
import torch 
import copy
import os
import errno



class Replay_buffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size,1))
        self.not_done = np.zeros((max_size,1))



        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1 - done

        self.ptr = (self.ptr +1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def len(self):
        return self.size

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device), 
            torch.FloatTensor(self.next_state[ind]).to(self.device), 
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device) 
        )

class TargetNet:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict)

    def alpha_sync(self, alpha):
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k,v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1- alpha) * v
        self.target_model.load_state_dict(tgt_state)



    
