#!/usr/bin/env python


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distr
import torch
import numpy as np
import copy
from utils import Replay_buffer, TargetNet



STATE_DIMENSION = 24
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.5  # m/s
ACTION_W_MAX = 0.785  # rad/s


GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 100000
REPLAY_INITIAL = 4000
MAX_STEPS = 1000
LR_ACTS = 3e-4
LR_VALS = 3e-4
LR_ALPHA = 3e-4
HID_SIZE = 256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACActor(nn.Module):
    def init_weights(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self, obs_size, act_size):
        super(SACActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size,HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size)
        )
        
        self.sigma = nn.Sequential(
            nn.Linear(obs_size,HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
        )
        
        self.mu.apply(self.init_weights)
        self.sigma.apply(self.init_weights)

    def forward(self,x):
        mu, sigma = self.mu(x) , self.sigma(x)
        sigma  = torch.clamp(sigma, min=1e-6, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probs = distr.Normal(mu, sigma)

        if reparameterize:
            x_t = probs.rsample()
        else:
            x_t = probs.sample()

        action = torch.tanh(x_t).to(device)
        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(1-action.pow(2)+(1e-6))
        log_probs = log_probs.sum(1, keepdim=True) 
                    
        return action, log_probs, mu, sigma



class TwinQNets(nn.Module):
    def __init__(self, obs_size, act_size):
        super(TwinQNets, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(), 
            nn.Linear(HID_SIZE, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(), 
            nn.Linear(HID_SIZE, 1)
        )
       
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1(x), self.q2(x)

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action_v, max_action_w, discount = 0.99, reward_scale=2):

        self.actor = SACActor(STATE_DIMENSION, ACTION_DIMENSION).to(device)
        self.qnets = TwinQNets(STATE_DIMENSION,ACTION_DIMENSION).to(device)
        self.qs_tgt = TargetNet(self.qnets)
        self.log_alpha = torch.tensor(np.log(INIT_TEMPERATURE)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        
        self.act_opt = optim.Adam(self.actor.parameters(), lr=LR_ACTS)
        self.qnets_opt = optim.Adam(self.qnets.parameters(), lr=LR_VALS)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)

        self.max_action_v = max_action_v
        self.max_action_w = max_action_w
        self.discount = discount
        self.reward_scale = reward_scale
    
    def select_action(self, states, eval=False):
        states_v = torch.FloatTensor(states).to(device).unsqueeze(0)
        if eval == False:
            actions,_,_,_ = self.actor.sample_normal(states_v, reparameterize=True)
        else:
            _,_,actions,_= self.actor.sample_normal(states_v, reparameterize=False)
            actions = torch.tanh(actions)
        
        actions = actions.detach().cpu().numpy()[0]
        return actions
    

    def train(self, replay_buffer, batch_size=256):
        
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():

            actions_smpld, log_probs, _,_ = self.actor.sample_normal(next_state, reparameterize=False)
            q1, q2 = self.qs_tgt.target_model(next_state, actions_smpld)
            target_q = torch.min(q1, q2) - (self.log_alpha.exp().detach()) * log_probs
            ref_q = self.reward_scale * reward + not_done * self.discount * target_q
            ref_q_v  = ref_q.to(device)
            

        self.qnets_opt.zero_grad()
        
        q1_v, q2_v = self.qnets(state, action)
        q1_loss_v = F.mse_loss(q1_v, ref_q_v.detach())
        q2_loss_v = F.mse_loss(q2_v, ref_q_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v
        q_loss_v.backward()
        self.qnets_opt.step()


        self.act_opt.zero_grad()
        actions_, log_probs,_, _ = self.actor.sample_normal(state, reparameterize=True)
        q1_val, q2_val = self.qnets(state, actions_)
        min_val = torch.min(q1_val, q2_val)
        actor_loss = ((self.log_alpha.exp().detach()) * log_probs - min_val).mean() 
        actor_loss.backward()
        self.act_opt.step()

        self.qs_tgt.alpha_sync(alpha=1 - 1e-3)
        
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.log_alpha.exp() *
                        (-log_probs - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def action_unnormalized( self, action, high, low):
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action
