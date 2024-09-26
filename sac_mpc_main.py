#!/usr/bin/env python

import torch
import environment_mpc
import rospy
import errno
import os
import numpy as np
import random
import time
import sys 
import copy

from utils import Replay_buffer
from SAC_net import SAC
from tensorboardX import SummaryWriter
import datetime


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

STATE_DIMENSION = 24
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.5  # m/s
ACTION_W_MAX = 0.785  # rad/s

BATCH_SIZE = 256
REPLAY_SIZE = 100000
REPLAY_INITIAL = 4000
MAX_STEPS = 1000


TEST_ITERS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

#---Functions to make network updates---#

if __name__ == "__main__":
    rospy.init_node('agent25per')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action='store_true', help='Enable CUDA')

    data = []
    

    seed =  random.randint(1,1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L= []
    collisions = []
    date = datetime.datetime.now().strftime("%d.%h.%H")
    writer = SummaryWriter(dirPath+'/runs/agent25per/'+str(seed)+date)
    save_path = os.path.join(dirPath, "runs/agent25per/", str(seed)+date)
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    
    env = environment_mpc_finmod.Env(action_dim=ACTION_DIMENSION)

    past_action = np.zeros(ACTION_DIMENSION)

    replay_buffer = Replay_buffer(STATE_DIMENSION, ACTION_DIMENSION)

    policy= SAC(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, discount = 0.99, reward_scale=2)


    cntr = 0
    
    episode = 0
    best_reward = None
    test_episode = 0
    print(seed)
    
    while episode < 7001:
        
        mpc = False
        if episode==0 :
            mpc_rate = 0.25 * pow(0.995, episode//4)
        eps = random.random()
        if (eps<mpc_rate) : 
            mpc = True
        done = False
        episode += 1
 
        rewards_current_episode = 0.0
        print("agent agent25per episode: ", episode)
        if not episode % TEST_ITERS == 0:
            state = env.reset()
            r = rospy.Rate(100)
            for step in range(MAX_STEPS):
                state = np.float32(state)
                actions = policy.select_action(state, eval=False)

                action = np.array([policy.action_unnormalized(actions[0], ACTION_V_MAX, -ACTION_V_MAX), policy.action_unnormalized(actions[1], ACTION_W_MAX, -ACTION_W_MAX)])
                next_state, reward, done, act_taken = env.step(action, get_mpc=False)

                rewards_current_episode += reward
                next_state = np.float32(next_state)

                    
                replay_buffer.add(state, act_taken, next_state, reward, done)

                state = copy.deepcopy(next_state)

                if replay_buffer.len() >= REPLAY_INITIAL:
                    policy.train(replay_buffer,BATCH_SIZE)

                if done or step == MAX_STEPS-1:
                    break

                r.sleep()
                

                
print("agent finished training")
print(seed)
pass

