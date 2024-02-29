from gym.wrappers import TimeLimit
from env_hiv import HIVPatient
from parsers import get_train_parser

import numpy as np
import torch
from torch import nn, optim

import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

args = get_train_parser()


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    
    def __init__(self, model, args=args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
    
        self.model = model.to(device)
        self.model_name = 'model'
        self.filename = self.model_name + '.pkl'
        
        self.gamma = args.gamma
        self.epsilon_max = args.epsilon_max
        self.epsilon_min = args.epsilon_min
        self.epsilon_delay = args.epsilon_delay
        self.epsilon_stop = args.epsilon_stop
        
        self.memory = ReplayBuffer(args.buffer_size, device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.criterion = nn.MSELoss()
        self.nb_epoch = args.nb_epoch
        self.batch_size = args.batch_size
        
    
    
    def act(self, observation, use_random=False):
        
        return self.model(observation)

    def save(self, path):
        f = open(self.filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close() 

    def load(self):
        f = open(self.filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self):
        
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        max_episode = self.nb_epoch

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
                
        return episode_return        
    
    
class FFModel(nn.Module):
    def __init__(self, state_dim, action_dim, nhid=64):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(state_dim, nhid)
        self.fc2 = nn.Linear(nhid, nhid)
        self.fc3 = nn.Linear(nhid, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=1)
