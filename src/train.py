from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from parsers import get_train_parser

import numpy as np
import random
import torch
from torch import nn, optim

from decimal import Decimal
from copy import deepcopy

from tqdm.auto import tqdm, trange



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

class FFModel(nn.Module):
    def __init__(self, state_dim, action_dim, nlayers = 1, nhid=64):
        super(FFModel, self).__init__()
        
        self.nlayers = nlayers
        self.nhid = nhid
        
        
        self.fc1 = nn.Linear(state_dim, nhid)
        
        self.layer_norm = nn.LayerNorm(nhid)
        
        self.nlayers = nlayers
        
        self.hidden_layers = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(nlayers)])
        
        self.fc3 = nn.Linear(nhid, action_dim)

    def forward(self, x):
        x = self.layer_norm(torch.relu( (self.fc1(x))))
        
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        return self.fc3(x)


args = get_train_parser()

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    
    def __init__(self, model=FFModel(args.state_dim, args.action_dim, args.nlayers, args.nhid)
                 , args=args):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
    
        self.model = model.to(self.device)
        self.model_name = 'model'
        
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = args.update_target_strategy
        self.update_target_freq = args.update_target_freq
        self.update_target_tau = args.update_target_tau

        self.gamma = args.gamma
        self.epsilon_max = args.epsilon_max
        self.epsilon_min = args.epsilon_min
        self.epsilon_delay = args.epsilon_delay
        self.epsilon_stop = args.epsilon_stop
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        
        self.memory = ReplayBuffer(args.buffer_size, self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        #self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, total_steps=args.nb_epoch*200, pct_start=0.1, final_div_factor=1000)
        
        self.criterion = nn.SmoothL1Loss() if args.criterion == 'l1' else nn.MSELoss()
        self.nb_epoch = args.nb_epoch
        self.batch_size = args.batch_size
        self.nb_gradient_steps = args.nb_gradient_steps
        
        self.monitoring_nb_trials = args.monitoring_nb_trials
        self.monitoring_frequency = args.monitoring_frequency

    def greedy_action(self, network, state):
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def act(self, observation, use_random=False):
        return self.greedy_action(self.model, observation)

    def save(self, path='./model.pt'):
        torch.save(self.model.state_dict(), path)

    def load(self):
        state_dict = torch.load('./model.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        
    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)

    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
            return loss.item()
        
        else:
            return 1e9
            
    def fill_buffer(self, env, nb_example=10_000):
        state, _ = env.reset()
        for i in trange(nb_example):
            action = self.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if done or trunc:
                state, _ = env.reset()
            else:
                state = next_state

    
    def train(self, env):
        episode_return = []
        MC_avg_total_reward = []   # NEW NEW NEW
        best_MC_avg_total_reward = 0
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        
        self.model.train()
        
        while episode < self.nb_epoch:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                current_loss = self.gradient_step()
            
            #self.scheduler.step()    
            
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials>0 and episode%self.monitoring_frequency==1:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    best_MC_avg_total_reward = max(best_MC_avg_total_reward, MC_tr)
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.2f}'.format(np.log10(episode_cum_reward)), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          ", loss ", '{:6.2f}'.format(current_loss),
                          sep='')
                    
                    if best_MC_avg_total_reward == MC_tr:
                        print('Best model, saving...')
                        self.save()
                    
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.2f}'.format(np.log10(episode_cum_reward)), 
                          ", loss ", '{:6.2f}'.format(current_loss),
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state    
    

if __name__ == "__main__":
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False, logscale=False), max_episode_steps=200
    )  # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.

    args = get_train_parser()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqn = ProjectAgent(FFModel(args.state_dim, args.action_dim, args.nlayers, args.nhid), args=args)
    dqn.fill_buffer(env)

    dqn.train(env)
    dqn.save()