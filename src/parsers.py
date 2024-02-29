import argparse


def get_train_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--state_dim', type=int, default=6,  description='state dimension')
    parser.add_argument('--action_dim', type=int, default=4, description='action dimension')
    parser.add_argument('--gamma', type=float, default=0.95, description='discount factor')
    
    parser.add_argument('--epsilon_max', type=float, default=1.0, description='epsilon max')
    parser.add_argument('--epsilon_min', type=float, default=0.01, description='epsilon min')
    parser.add_argument('--epsilon_delay', type=int, default=20, description='epsilon delay')
    parser.add_argument('--epsilon_stop', type=int, default=1000, description='epsilon stop')
    
    parser.add_argument('--lr', type=float, default=0.0001, description='learning rate')
    parser.add_argument('--wd', type=float, default=0.01, description='weight decay')
    parser.add_argument('--nb_epoch', type=int, default=100, description='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, description='batch size')

    return parser.parse_args()