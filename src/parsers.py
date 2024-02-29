import argparse


def get_train_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--state_dim', type=int, default=6,  help='state dimension')
    parser.add_argument('--action_dim', type=int, default=4, help='action dimension')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    
    parser.add_argument('--epsilon_max', type=float, default=1.0, help='epsilon max')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='epsilon min')
    parser.add_argument('--epsilon_delay', type=int, default=20, help='epsilon delay')
    parser.add_argument('--epsilon_stop', type=int, default=1000, help='epsilon stop')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='weight decay')
    parser.add_argument('--nb_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    parser.add_argument('--buffer_size', type=int, default=1000000, help='replay buffer size')


    return parser.parse_args()