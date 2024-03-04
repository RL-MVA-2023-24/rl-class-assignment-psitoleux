import argparse


def get_train_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--state_dim', type=int, default=6,  help='state dimension')
    parser.add_argument('--action_dim', type=int, default=4, help='action dimension')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    
    parser.add_argument('--update_target_strategy', type=str, default='ema', choices=['replace', 'ema'], help='update target strategy')
    parser.add_argument('--update_target_freq', type=int, default=20, help='update target frequency')
    parser.add_argument('--update_target_tau', type=float, default=0.005, help='update target tau')
    
    parser.add_argument('--epsilon_max', type=float, default=1.0, help='epsilon max')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='epsilon min')
    parser.add_argument('--epsilon_delay', type=int, default=400, help='epsilon delay')
    parser.add_argument('--epsilon_stop', type=int, default=10000, help='epsilon stop')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--nb_epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--nb_gradient_steps', type=int, default=2, help='number of gradient steps')
    parser.add_argument('--criterion', type=str, default='mse', choices=['mse', 'l1'], help='criterion')
    
    parser.add_argument('--buffer_size', type=int, default=100000, help='replay buffer size')
    
    parser.add_argument('--nhid', type=int, default=128, help='hidden layer size')
    parser.add_argument('--nlayers', type=int, default=4, help='number of layers')
    parser.add_argument('--activation', type=str, default='relu', choices=['gelu', 'relu'], help='activation function')
    
    parser.add_argument('--monitoring_nb_trials', type=int, default=1, help='number of trials for monitoring')
    parser.add_argument('--monitoring_frequency', type=int, default=50, help='monitoring frequency')


    return parser.parse_args()