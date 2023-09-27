"""
Author: Steve Paul 
Date: 6/17/22 """
import argparse
import torch
# Configuration file for the MRTA_Flood problem. (This configuration can only be used for the MRTA_Flood)

def get_config(args=None):

    parser = argparse.ArgumentParser(
        description="Graph Neural Network based reinforcement learning solution for MRTA-Flood")
    # Environment related
    parser.add_argument('--problem', type=str, default='MRTA_Flood', help='Name of the problem')
    parser.add_argument('--n_locations', type=int, default=51, help="Number of locations (including the depot if any).")
    parser.add_argument('--n_initial_tasks', type=int, default=41,
                        help="Number of initial tasks available. Used for problem with dynamic tasks.")
    parser.add_argument('--n_robots', type=int, default=5, help="Number of robots")
    parser.add_argument('--max_range', type=float, default=1.4, help='Maximum range for the robots')
    parser.add_argument('--max_capacity', type=int, default=5, help='Maximum capacity for the robots')
    parser.add_argument('--enable_dynamic_tasks', type=bool, default=False,
                        help="Boolean to enable dynamic tasks. The environment starts with n_initial_tasks, and"
                             "new tasks are introduced based on its start time")
    parser.add_argument('--enable_topological_features', type=bool, default=False,
                        help="Setting this as true, a graph laplacian based on topological features will be calculated"
                             "and used for the node embeddings")
    # Policy parameters
    parser.add_argument('--features_dim', type=int, default=128, help="Embedding length")
    parser.add_argument('--node_encoder', type=str, default='CAPAM',
                        help='Node embedding type. Available ones are [CAPAM, AM, MLP]')
    parser.add_argument('--K', type=int, default=3, help='K value for AM')
    parser.add_argument('--Le', type=int, default=2, help='Le value for CAPAM')
    parser.add_argument('--P', type=int, default=4, help='P value for CAPAM')
    parser.add_argument('--tda', type=bool, default=False, help='use topological data too. ')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--tanh_clipping', type=float, default=10, help='tanh clipping for logits')
    parser.add_argument('--mask_logits', type=bool, default=True, help='Masking enabled')
    parser.add_argument('--temp', type=float, default=1.00, help='softmax temp')

    # training algorithm parameters
    parser.add_argument('--total_steps', type=int, default=20000000, help='Total number of steps')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of steps for rollout')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--ent_coef', type=float, default=0.001, help='Entropy coefficient')
    parser.add_argument('--val_coef', type=float, default=0.5, help='Value coefficient')
    parser.add_argument('--gamma', type=float, default=1.00, help='Discount factor')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per rollout')

    parser.add_argument('--logger', type=str, default='Tensorboard_logger/', help='Directory for tensorboard logger')
    parser.add_argument('--model_save', type=str, default='Trained_Models/',
                        help='Directory for saving the trained models')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    config = parser.parse_args(args)
    config.use_cuda = torch.cuda.is_available() and not config.no_cuda

    return config
    #2000000