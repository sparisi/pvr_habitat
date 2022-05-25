import argparse

parser = argparse.ArgumentParser(description='PyTorch Training Agent')

# DistributedDataParallel
parser.add_argument('--world_size', default=1, type=int,
                    help='Number of jobs running in parallel (= nodes * gpu_per_node).')
parser.add_argument('--dist_url', default='env://', type=str,
                    help='URL used to set up distributed training.')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='Distributed backend.')
parser.add_argument('--rank', default=-1, type=int,
                    help='Node rank for distributed training.')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='Local rank for distributed training.')
parser.add_argument('--cpus_per_task', default=0, type=int)

# Behavioral Cloning
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs.')
parser.add_argument('--num_trajectories_train', type=int, default=None,
                    help='How many trajectory per environment are used to learn. \
                    If not specified, all will be used.')
parser.add_argument('--num_episodes_test', type=int, default=50,
                    help='Number of episodes to test the policy.')
parser.add_argument('--debug', action='store_true',
                    help='If True, we use only few trajectories to learn, \
                    and only 2 episodes to test.')
parser.add_argument('--disable_save', action='store_true',
                    help='Results and policy model are not saved.')
parser.add_argument('--save_path', type=str, default='results',
                    help='Where to save results and policy model.')
parser.add_argument('--data_path', type=str, default='/checkpoint/sparisi/habitat_pickles/',
                    help='Where optimal trajectories are stored.')

# Embedding
parser.add_argument('--embedding_name', type=str, default='random5',
                    help='Name of the embedding model.')
parser.add_argument('--train_embedding', action='store_true',
                    help='Train the embedding or keep it frozen.')
parser.add_argument('--disable_pretrained_embedding', action='store_false', dest='pretrained_embedding',
                    help='Use it to prevent loading torchvision pretrained weights.')

# Environment
parser.add_argument('--env', type=str, default='HabitatImageNav-apartment_0',
                    help='Training environments. To enter multiple environments \
                    trained in parallel, add them as a comma-separated list. \
                    If you do, be sure that environments have the same observation \
                    and action spaces.')
parser.add_argument('--num_input_frames', type=int, default=1,
                    help='Number of input frames per observation. \
                    When > 1, the environment will stack the previous \
                    num_input_frames - 1 frames to the current \
                    frame according to argument `frame_stack_mode`. \
                    Also, if > 1 the policy will not use the LSTM layer.')
parser.add_argument('--frame_stack_mode', type=str, default='cat', choices=['cat', 'diff'],
                    help='How frames are stacked. If `cat`, frames are concatenated. \
                    If `diff` the differences between consecutive frames are concatenated.')

# General
parser.add_argument('--xpid', default=None,
                    help='Experiment id. If not specified, the current date and time is used.')
parser.add_argument('--seed', default=1, type=int,
                    help='Random seed.')

# Training
parser.add_argument('--batch_size', default=16, type=int,
                    help='Gradient batch size.')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length of each batch (time dimension).')
parser.add_argument('--batch_norm', action='store_true',
                    help='Place a BatchNorm1d layer at the beginning of the \
                    control layers of the policy (after the perception module). \
                    Should always be used unless there is no perception (i.e., \
                    if we use true state vector observations).')

# Optimizer
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon.')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients.')
