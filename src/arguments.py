import argparse

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

# Behavioral Cloning Settings.
parser.add_argument('--max_frames', type=int, default=200000000)
parser.add_argument('--n_episodes_test', type=int, default=50)
parser.add_argument('--eval_frequency', type=int, default=200)
parser.add_argument('--to_env', type=str, default='HabitatImageNav-apartment_0')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--disable_save', action='store_true')
parser.add_argument('--essential_save_only', action='store_true')
parser.add_argument('--save_path', type=str, default='bc')
parser.add_argument('--data_path', type=str, default='behavioral_cloning')

# Embedding Settings.
parser.add_argument('--embedding_name', type=str, default='resnet50',
                    help='Name of the embedding model.')
parser.add_argument('--train_embedding', action='store_true',
                    help='Train observation embedding or keep it fixed.')
parser.add_argument('--disable_pretrained_embedding', action='store_false', dest='pretrained_embedding',
                    help='Use it to prevent loading torchvision pretrained weights.')
parser.add_argument('--batch_norm', action='store_true',
                    help='Place a BatchNorm1d layer at the beginning of the policy.')

# Environment Settings.
parser.add_argument('--env', type=str, default='HabitatImageNav-apartment_0',
                    help='Training environments. To enter multiple environments \
                    trained in parallel, add them as a comma-separated list.')
parser.add_argument('--num_input_frames', type=int, default=1,
                    help='Number of input frames per observation. \
                    When num_input_frames > 1, the environment will \
                    stack the previous num_input_frames - 1 frames to the current frame.')

# General Settings.
parser.add_argument('--xpid', default=None,
                    help='Experiment ID.')
parser.add_argument('--run_id', default=1, type=int,
                    help='Run ID used for running multiple instances of the same hyperparameters set \
                    (instead of a different random seed since torchbeast does not accept this).')
parser.add_argument('--seed', default=1, type=int,
                    help='Random seed.')

# Training settings.
parser.add_argument('--total_frames', default=50000000, type=int,
                    help='Total environment frames to train for.')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size.')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension).')
parser.add_argument('--mp_start', default='spawn', type=str,
                    help='Start method of multiprocesses. \
                    Depending on your machine, there can be problems between CUDA \
                    with some environments. To avoid them, use `spawn`.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')

# Optimizer settings.
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
