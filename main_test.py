import os
import numpy as np
import torch
import pickle
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.models import PolicyNet
from src.embeddings import EmbeddingNet
from src.env_utils import make_environment
from src.test_model import test
from src.arguments import parser

parser.add_argument('--n_episodes_test', type=int, default=100)
parser.add_argument('--from_env', type=str, default='HabitatImageNav-apartment_0')
parser.add_argument('--logdir', type=str)

def run(flags):
    flags.device = torch.device('cpu')

    stat_keys = ['episode_return', 'episode_step', 'episode_success']

    # Either pass 'checkpoint' (path to model.tar), or logdir (and the model is found depending on other flags)
    if flags.checkpoint:
        checkpoint = torch.load(flags.checkpoint)
    else:
        ri = '-ri' + str(flags.run_id)
        model = '-m' + flags.model
        embedding_name = '-en' + flags.embedding_name
        env = flags.from_env
        runs = os.listdir(flags.logdir)
        found = False
        for run in runs:
            if ri in run and model in run and env in run and embedding_name in run:
                checkpoint = torch.load(os.path.join(flags.logdir, run, 'model.tar'))
                print('model found:', exp['model'], exp['env'], run)
                found = True
                break
        assert found, 'logdir passed, but model not found'

    embedding_model = EmbeddingNet(flags.embedding_name,
                                   in_channels=3,
                                   pretrained=flags.pretrained_embedding,
                                   train=flags.train_embedding)
    embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])

    env = make_environment(flags, embedding_model)

    model = PolicyNet(env.gym_env.observation_space.shape, env.gym_env.action_space.n)
    model.load_state_dict(checkpoint["actor_model_state_dict"])

    stats = test(model, env, stat_keys, flags.n_episodes_test)

    namefile = 'test_' + flags.embedding_name + \
               '_from_' + flags.from_env + \
               '_to_' + flags.env + \
               '_' + str(flags.run_id)
    with open(namefile + '.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # for k in stat_keys:
    #     print(k, stats[k])

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
