# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import numpy as np
import torch
import itertools
import pickle
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn
import random
import cv2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.embeddings import EmbeddingNet
from src.arguments import parser

parser.add_argument('--n_trajectories', type=int, default=-1)
parser.add_argument('--source', type=str, default='png', choices=['png', 'pickle'])


def read_habitat_data_from_pickle(data_path, n_trajectories=-1):
    print('loading %s ...' % data_path)

    data = pickle.load(open(data_path + '.pickle', 'rb'))
    if n_trajectories == -1:
        n_trajectories = len(data['reward'])

    # Merge trajectories
    data['obs'] = np.concatenate(data['obs'][:n_trajectories])
    data['action'] = np.concatenate(data['action'][:n_trajectories])
    data['reward'] = np.concatenate(data['reward'][:n_trajectories])
    data['done'] = np.concatenate(data['done'][:n_trajectories])
    data['true_state'] = np.concatenate(data['true_state'][:n_trajectories])

    n_samples = len(data['reward'])
    print('  ', '%d trajectories for a total of %d samples' % (n_trajectories, n_samples))
    print('  ', 'avg. return is', data['reward'].sum() / n_trajectories)

    return data


def read_habitat_data_from_png(data_path, model=None, n_trajectories=-1):
    print('loading %s ...' % data_path)
    data = dict(obs=[], action=[], reward=[], done=[], true_state=[], png=[])

    if n_trajectories == -1:
        # n_trajectories = len([f for f in os.listdir(data_path) if f.endswith('_0.png')]) # too slow
        n_trajectories = 100000

    # Merge trajectories
    for t in tqdm(range(n_trajectories)):
        try:
            tmp = pickle.load(open(os.path.join(data_path, str(t) + '.pickle'), 'rb'))
            for k in data.keys():
                try:
                    data[k].append(tmp[k])
                except:
                    pass
            goal = cv2.imread(os.path.join(data_path, str(t) + '_goal' + '.png'))
            if model is not None:
                goal = model(torch.from_numpy(goal[None,:])).reshape(-1,)
        except:
            break
        for s in range(500): # 500 is the max step per trajectory according to Habitat's YAML config
            try:
                obs = cv2.imread(os.path.join(data_path, str(t) + '_' + str(s) + '.png'))
                if model is not None:
                    obs = model(torch.from_numpy(obs[None,:])).reshape(-1,)
                data['obs'].append(np.concatenate((obs, goal), -1))
                data['png'] += [os.path.join(data_path, str(t) + '_' + str(s)) + '.png']
            except:
                break

    n_trajectories = t
    data['obs'] = np.stack(data['obs'])
    data['action'] = np.concatenate(data['action'])
    data['reward'] = np.concatenate(data['reward'])
    data['done'] = np.concatenate(data['done'])
    data['true_state'] = np.concatenate(data['true_state'])

    n_samples = len(data['reward'])
    print('  ', '%d trajectories for a total of %d samples' % (n_trajectories, n_samples))
    print('  ', 'avg. return is', data['reward'].sum() / n_trajectories)

    return data


def run(flags):
    save_name = os.path.join(flags.data_path,
                             flags.env + '_' +
                             flags.embedding_name + '.pickle')
    if os.path.isfile(save_name):
        return

    # Fix seeds
    torch.manual_seed(flags.run_id)
    torch.cuda.manual_seed(flags.run_id)
    np.random.seed(flags.run_id)
    random.seed(flags.run_id)

    # Device setup
    flags.device = None
    if torch.cuda.is_available() and not flags.disable_cuda:
        print('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        print('Not using CUDA.')
        flags.device = torch.device('cpu')

    # Init models, env, optimizer, ...
    embedding_model = EmbeddingNet(flags.embedding_name,
                                   in_channels=3,
                                   pretrained=flags.pretrained_embedding,
                                   train=flags.train_embedding,
                                   disable_cuda=flags.disable_cuda) # Always on GPU, unless CUDA is disabled

    # Save model that will be used in main_bc
    emb_path = os.path.join(flags.data_path, flags.embedding_name)
    if flags.embedding_name == 'random':
        emb_path += '_' + str(flags.run_id)
    torch.save({
        'embedding_model_state_dict': embedding_model.state_dict(),
    }, emb_path + '.tar')

    print('=== Loading trajectories ===')

    if flags.source == 'png':
        data = read_habitat_data_from_png(
            os.path.join(flags.data_path, flags.env),
            embedding_model,
            flags.n_trajectories
        )

    if flags.source == 'pickle':
        data = read_habitat_data_from_pickle(
            os.path.join(flags.data_path, flags.env)
        )

        print('  ', 'passing observations through embedding model')
        n_samples = data['obs'].shape[0]
        n_frames = max(data['obs'].shape[3] // 3, 1)
        obs_scene = []
        for i in tqdm(range(0, n_samples, flags.batch_size)): # To avoid OutOfMemory we loop through mini-batches
            o = data['obs'][i:i+flags.batch_size]
            o = np.concatenate(np.split(o, n_frames, axis=3), axis=0) # (N, H, W, n_frames * 3) -> (N * n_frames, H, W, 3)
            o = embedding_model(torch.from_numpy(o)) # (N * n_frames, O)
            o = np.concatenate(np.split(o, n_frames, axis=0), axis=-1) # (N, O * n_frames)
            obs_scene.append(o)
        obs_scene = np.concatenate(obs_scene)[:n_samples]

        obs = np.array(obs_scene)
        true_state = data['true_state'][:n_samples]
        action = data['action'][:n_samples]
        reward = data['reward'][:n_samples]
        done = data['done'][:n_samples]

        data = dict(obs=obs, action=action, reward=reward, done=done, true_state=true_state)

    n_samples = len(data['reward'])
    assert n_samples > 0, 'no data found'
    print('  ', 'total number of samples', n_samples)

    with open(save_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
