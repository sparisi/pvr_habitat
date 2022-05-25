"""
This script passes all images saved in `.pickle` files through embedding models.
Used when we keep the embedding frozen, since this way images pass through it only once.
"""

import os
import numpy as np
import torch
import pickle5 as pickle
from tqdm import tqdm
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.embeddings import EmbeddingNet
from src.arguments import parser

def run(flags):
    save_path = os.path.join(flags.data_path, flags.env + '_' +
                             flags.embedding_name + '.pickle')
    if os.path.isfile(save_path): # if the data already exists, skip it
        print(f'data already exists: {save_path}')
        return
    data_path = os.path.join(flags.data_path, flags.env + '.pickle')

    # Fix seeds
    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)

    # Device setup
    flags.device = None
    if torch.cuda.is_available():
        print('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        print('Not using CUDA.')
        flags.device = torch.device('cpu')

    # Load data with raw images
    data_path = os.path.join(flags.data_path, flags.env + '.pickle')
    data = pickle.load(open(data_path, 'rb'))
    n_frames = data['obs'][0].shape[-1] // 3
    obs_shape = data['obs'][0].shape[1:]

    # Init embedding
    embedding_model = EmbeddingNet(
        flags.embedding_name, obs_shape, pretrained=True, train=False
   ).to(device=flags.device)

    # Parse images through embedding model
    obs = []
    for frame in tqdm(data['obs'], desc='trajectories'):
        o = np.concatenate(np.split(frame, n_frames, axis=3), axis=0) # (N, H, W, n_frames * 3) -> (N * n_frames, H, W, 3)
        o = embedding_model(torch.from_numpy(o).to(device=flags.device)).cpu().numpy() # (N * n_frames, O)
        o = np.concatenate(np.split(o, n_frames, axis=0), axis=-1) # (N, O * n_frames)
        obs.append(o)

    # Replace raw images with parsed embeddings and save new data
    data.update({'obs': obs})
    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
