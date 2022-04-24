# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image

from src.gym_wrappers import make_gym_env

from habitat.datasets.utils import get_action_shortest_path
from habitat_sim.errors import GreedyFollowerError

from save_opt_trajectories import get_shortest_path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_trajectories', type=int, default=20000)
parser.add_argument('--env', type=str, default='HabitatPointNav-apartment_0')
parser.add_argument('--save_path', type=str, default='/checkpoint/sparisi/habitat_frames/')
parser.add_argument('--frameskip', type=int, default=3)


def gen_data_habitat(flags):
    flags.num_input_frames = 1
    flags.embedding_name = None

    save_path = os.path.join(flags.save_path, flags.env)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    env = make_gym_env(flags)

    for trajectory in tqdm(range(flags.n_trajectories), desc='trajectory'):
        env.randomize()
        env.reset()
        o, a, r, d, s = get_shortest_path(env)

        # Save agent's states
        true_state = np.asarray(s)[:, :8] # Take only agent position and orientation
        pickle_name = str(trajectory) + '.pickle'
        with open(os.path.join(save_path, pickle_name), 'wb') as handle:
            pickle.dump(true_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save frames
        for i in range(0, len(o), flags.frameskip):
            jpeg_name = str(trajectory) + '_' + str(i) + '.jpeg'
            image = Image.fromarray(o[i][:,:,:3]) # Take only current frame, ignore goal frame
            image.save(os.path.join(save_path, jpeg_name))

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    gen_data_habitat(flags)
