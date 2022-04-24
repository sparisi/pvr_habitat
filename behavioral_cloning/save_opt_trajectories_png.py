# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import pickle
import cv2
from tqdm import tqdm

from src.gym_wrappers import make_gym_env

from habitat.datasets.utils import get_action_shortest_path
from habitat_sim.errors import GreedyFollowerError

from save_opt_trajectories import get_shortest_path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_trajectories', type=int, default=10000)
parser.add_argument('--env', type=str, default='HabitatImageNav-apartment_0')
parser.add_argument('--save_path', type=str, default='behavioral_cloning')


def gen_data_habitat(flags):
    flags.num_input_frames = 1
    flags.embedding_name = None

    env = make_gym_env(flags)

    save_path = os.path.join(flags.save_path, flags.env)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for trajectory in tqdm(range(flags.n_trajectories), desc='trajectory'):
        env.randomize()
        env.reset()
        o, a, r, d, s = get_shortest_path(env)

        # Save obs frames
        for i in range(0, len(o)):
            # Save agent's view
            img_name = str(trajectory) + '_' + str(i) + '.png'
            cv2.imwrite(os.path.join(save_path, img_name), o[i][:,:,:3])
        try:
            # Save goal frame if ImageNav
            img_name = str(trajectory) + '_goal.png'
            cv2.imwrite(os.path.join(save_path, img_name), o[i][:,:,3:])
        except:
            # If PointNav
            pass

        data = dict(action=a, reward=r, done=d, true_state=s)
        with open(os.path.join(save_path, str(trajectory) + '.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    gen_data_habitat(flags)
