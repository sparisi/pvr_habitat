"""
This script saves optimal trajectories using Habitat's native solver.
Data is saved as `.pickle` except for observations, that are saved as `.png`.
Data take less space, but loading it is much slower.
The current training script does not support loading from `.png`.
"""

import os
import numpy as np
import pickle5 as pickle
from tqdm import tqdm
import cv2

from src.gym_wrappers import make_gym_env
from save_opt_trajectories import get_shortest_path

from src.arguments import parser
parser.add_argument('--n_trajectories', type=int, default=10000)
parser.add_argument('--frameskip', type=int, default=1,
                    help='Save only 1 frame every n. Used to make the dataset \
                    for pre-training in order to have more diverse images.')


def gen_data_habitat(flags):
    flags.num_input_frames = 1
    flags.embedding_name = None

    save_path = os.path.join(flags.data_path, flags.env)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    env = make_gym_env(flags)

    for trajectory in tqdm(range(flags.n_trajectories), desc='trajectory'):
        env.reset()
        o, a, r, d, s = get_shortest_path(env)

        for i in range(0, len(o), flags.frameskip): # Save agent's view
            img_name = str(trajectory) + '_' + str(i) + '.png'
            cv2.imwrite(os.path.join(save_path, img_name), o[i][:,:,:3])

        try: # Save goal frame if ImageNav
            img_name = str(trajectory) + '_goal.png'
            cv2.imwrite(os.path.join(save_path, img_name), o[0][:,:,3:])
        except: # If PointNav
            pass

        data = dict(action=a, reward=r, done=d, true_state=s) # Save (act, rwd, done, true_state)
        with open(os.path.join(save_path, str(trajectory) + '.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    gen_data_habitat(flags)
