"""
This script saves optimal trajectories using Habitat's native solver.
Data is saved as `.pickle`: it takes a lot of space, but it is fast to load.
"""

import os
import numpy as np
import pickle5 as pickle
from tqdm import tqdm

from src.gym_wrappers import make_gym_env

from habitat.datasets.utils import get_action_shortest_path
from habitat_sim.errors import GreedyFollowerError

from src.arguments import parser
parser.add_argument('--n_trajectories', type=int, default=10000)


def get_shortest_path(env):
    '''
    Returns trajectory (obs, act, rwd, done, true_state) corresponding to the
    shortest path to the goal.
    `obs` shape is (H, W, 3) for PointNav (1 RGB frame) or (H, W, 6) for ImageNav (2 frames)
    where (H, W) are defined in habitat_config/nav_task.yaml, while
    `true_state = [agent_position, agent_orientation, goal_position, scene_id, scene_number]`
    (total size is 12).

    If the goal cannot be reached within the episode steps limit, the best
    path (closest to the goal) will be returned.

    '''
    max_steps = env.unwrapped._env._max_episode_steps
    try:
        shortest_path = [
            get_action_shortest_path(
                env.unwrapped._env._sim,
                source_position=env.unwrapped._env._dataset.episodes[0].start_position,
                source_rotation=env.unwrapped._env._dataset.episodes[0].start_rotation,
                goal_position=env.unwrapped._env._dataset.episodes[0].goals[0].position,
                success_distance=env.unwrapped._core_env_config.TASK.SUCCESS_DISTANCE,
                max_episode_steps=max_steps,
            )
        ][0]

        action = [p.action - 1 for p in shortest_path]

        shortest_steps = len(action)
        if shortest_steps == max_steps:
            print('WARNING! Shortest path not found with the given steps limit ({steps}).'.format(steps=max_steps),
                    'Returning best path.')
        else:
            print('Shortest path found: {steps} steps.'.format(steps=shortest_steps))

        # Get MDP-like trajectory to include reward
        obs = [env.reset()]
        reward = []
        done = []
        true_state = [env._true_state]
        for a in action:
            o, r, d, _ = env.step(a)
            obs.append(o)
            reward.append(r)
            done.append(d)
            true_state.append(env._true_state)

        return obs[:-1], action, reward, done, true_state[:-1]

    except GreedyFollowerError:
        print('WARNING! Cannot find shortest path (GreedyFollowerError).')
        return None, None, None, None, None


def gen_data_habitat(flags):
    flags.num_input_frames = 1
    flags.embedding_name = None

    env = make_gym_env(flags)

    obs = []
    action = []
    reward = []
    done = []
    true_state = []
    for trajectory in tqdm(range(flags.n_trajectories), desc='trajectory'):
        env.reset()
        o, a, r, d, s = get_shortest_path(env)
        obs.append(np.asarray(o))
        action.append(np.asarray(a))
        reward.append(np.asarray(r))
        done.append(np.asarray(d))
        true_state.append(np.asarray(s))

    data = dict(obs=obs, action=action, reward=reward, done=done, true_state=true_state)

    if not os.path.exists(flags.data_path):
        os.makedirs(flags.data_path, exist_ok=True)

    with open(os.path.join(flags.data_path, flags.env + '.pickle'), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    gen_data_habitat(flags)
