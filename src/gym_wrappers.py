import gym
try:
    from gym.wrappers.pixel_observation import PixelObservationWrapper
    import gym_minigrid
except:
    pass
from collections import deque
import numpy as np
import os
import pathlib
import re

from src.embeddings import EmbeddingWrapper

if 'VERBOSE_HABITAT' not in os.environ: # To suppress Habitat messages
    os.environ['MAGNUM_LOG'] = 'quiet'
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
else:
    os.environ['GLOG_minloglevel'] = '0'
    os.environ['MAGNUM_LOG'] = 'verbose'
    os.environ['MAGNUM_GPU_VALIDATION'] = 'ON'

try:
    import quaternion
    import habitat
    from habitat_baselines.config.default import get_config
    from habitat_baselines.common.environments import get_env_class
    from habitat_baselines.utils.env_utils import make_env_fn as make_env_habitat
    from habitat.utils.visualizations.utils import observations_to_image
    from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
except:
    pass

def make_gym_env(flags, embedding_model=None, seed=0):
    if 'Habitat' in flags.env:
        # Absolute path of gym_wrappers.py, used to retrieve the correct
        # path of config files
        abs_root = pathlib.Path(__file__).parent.resolve()

        config_file = os.path.join(abs_root, '..', 'habitat_config', 'nav_task.yaml')
        config = get_config(config_paths=config_file,
            opts=['BASE_TASK_CONFIG_PATH', config_file])

        config.defrost()

        # Absolute paths
        config.TASK_CONFIG.DATASET.DATA_PATH = \
            os.path.join(abs_root, '..', config.TASK_CONFIG.DATASET.DATA_PATH)
        config.TASK_CONFIG.DATASET.SCENES_DIR = \
            os.path.join(abs_root, '..', config.TASK_CONFIG.DATASET.SCENES_DIR)

        # Set Replica scene
        scene = flags.env.split('-')[1]
        assert len(scene) > 0, 'Undefined scene.'
        config.TASK_CONFIG.DATASET.SCENES_DIR += scene

        config.freeze()

        # Make env
        env_class = get_env_class(config.ENV_NAME)
        env = make_env_habitat(env_class=env_class, config=config)
        env = HabitatNavigationWrapper(env, scene,
                                       image_goal='ImageNav' in flags.env,
                                       true_state=flags.embedding_name == 'true_state')

    elif 'Atari' in flags.env:
        env = gym.make(flags.env.split('Atari-')[-1])
        env = AtariWrapper(env)

    elif 'MiniGrid' in flags.env:
        env = gym.make(flags.env)
        env = MiniGridWrapper(env)

    else:
        env = gym.make(flags.env)
        env = PixelObservationWrapper(env)
        env = DefaultWrapper(env)
        # TODO needs a screen to render, and I can't change height/width
        # TODO does not work with all gym envs

    env.seed(seed)

    if embedding_model is not None and flags.embedding_name != 'true_state':
        env = EmbeddingWrapper(env, embedding_model)

    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)

    return env


# ------------------------------------------------------------------------------
# Default wrapper
# ------------------------------------------------------------------------------


class DefaultWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['pixels']

    def reset(self):
        obs = self.env.reset()
        return np.asarray(obs['pixels'])

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        obs = np.asarray(obs['pixels'])
        info.update({'success': 0.})
        return obs, rwd, done, info

    def randomize(self):
        pass


# ------------------------------------------------------------------------------
# MiniGrid wrapper
# ------------------------------------------------------------------------------


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def reset(self):
        obs = self.env.reset()
        return np.asarray(obs['image'])

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        obs = np.asarray(obs['image'])
        info.update({'success': rwd > 0.})
        return obs, rwd, done, info

    def randomize(self):
        pass


# ------------------------------------------------------------------------------
# Habitat wrapper
# ------------------------------------------------------------------------------


def _sample_start_habitat(sim, target_position, number_retries=100, difficulty='random'):
    geodesic_to_euclid_ratio = {
        'easy': 0.8,
        'medium': 1.0,
        'hard': 1.2,
        'random': 1.1, # Habitat default
    }
    for _retry in range(number_retries):
        source_position = sim.sample_navigable_point()
        is_compatible, _ = is_compatible_episode(
            source_position,
            target_position,
            sim,
            near_dist=1,
            far_dist=30,
            geodesic_to_euclid_ratio=geodesic_to_euclid_ratio[difficulty],
        )
        if is_compatible:
            break
    if not is_compatible:
        raise ValueError('Cannot find a goal position.')
    return source_position


class HabitatNavigationWrapper(gym.Wrapper):
    def __init__(self, env, scene, image_goal=False, true_state=False):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)
        self.observation_space = self.env.observation_space['rgb']

        scene_to_id = {
            'apartment': 0.,
            'frl_apartment': 1.,
            'room': 2.,
            'office': 3.,
            'hotel': 4.,
        }
        scene_name, scene_version = re.split('_(\d+)', scene)[:2]
        self._scene_id = scene_to_id[scene_name]
        self._scene_version = float(scene_version)

        self._true_state = None # To store true state (agent pos, agent rot, goal pos)
        self.use_true_state = true_state

        self.image_goal = None
        if image_goal:
            shape = (self.observation_space.shape[0],
                     self.observation_space.shape[1],
                     self.observation_space.shape[2] * 2)
            self.observation_space = gym.spaces.Box(low=0.,
                                                    high=255.,
                                                    dtype=self.observation_space.dtype,
                                                    shape=shape)
            self.randomize_goal()

        if true_state:
            self.observation_space = gym.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    dtype=self.observation_space.dtype,
                                                    shape=self.get_true_state().shape)

    def get_true_state(self):
        agent_state = self.unwrapped._env.sim.get_agent_state()
        goal_position = self.unwrapped._env._dataset.episodes[0].goals[0].position
        true_state = np.concatenate((np.asarray(agent_state.position),
                                     quaternion.as_float_array(agent_state.rotation),
                                     np.asarray(goal_position),
                                     [self._scene_id],
                                     [self._scene_version]))
        return true_state

    def reset(self):
        obs = self.env.reset()
        obs = np.asarray(obs['rgb'])

        self._true_state = self.get_true_state()

        if self.use_true_state:
            obs = self._true_state
        elif self.image_goal is not None:
            obs = np.concatenate((obs, self.image_goal), axis=-1)

        return obs

    def step(self, action):
        obs, rwd, done, info = self.env.step(**{'action': action + 1})
        obs = np.asarray(obs['rgb'])
        rwd /= self.unwrapped._rl_config.SUCCESS_REWARD # Normalize rewards

        self._true_state = self.get_true_state()

        if self.use_true_state:
            obs = self._true_state
        elif self.image_goal is not None:
            obs = np.concatenate((obs, self.image_goal), axis=-1)

        return obs, rwd, done, info

    def render(self, mode='rgb_array'):
        pass

    def stop(self):
        self.env.close()

    def randomize_goal(self):
        random_location = self.unwrapped._env.sim.sample_navigable_point()
        random_heading = np.random.uniform(-np.pi, np.pi)
        random_rotation = [
            0,
            np.sin(random_heading / 2),
            0,
            np.cos(random_heading / 2),
        ]
        self.unwrapped._env._dataset.episodes[0].goals[0].position = random_location
        obs = self.unwrapped._env.sim.get_observations_at(random_location, random_rotation)
        self.image_goal = np.asarray(obs['rgb'])

    def randomize_start(self):
        random_location = _sample_start_habitat(self.unwrapped._env._sim,
                self.unwrapped._env._dataset.episodes[0].goals[0].position)
        random_heading = np.random.uniform(-np.pi, np.pi)
        random_rotation = [
            0,
            np.sin(random_heading / 2),
            0,
            np.cos(random_heading / 2),
        ]
        self.unwrapped._env._dataset.episodes[0].start_position = random_location
        self.unwrapped._env._dataset.episodes[0].start_rotation = random_rotation

    def randomize(self):
        ok = False
        while not ok:
            try:
                if self.image_goal is not None:
                    self.randomize_goal()
                self.randomize_start()
                ok = True
            except:
                pass


# ------------------------------------------------------------------------------
# Atari wrapper
# ------------------------------------------------------------------------------


class AtariWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_lives = self.env.unwrapped.ale.lives()
        self._lives = self._max_lives
        self._force_fire = None
        self._real_reset = True
        self._max_no_op_actions = 30
        self._current_no_op = None
        assert self.unwrapped.get_action_meanings()[0] == 'NOOP'

    def randomize(self):
        pass

    def reset(self):
        if self._real_reset:
            obs = self.env.reset()
            self._lives = self._max_lives

        self._force_fire = self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
        self._current_no_op = np.random.randint(self._max_no_op_actions + 1)
        return np.asarray(obs)

    def step(self, action):
        # Force FIRE action to start episodes in games with lives
        if self._force_fire:
            obs, _, _, _ = self.env.step(1)
            self._force_fire = False
        while self._current_no_op > 0:
            obs, _, _, _ = self.env.step(0)
            self._current_no_op -= 1

        obs, reward, absorbing, info = self.env.step(action)

        self._real_reset = absorbing
        if info['ale.lives'] != self._lives:
            self._lives = info['ale.lives']
            self._force_fire = self.env.unwrapped.get_action_meanings()[
                1] == 'FIRE'

        info.update({'success': 0.}) # TODO

        return np.asarray(obs), reward, absorbing, info

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def stop(self):
        self.env.close()
        self._real_reset = True


# ------------------------------------------------------------------------------
# Wrappers to stack frames
# ------------------------------------------------------------------------------


# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255.,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.asarray(LazyFrames(list(self.frames), self.k))


# https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/utils/frames.py
class LazyFrames(object):
    def __init__(self, frames, history_length):
        self._frames = frames
        assert len(self._frames) == history_length

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=-1)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def copy(self):
        return self

    @property
    def shape(self):
        return (len(self._frames),) + self._frames[0].shape
