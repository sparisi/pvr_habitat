"""
- Gym and MuJoCo: pass its usual environment name.
- Gym MiniGrid: pass its usual environment name.
- Habitat Image-Goal Navigation: pass `HabitatImageNav-<scene_id>`.
- Hand Manipulation Suite: pass `HMS-<env_name>`.
- DeepMind Control: pass `DMC-<env_name>-<task_name>`.
- Atari: pass `Atari-<game_name>`.

Examples: `Pendulum-v1`, `HalfCheetah-v3`, `MiniGrid-DoorKey-8x8-v0`,
`HabitatImageNav-apartment_0`, `HMS-pen-v0`, `DMC-cartpole-swingup`,
`Atari-BreakoutDeterministic-v4`.

They all return pixel observations. Except for MinGrid and Atari, all also provide
access to default low-dimensional vector observations by calling `env._true_state`.

Some also provide the `success` info at every step. This is then used to calculate
the episode success in `testing.py`.
"""

from collections import deque
import numpy as np
import os
import pathlib
import re
import gym
import quaternion

if 'VERBOSE_HABITAT' not in os.environ: # To suppress Habitat messages
    os.environ['GLOG_minloglevel'] = '2'
    os.environ['MAGNUM_LOG'] = 'quiet'
    os.environ['HABITAT_SIM_LOG'] = 'quiet'
else:
    os.environ['GLOG_minloglevel'] = '0'
    os.environ['MAGNUM_LOG'] = 'verbose'
    os.environ['MAGNUM_GPU_VALIDATION'] = 'ON'

from src.embeddings import EmbeddingWrapper


def make_gym_env(env_name, flags, embedding_model=None):
    # In some envs, pixels obs are optional and make steps slower, so use them only if needed
    use_env_embedding = embedding_model is not None
    use_pixels = flags.train_embedding or \
        use_env_embedding and embedding_model.embedding_name != 'true_state'

    if 'Habitat' in env_name:
        from habitat_baselines.config.default import get_config
        from habitat_baselines.common.environments import get_env_class
        from habitat_baselines.utils.env_utils import make_env_fn

        # Absolute path of gym_wrappers.py, used to retrieve the correct
        # path of config files
        abs_root = pathlib.Path(__file__).parent.resolve()

        config_file = os.path.join(abs_root, '..', 'habitat_config', 'nav_task.yaml')
        config = get_config(config_paths=config_file,
            opts=['BASE_TASK_CONFIG_PATH', config_file])

        config.defrost()

        # Update paths
        config.TASK_CONFIG.DATASET.DATA_PATH = \
            os.path.join(abs_root, '..', config.TASK_CONFIG.DATASET.DATA_PATH)
        config.TASK_CONFIG.DATASET.SCENES_DIR = \
            os.path.join(abs_root, '..', config.TASK_CONFIG.DATASET.SCENES_DIR)

        # Set Replica scene
        scene = env_name.split('-')[1]
        assert len(scene) > 0, 'Undefined scene.'
        config.TASK_CONFIG.DATASET.SCENES_DIR += scene

        config.freeze()

        # Make env
        env_class = get_env_class(config.ENV_NAME)
        env = make_env_fn(env_class=env_class, config=config)
        env = HabitatNavigationWrapper(env, scene)

    elif 'Atari' in env_name:
        env = gym.make(env_name.split('Atari-')[-1])
        env = AtariWrapper(env)

    elif 'MiniGrid' in env_name:
        import gym_minigrid
        env = gym.make(env_name) # (default) 7x7x3 partial obs
        # env = gym_minigrid.wrappers.FullyObsWrapper(env) # (optional) WxHx3 full obs, size depends on the grid
        # env = gym_minigrid.wrappers.RGBImgObsWrapper(env) # (optional) RGB-like full obs
        env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env) # (optional) RGB-like partial obs
        env = gym_minigrid.wrappers.ImgObsWrapper(env) # (mandatory) removes the 'mission' field

    elif 'HMS' in env_name:
        import mj_envs
        env = gym.make(env_name.split('HMS-')[-1])
        if use_pixels:
            env = HMSPixelObservationWrapper(env)
        else:
            use_env_embedding = False
        env = HMSWrapper(env)

    elif 'DMC' in env_name:
        import dmc2gym
        dmc_env, dmc_task = env_name.split('DMC-')[-1].split('-')
        env = dmc2gym.make(dmc_env, dmc_task, seed=flags.seed, from_pixels=use_pixels,
            visualize_reward=False, channels_first=False)
        if not use_pixels:
            use_env_embedding = False
        env = DMCWrapper(env)

    else: # basic gym and mujoco envs, like Pendulum, HalfCheetah, ...
        env = gym.make(env_name)
        if use_pixels:
            env.reset() # need to reset once or pixel wrapper will fail (not needed for MuJoCo)
            env = gym.wrappers.pixel_observation.PixelObservationWrapper(env, pixels_only=False)
            env = GymWrapper(env)
        else:
            use_env_embedding = False

    env.seed(flags.seed)

    if use_env_embedding:
        env = EmbeddingWrapper(env, embedding_model)

    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames, flags.frame_stack_mode)

    return env


# ------------------------------------------------------------------------------
# Gym and MuJoCo
# ------------------------------------------------------------------------------
class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['pixels']
        self._true_state = env.observation_space.spaces['state'].sample()

    def reset(self):
        obs = self.env.reset()
        self._true_state = np.asarray(obs['state'])
        return np.asarray(obs['pixels'])

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        self._true_state = np.asarray(obs['state'])
        obs = np.asarray(obs['pixels'])
        return obs, 1. * rwd, done, info


# ------------------------------------------------------------------------------
# DeepMind Control
# ------------------------------------------------------------------------------
class DMCWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._true_state = env.unwrapped._state_space.sample()

    def reset(self):
        obs = self.env.reset()
        self._true_state = self.env.unwrapped.current_state
        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high,
            dtype=self.action_space.dtype) # dmc2gym does not automatically clip the action and would raise an error
        obs, rwd, done, info = self.env.step(action)
        self._true_state = self.env.unwrapped.current_state
        return obs, 1. * rwd, done, info


# ------------------------------------------------------------------------------
# Hand Manipulation Suite
# ------------------------------------------------------------------------------
class HMSWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        info['success'] = info.pop('goal_achieved') # rename for consistency
        return obs, rwd * 1., done, info

    @property
    def _true_state(self):
        return self.env.unwrapped.get_obs()

class HMSPixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=256, height=256, camera_name='vil_camera', camera_id=0, depth=False):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0.,
            high=255.,
            shape=(width, height, 3),
            dtype=env.observation_space.dtype
        )
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.camera_id = camera_id

    def get_image(self):
        img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                camera_name=self.camera_name, device_id=self.camera_id)
        return img[::-1,:,:]

    def observation(self, observation):
        return self.get_image()


# ------------------------------------------------------------------------------
# Habitat
# ------------------------------------------------------------------------------
def _sample_start_habitat(sim, target_position, number_retries=100, difficulty='random'):
    from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode
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
        raise ValueError('Cannot find a starting position.')
    return source_position

class HabitatNavigationWrapper(gym.Wrapper):
    def __init__(self, env, scene):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)
        shp = self.env.observation_space['rgb'].shape
        shp = (shp[0], shp[1], shp[2] * 2) # x2 because obs has 2 frames: current view and goal view
        self.observation_space = gym.spaces.Box(
            low=0.,
            high=255.,
            shape=shp,
            dtype=self.env.observation_space['rgb'].dtype,
        )

        # For _true_state
        all_scenes = ['apartment', 'frl_apartment', 'room', 'office', 'hotel']
        scene_name, scene_version = re.split('_(\d+)', scene)[:2]
        self._scene_id = float(all_scenes.index(scene_name))
        self._scene_version = float(scene_version)

        self._image_goal = None
        self._randomize_goal()

    @property
    def _true_state(self):
        agent_state = self.unwrapped._env.sim.get_agent_state()
        goal_position = self.unwrapped._env._dataset.episodes[0].goals[0].position
        return np.concatenate((
            np.asarray(agent_state.position),
            quaternion.as_float_array(agent_state.rotation),
            np.asarray(goal_position),
            [self._scene_id],
            [self._scene_version],
        ))

    def reset(self):
        self._randomize()
        obs = self.env.reset()
        obs = np.asarray(obs['rgb'])
        obs = np.concatenate((obs, self._image_goal), axis=-1)
        return obs

    def step(self, action):
        obs, rwd, done, info = self.env.step(**{'action': action + 1})
        obs = np.asarray(obs['rgb'])
        obs = np.concatenate((obs, self._image_goal), axis=-1)
        return obs, 1. * rwd, done, info

    def render(self, mode='rgb_array'):
        pass

    def stop(self):
        self.env.close()

    def _randomize_goal(self):
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
        self._image_goal = np.asarray(obs['rgb'])

    def _randomize_start(self):
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

    def _randomize(self):
        for _ in range(100):
            try:
                self._randomize_goal()
                self._randomize_start()
                return
            except:
                pass
        raise ValueError('Cannot randomize Habitat environment.')


# ------------------------------------------------------------------------------
# Atari
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
        if info['lives'] != self._lives:
            self._lives = info['lives']
            self._force_fire = self.env.unwrapped.get_action_meanings()[
                1] == 'FIRE'

        return np.asarray(obs), 1. * reward, absorbing, info

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
    def __init__(self, env, k, mode='cat'):
        assert mode in ['cat', 'diff']
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.mode = mode
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255.,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype
        )

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
        return np.asarray(LazyFrames(list(self.frames), self.k, self.mode))

# https://github.com/MushroomRL/mushroom-rl/blob/dev/mushroom_rl/utils/frames.py
# 'diff' mode is from https://arxiv.org/pdf/2101.01857.pdf
class LazyFrames(object):
    def __init__(self, frames, history_length, mode):
        self._frames = frames
        assert len(self._frames) == history_length
        assert mode in ['cat', 'diff']
        self._mode = mode

    def __array__(self, dtype=None):
        if self._mode == 'cat':
            out = np.concatenate(self._frames, axis=-1)
        elif self._mode == 'diff': # [frames[1] - frames[0], frames[2] - frames[1], ..., frames[-1]]
            out = np.concatenate(np.diff(self._frames, axis=0, append=2*self._frames[-1][None]), axis=-1)
        else:
            raise NotImplementedError(f"Unknown frame stack mode: {self._mode}.")
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def copy(self):
        return self

    @property
    def shape(self):
        return (len(self._frames),) + self._frames[0].shape
