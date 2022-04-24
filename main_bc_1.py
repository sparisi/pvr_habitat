import os
import numpy as np
import torch
import itertools
import pickle
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.models import PolicyNet
from src.embeddings import EmbeddingNet
from src.env_utils import make_environment
from src.test_model import test
from src.arguments import parser
from src.utils_bc import (
    is_essential_save,
    sample_with_minimum_distance,
    read_habitat_data,
)


def run(flags):
    # Fix seeds
    torch.manual_seed(flags.run_id)
    torch.cuda.manual_seed(flags.run_id)
    np.random.seed(flags.run_id)
    random.seed(flags.run_id)

    if flags.debug:
        flags.n_episodes_test = np.minimum(2, flags.n_episodes_test)

    from_env = flags.env
    to_env = flags.to_env

    # Save paths
    base_path = flags.save_path
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path,
                from_env + '_em' + \
                flags.embedding_name + '_s' + \
                str(flags.run_id) + '_' + \
                to_env)

    # Quick check for resuming runs
    resume = False
    if os.path.isfile(save_path + '.pickle'):
        stats = pickle.load(open(save_path + '.pickle', 'rb'))
        if stats[to_env]['frames'][-1] >= flags.max_frames:
            print('   WARNING! This run was already completed. Stopping now.')
            return
        resume = True

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
                                   pretrained=True,
                                   train=False,
                                   disable_cuda=flags.disable_cuda)

    flags.env = to_env
    env = make_environment(flags, embedding_model)
    obs_shape = env.gym_env.observation_space.shape

    actor_model = PolicyNet(obs_shape, env.gym_env.action_space.n, flags.batch_norm).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        actor_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    max_epochs = flags.max_frames // (flags.unroll_length * flags.batch_size) + 1
    def lr_lambda(epoch):
        return 1 - epoch / max_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume old run
    if resume:
        checkpoint = torch.load(save_path + '.tar')
        embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])
        actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
        optimizer.load_state_dict(checkpoint["actor_model_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    test_model = PolicyNet(obs_shape, env.gym_env.action_space.n, flags.batch_norm).to(device=flags.device)
    test_model.load_state_dict(actor_model.state_dict())
    test_model.eval()

    print('=== BC run ===')
    print('  ', 'embedding:', flags.embedding_name)
    print('  ', 'training environment(s):', from_env)
    print('  ', 'testing environment(s):', to_env)
    if flags.debug:
        print('  ', 'RUNNING IN DEBUG MODE!')

    # Read data
    # If multiple environments are used, we read one environment at the time
    # and immediately pass frames through the embedding, in order to save space
    # (frames are not kept in memory).
    print('=== Loading trajectories ===')
    first = True
    for env_id in from_env.split(','):
        data = read_habitat_data(os.path.join(flags.data_path, env_id + '.pickle'))

        if flags.debug:
            n_samples_scene = flags.batch_size * flags.unroll_length
        else:
            n_samples_scene = data['obs'].shape[0]

        # The random embedding is not pre-trained, but randomly initialized
        # So its weights depend on the seed, and we need to pass the obs through it every time
        print('  ', 'passing observations through embedding model')
        n_frames = max(data['obs'].shape[3] // 3, 1)
        obs_scene = []
        for i in tqdm(range(0, n_samples_scene, flags.batch_size)): # To avoid OutOfMemory we loop through mini-batches
            o = data['obs'][i:i+flags.batch_size]
            if o.shape[-1] == 1: # grayscale (Atari)
                o = np.repeat(o, 3, -1)
            o = np.concatenate(np.split(o, n_frames, axis=3), axis=0) # (N, H, W, n_frames * 3) -> (N * n_frames, H, W, 3)
            o = embedding_model(torch.from_numpy(o)) # (N * n_frames, O)
            o = np.concatenate(np.split(o, n_frames, axis=0), axis=-1) # (N, O * n_frames)
            obs_scene.append(o)
        obs_scene = np.concatenate(obs_scene)[:n_samples_scene]

        if first:
            obs = np.array(obs_scene)
            action = data['action'][:n_samples_scene]
            reward = data['reward'][:n_samples_scene]
            done = data['done'][:n_samples_scene]
            first = False
        else:
            obs = np.concatenate((obs, obs_scene))
            action = np.concatenate((action, data['action'][:n_samples_scene]))
            reward = np.concatenate((reward, data['reward'][:n_samples_scene]))
            done = np.concatenate((done, data['done'][:n_samples_scene]))

    assert len(obs) == len(action) == len(reward) == len(done), 'data length does not match'
    n_samples = len(reward)
    assert n_samples > 0, 'no data found'
    print('  ', 'total number of samples', n_samples)

    del data # Free memory of data we do not need anymore

    stat_keys = ['episode_return', 'episode_success']

    if resume:
        print('=== Resuming previous run ===')
        stats = pickle.load(open(save_path + '.pickle', 'rb'))
        print('  ', 'frames', stats[to_env]['frames'][-1])
        print('  ', 'training loss', stats[to_env]['training_loss'][-1])
        print('  ', 'gradient norm', stats[to_env]['gradient_norm'][-1])
        for k in stat_keys:
            print('  ', k, stats[to_env][k][-1])
        init_frames = stats[to_env]['frames'][-1]
    else:
        print('=== Initial evaluation ===')
        stats = dict()
        stats.update({to_env: dict({**{k: [] for k in stat_keys}, \
                                    **{'frames': []}, \
                                    **{'training_loss': []}, \
                                    **{'gradient_norm': []}}) \
                     })
        test_model.load_state_dict(actor_model.state_dict())
        stats_ep = test(test_model, env, stat_keys, flags.n_episodes_test)
        for k in stat_keys:
            mu = np.mean(stats_ep[k])
            print('  ', k, mu)
            stats[to_env][k].append(mu)
        stats[to_env]['frames'].append(0)
        stats[to_env]['training_loss'].append(np.nan)
        stats[to_env]['gradient_norm'].append(np.nan)
        init_frames = 0

    print('=== Training policy ===')
    frames_range = range(init_frames,
                         flags.max_frames,
                         flags.batch_size * flags.unroll_length)
    for frames in tqdm(frames_range, desc='epoch'):
        epoch = frames // (flags.batch_size * flags.unroll_length)
        starting_i = sample_with_minimum_distance(n=n_samples, k=flags.batch_size, d=flags.unroll_length)

        # Prepare batches: each is composed of `unroll_length` consecutive samples (see IMPALA)
        o = []
        a = []
        d = []
        for i in starting_i:
            idx = np.mod(np.arange(i, i+flags.unroll_length), n_samples)
            o.append(obs[idx])
            a.append(action[idx])
            d.append(done[idx])
        o = np.stack(o, axis=1)
        a = np.stack(a, axis=1)
        d = np.stack(d, axis=1)
        o = torch.from_numpy(o).to(device=flags.device)
        a = torch.from_numpy(a).to(device=flags.device)
        d = torch.from_numpy(d).to(device=flags.device)

        input = dict(obs=o, done=d)
        agent_state = actor_model.initial_state(batch_size=flags.batch_size)
        agent_state = tuple(s.to(device=actor_model.device) for s in agent_state)
        output, agent_state = actor_model(input, agent_state)

        loss = F.nll_loss(
            F.log_softmax(torch.flatten(output['policy_logits'], 0, 1), dim=-1),
            target=torch.flatten(a, 0, 1).long(),
        )

        scheduler.step()
        optimizer.zero_grad()
        loss.backward()

        gradient_norm = 0.
        for p in actor_model.parameters():
            if p.grad is not None and p.requires_grad:
                gradient_norm += p.grad.detach().data.norm(2).item() ** 2
        gradient_norm = gradient_norm ** 0.5

        nn.utils.clip_grad_norm_(actor_model.parameters(), flags.max_grad_norm)
        optimizer.step()

        # Evaluation and stats
        if (epoch + 1) % flags.eval_frequency == 0:
            test_model.load_state_dict(actor_model.state_dict())

            if (flags.essential_save_only and is_essential_save(epoch, max_epochs, flags.eval_frequency)) or \
                    not flags.essential_save_only: # Save only data that will be used in errorbars
                stats_ep = test(test_model, env, stat_keys, flags.n_episodes_test)
                for k in stat_keys:
                    mu = np.mean(stats_ep[k])
                    print('  ', k, mu)
                    stats[to_env][k].append(mu)
            else: # Fill non-essential points with nan (we need data to have the correct length)
                for k in stat_keys:
                    stats[to_env][k].append(np.nan)

            stats[to_env]['frames'].append(frames)
            stats[to_env]['training_loss'].append(loss.item())
            stats[to_env]['gradient_norm'].append(gradient_norm)

            print('  ', 'frames', frames)
            print('  ', 'training loss', loss.item())
            print('  ', 'gradient norm', gradient_norm)

            if not flags.disable_save:
                pickle.dump(stats, open(save_path + '.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                torch.save({
                    'embedding_model_state_dict': embedding_model.state_dict(),
                    'actor_model_state_dict': actor_model.state_dict(),
                    'actor_model_optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'flags': vars(flags),
                }, save_path + '.tar')

    env.close()


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
