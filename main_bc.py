import builtins
import traceback
import signal
import os
import time
import datetime
import numpy as np
import torch
import pickle5 as pickle
import random
from collections import deque
from tqdm import tqdm
from copy import deepcopy
import warnings

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import nn
from torch import multiprocessing as mp
mp.set_sharing_strategy('file_system')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.embeddings import EmbeddingNet
from src.models import DiscretePolicy, ContinuousPolicy
from src.gym_wrappers import make_gym_env, LazyFrames
from src.testing import online_test, STOP
from src.arguments import parser


# Given `S` data points and a window `W`, the dataset is made of `N = ceil(S / W)`,
# points [t, t + D, t + 2*W, ..., t + N*W]. If `t + N*W > S`, the index is
# wrapped back to the start of the list. To ensure uniform sampling, `t` will be set
# randomly by `randomize()` at every epoch. Then, indices are shuffled by the sampler.
class SequentialWindowedDataset(Dataset):
    def __init__(self, size, window):
        self.size = size
        self.window = window
        self.data = np.arange(0, self.size, self.window)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def randomize(self, seed):
        rng = np.random.default_rng(seed)
        pad = rng.integers(0, self.size)
        self.data = (self.data + pad) % self.size


# To exit gracefully if runs/jobs are killed
def proc_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def is_essential_save(epoch, max_epochs):
    essential_progress = [0.01, 0.02, 0.03, 0.04, 0.05, # 0.01 stands for 1% of max_epochs
                          0.1, 0.2, 0.3, 0.4, 0.5,
                          0.6, 0.8, 1.0]
    # if `max_epoch * 0.01 = 1.5` then we eval for both epoch 1 and 2
    epochs_ceil = [np.ceil(e * max_epochs) for e in essential_progress]
    epochs_floor = [np.floor(e * max_epochs) for e in essential_progress]
    if (epoch in epochs_ceil) or (epoch in epochs_floor):
        return True
    return False


def run(flags):
    ### true_state is special case: it is not an embedding but it is passed with flags.embedding_name
    assert not (flags.train_embedding and flags.embedding_name == 'true_state'), \
        'Cannot train true_state embedding.'

    if flags.embedding_name != 'true_state' and not flags.batch_norm:
        warn = f'WARN: You are using a perception module without batch normalization.'
        yellow_warn = f"\033[33m{warn}\033[m"
        warnings.warn(
            yellow_warn)

    if flags.num_input_frames > 1 and not flags.disable_lstm:
        warn = f'WARN: You are stacking frames but still using a LSTM.'
        yellow_warn = f"\033[33m{warn}\033[m"
        warnings.warn(
            yellow_warn)

    ### Distributed settings
    if "WORLD_SIZE" in os.environ:
        flags.world_size = int(os.environ["WORLD_SIZE"])
    flags.distributed = flags.world_size > 1

    batch_size_split = int(np.ceil(flags.batch_size / flags.world_size))
    assert batch_size_split > 0, \
        'You requested too many nodes or GPUs for your batch size. \
        Be sure that the world size (= nodes * gpu_per_node) is smaller than the batch size.'

    if flags.distributed:
        if flags.local_rank != -1: # for torch.distributed.launch
            flags.rank = flags.local_rank
            flags.gpu = flags.local_rank
        elif 'SLURM_PROCID' in os.environ: # for Slurm scheduler
            flags.rank = int(os.environ['SLURM_PROCID'])
            flags.gpu = flags.rank % torch.cuda.device_count()

        torch.distributed.init_process_group(
            backend=flags.dist_backend,
            init_method=flags.dist_url,
            world_size=flags.world_size,
            rank=flags.rank,
            timeout=datetime.timedelta(seconds=180),
            )
    else: # for local runs
        flags.gpu = 0
        flags.rank = 0
        # if distributed, the seed is set in the worker (see submitit_bc.py)
        random.seed(flags.seed)
        torch.manual_seed(flags.seed)
        np.random.seed(flags.seed)
        torch.cuda.manual_seed(flags.seed)

    ### Save path
    if flags.xpid is None:
        flags.xpid = 'bc-%s' % (time.strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(flags.save_path):
        os.makedirs(flags.save_path, exist_ok=True)
    save_path = os.path.join(flags.save_path, flags.xpid)

    ### Quick check for resuming runs
    resume = False
    if os.path.isfile(save_path + '.pickle'):
        stats = pickle.load(open(save_path + '.pickle', 'rb'))
        if stats['epoch'][-1] >= flags.epochs:
            print('   WARNING! This run was already completed. Stopping now.')
            return
        resume = True

    ### Use first env to get raw obs and act shape
    env_list = flags.env.split(',')
    env = make_gym_env(env_list[0], flags)
    obs_shape = env.observation_space.shape
    act_space = env.action_space
    if env.action_space.shape == ():
        Policy = DiscretePolicy
    else:
        Policy = ContinuousPolicy
    env.close()

    ### Init embedding
    if flags.train_embedding: # the embedding is trained as part of the policy
        embedding_model = None
    else: # the embedding is part of the env as wrapper
        embedding_model = EmbeddingNet(
            flags.embedding_name, obs_shape,
            pretrained=True, train=False,
        ).to(device=flags.gpu)
        env = make_gym_env(env_list[0], flags, embedding_model)
        obs_shape = env.observation_space.shape # obs_shape now takes the embedding into account
        env.close()

    ### Init actor policy
    actor_model = Policy(
        obs_shape,
        act_space,
        flags.embedding_name if flags.train_embedding else None,
        use_lstm=flags.num_input_frames == 1,
        batch_norm=flags.batch_norm,
        embedding_pretrained=flags.pretrained_embedding,
        embedding_train=flags.train_embedding,
    ).to(device=flags.gpu)

    if flags.distributed:
        torch.cuda.set_device(flags.gpu)
        actor_model.cuda(flags.gpu)
        actor_model = torch.nn.parallel.DistributedDataParallel(actor_model, device_ids=[flags.gpu])
        actor_module = actor_model.module
    else: # actor_model is a pointer for copying the actor parameters into test_model
        actor_module = actor_model

    ### Init optimizer and scheduler
    optimizer = torch.optim.RMSprop(
        actor_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch): # decaying learning rate
        return 1 - epoch / flags.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ### Resume old run
    if resume:
        loc = 'cuda:{}'.format(flags.gpu)
        checkpoint = torch.load(save_path + '.tar', map_location=loc)
        actor_module.load_state_dict(checkpoint["actor_model_state_dict"])
        optimizer.load_state_dict(checkpoint["actor_model_optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        agent_state = checkpoint["agent_state"]

    ### Make test model and copy actor model parameters
    test_model = deepcopy(actor_model).to(device=flags.gpu)
    test_model.eval()
    for p in test_model.parameters():
        p.requires_grad = False # to reduce memory usage
    test_model.share_memory()

    ### Recap print
    print('======== Run Recap ========')
    print(f"  embedding: {flags.embedding_name} ({'finetuned' if flags.train_embedding else 'frozen'})")
    print(f"  training environment(s):")
    for env_id in env_list:
        print('     ', env_id)
    print(f'  world_size: {flags.world_size}')
    print(f'  rank: {flags.rank}')
    print(f'  gpu: {flags.gpu}')
    print(f'  batch_size per gpu: {batch_size_split}')
    if flags.debug:
        print(f'  *** debug mode is active ***')
    print('===========================')

    ### Prepare test process
    # Testing runs in online_testing() at every epoch. In the meantime, training
    # proceeds for the next epoch without having to wait for testing to be done.
    if flags.rank == 0: # only master tests
        flags.num_episodes_test = 2 if flags.debug else flags.num_episodes_test
        mp.set_start_method('spawn') # if we do not use 'spawn' Habitat cannot init
        test_manager = mp.Manager()
        test_queue = test_manager.JoinableQueue() # to communicate between parent and children processes
        shared_stats = test_manager.dict() # will be shared and populated by the test processes
        test_pool = mp.Pool(len(env_list), initializer=proc_initializer)
        for env_id in env_list: # spawn one child process for every env to be tested
            test_pool.apply_async(online_test,
                args=(test_queue, test_model, embedding_model, shared_stats, flags),
            )

    try: # wrap all in a try-except to handle KeyboardInterrupt

        ### Read data
        # Pickles have list of trajectories, each with data stored as np.array,
        # data = {'obs': [np.array, np.array, ...], 'act': [np.array, np.array, ...]}
        n_traj = 20 if flags.debug else flags.num_trajectories_train
        if flags.train_embedding or flags.embedding_name == 'true_state':
            data_suffix = '' # load raw trajectories
        else:
            data_suffix = '_' + flags.embedding_name # load embedded trajectories
        data_path_list = [os.path.join(flags.data_path, env_id + data_suffix + '.pickle')
                            for env_id in env_list]
        data = [pickle.load(open(data_path, 'rb'))
                                for data_path in tqdm(data_path_list, desc='loading pickle')]
        print('... merging trajectories, please wait ... ')
        obs_key = 'true_state' if flags.embedding_name == 'true_state' else 'obs'
        obs = np.concatenate(sum([d[obs_key][:n_traj] for d in data], [])) # merge trajectories
        action = np.concatenate(sum([d['action'][:n_traj] for d in data], []))
        done = np.concatenate(sum([d['done'][:n_traj] for d in data], []))

        ### Stack observations
        if flags.num_input_frames > 1:
            stacked_obs = np.empty((*obs.shape[:-1], obs.shape[-1] * flags.num_input_frames))
            frames = deque([], maxlen=flags.num_input_frames)
            reset = True
            for i in tqdm(range(len(obs)), desc='stacking obs'):
                if reset:
                    for j in range(flags.num_input_frames):
                        frames.append(obs[i])
                frames.append(obs[i])
                stacked_obs[i] = np.asarray(LazyFrames(frames, flags.num_input_frames, flags.frame_stack_mode))
                reset = done[i]
            obs = stacked_obs

        assert len(obs) == len(action) == len(done), 'data length does not match'
        n_samples = len(obs)
        assert n_samples > 0, 'no data found'
        print(f'total number of samples: {n_samples}')
        del data # free memory we do not need anymore

        ### Init stats dictionary
        if resume:
            stats = pickle.load(open(save_path + '.pickle', 'rb'))
            init_epoch = int(stats['epoch'][np.isfinite(stats['epoch'])][-1])
            print(f'... resuming previous run from epoch {init_epoch}')
        else:
            stats = {**{k: np.full(flags.epochs + 1, np.nan) for k in ['epoch', 'training_loss', 'gradient_norm']},
                     **{env_id: {k: np.full(flags.epochs + 1, np.nan) for k in ['episode_return', 'episode_success']} for env_id in env_list}}
            init_epoch = 0
            agent_state = actor_module.initial_state(batch_size=n_samples)
            stats['epoch'][init_epoch] = 0

        ### Suppress printing if not on master (if distributed)
        if flags.rank != 0:
            print(f'=== training has strted for rank {flags.rank}, but it is not master and will not print anymore.')
            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass

        ### Prepare dataset and loader
        train_dataset = SequentialWindowedDataset(n_samples, flags.unroll_length)
        train_sampler = DistributedSampler(train_dataset) if flags.distributed else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size_split,
                shuffle=(train_sampler is None), num_workers=flags.cpus_per_task,
                pin_memory=True, sampler=train_sampler)

        ### Function to save stats and models
        def checkpoint():
            pickle.dump(stats, open(save_path + '.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            torch.save({
                'actor_model_state_dict': actor_module.state_dict(),
                'actor_model_optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'agent_state': agent_state,
                'flags': vars(flags),
            }, save_path + '.tar')

        ### Start training and testing in parallel
        if flags.rank == 0:
            for env_id in env_list:
                test_queue.put((init_epoch, env_id))
        eval_epoch = init_epoch # keep track of what epoch is being evaluated
        print('======== Training =========')
        for epoch in tqdm(range(init_epoch, flags.epochs), initial=init_epoch, total=flags.epochs, desc='epoch', mininterval=5):
            # Set data and sampler seed based on epoch and global seed for reproducibility
            train_dataset.randomize(flags.seed + epoch)
            if flags.distributed:
                train_sampler.set_epoch(epoch)

            # Save avg gradient norm and loss to check how training goes
            gradient_norm = 0.
            loss = 0.

            for i_start in tqdm(train_loader, desc='batch', disable=True):
                # Each batch has consecutive samples for the LSTM with indices
                # [t, t+1, ..., t + flags.unroll_length]. If t + flags.unroll_length > n_samples,
                # we wrap around and go back to the start with np.mod().
                # This works as long as the last sample has `done = True`,
                # since `done = True` resets the LSTM state (see `models.py`).
                batch = np.array([np.mod(np.arange(i, i + flags.unroll_length), n_samples) for i in i_start]).T
                obs_batch = torch.from_numpy(obs[batch]).to(flags.gpu)
                action_batch = torch.from_numpy(action[batch]).to(flags.gpu)
                done_batch = torch.from_numpy(done[batch]).to(flags.gpu)

                # Prediction with agent state at step t
                actor_output, new_agent_state = actor_model(
                    dict(obs=obs_batch, done=done_batch),
                    tuple(agent_state[i][:, batch[0], :] for i in range(2)), # this will be ignored it the policy does not have a LSTM
                    full_output=True
                )

                # Update the LSTM state at step t + flags.unroll_length + 1
                for i in range(2):
                    agent_state[i][:, np.mod(batch[-1] + flags.unroll_length, n_samples), :] = \
                        new_agent_state[i].detach().to(device=agent_state[i].device)

                # Loss
                loss_batch = - actor_model.log_probs(actor_output, action_batch).mean()
                optimizer.zero_grad()
                loss_batch.backward()

                # Update running gradient norm and loss
                gradient_norm_batch = 0.
                for p in actor_model.parameters():
                    if p.grad is not None and p.requires_grad:
                        gradient_norm_batch += p.grad.detach().data.norm(2).item() ** 2
                gradient_norm += gradient_norm_batch ** 0.5 / len(train_loader)
                loss += loss_batch.item() / len(train_loader)

                # Clip gradient and do update
                nn.utils.clip_grad_norm_(actor_model.parameters(), flags.max_grad_norm)
                optimizer.step()

            # Decay learning rate
            scheduler.step()

            # Evaluation (if distributed, only master evaluates and saves stats)
            if is_essential_save(epoch + 1, flags.epochs) and flags.rank == 0:
                test_queue.join() # wait for previous eval to be done
                test_model.load_state_dict(actor_module.state_dict())
                for env_id in env_list:
                    for k, v in shared_stats[env_id].items():
                        stats[env_id][k][eval_epoch] = np.mean(v) # first copy stats here
                    test_queue.put((epoch + 1, env_id)) # then launch new eval
                eval_epoch = epoch + 1

            # Print basic info
            stats['epoch'][epoch+1] = epoch + 1
            stats['training_loss'][epoch+1] = loss
            stats['gradient_norm'][epoch+1] = gradient_norm
            print(f'   ### {epoch + 1} | loss {loss}, norm {gradient_norm}')

            # Save model and stats (only master)
            if not flags.disable_save and flags.rank == 0:
                checkpoint()

    except:
        traceback.print_exc()
        if flags.rank == 0:
            test_pool.terminate()

    else:
        if flags.rank == 0:
            test_queue.join() # be sure that last eval was done
            for env_id in env_list:
                test_queue.put((-1, STOP)) # notify all procs to end
                for k, v in shared_stats[env_id].items():
                    stats[env_id][k][eval_epoch] = np.mean(v) # copy last epoch eval
            if not flags.disable_save: # save latest model and stats
                checkpoint()
            test_pool.close()

    if flags.rank == 0:
        test_pool.join()

    print('=========== End ===========')


if __name__ == '__main__':
    flags = parser.parse_args()
    run(flags)
