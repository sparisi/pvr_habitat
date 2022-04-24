# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import typing
import gym
import threading
from torch import multiprocessing as mp
import logging
import traceback
import os
import numpy as np
from copy import deepcopy

from src.core import prof
from src.env_utils import make_environment


shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('torchbeast')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              agent_state_buffers,
              flags,
              timings,
              lock=threading.Lock()):
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')

    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[agent_state_buffers[m] for m in indices])
    )
    timings.time('batch')

    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')

    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in agent_state)
    timings.time('device')

    return batch, agent_state


def create_buffers(obs_shape, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        obs=dict(size=(T + 1, *obs_shape), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_success=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module, embedding_model: torch.nn.Module,
        buffers: Buffers, initial_agent_state_buffers, flags):
    try:
        timings = prof.Timings()

        env = make_environment(flags, embedding_model, i)

        log.info('Actor %i started on environment %s ...', i, flags.env)

        env_output = env.initial()

        agent_state = actor_model.initial_state(batch_size=1)
        agent_output, unused_state = actor_model(env_output, agent_state)

        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for key, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][key][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output, agent_state = actor_model(env_output, agent_state)

                timings.time('actor_model')

                env_output = env.step(agent_output['action'])

                timings.time('step')

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]

                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time('write')

            full_queue.put(index)

        if i == 0:
            log.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass

    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
