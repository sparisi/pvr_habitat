import logging
import os
import sys
import threading
import time

from copy import deepcopy
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.core import file_writer
from src.models import PolicyNet
from src.embeddings import EmbeddingNet
from src.gym_wrappers import make_gym_env
from src.utils import get_batch, log, create_buffers


def init_models_and_states(flags):
    """Initialize models and LSTM states for all algorithms."""
    torch.manual_seed(flags.run_id)
    torch.cuda.manual_seed(flags.run_id)
    np.random.seed(flags.run_id)

    # Set device
    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        log.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        log.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    # Init embedding
    embedding_model = EmbeddingNet(flags.embedding_name,
                                   in_channels=3,
                                   pretrained=flags.pretrained_embedding,
                                   train=flags.train_embedding)

    # Retrieve action_space and observation_space shapes
    env = make_gym_env(flags, embedding_model)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    env.close()

    # Init policy models
    actor_model = PolicyNet(obs_shape, n_actions)
    learner_model = PolicyNet(obs_shape, n_actions).to(device=flags.device)

    # Load models (if there is a checkpoint)
    if flags.checkpoint:
        log.info(' ... loading model from %s %s ', flags.checkpoint, '...')
        checkpoint = torch.load(flags.checkpoint)
        actor_model.load_state_dict(checkpoint["actor_model_state_dict"])
        learner_model = deepcopy(actor_model).to(device=flags.device)
        embedding_model.load_state_dict(checkpoint["embedding_model_state_dict"])

    # Actors will run across multiple processes
    actor_model.share_memory()
    embedding_model.share_memory()

    # Init LSTM states
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = actor_model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    # Init optimizers
    learner_model_optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    # LR scheduler
    def lr_lambda(epoch):
        x = np.maximum(flags.total_frames, 5e6)
        return 1 - min(epoch * flags.unroll_length * flags.batch_size, x) / x
    scheduler = torch.optim.lr_scheduler.LambdaLR(learner_model_optimizer, lr_lambda)

    # Buffer
    buffers = create_buffers(obs_shape, n_actions, flags)

    return dict(
        actor_model=actor_model,
        learner_model=learner_model,
        embedding_model=embedding_model,
        initial_agent_state_buffers=initial_agent_state_buffers,
        learner_model_optimizer=learner_model_optimizer,
        scheduler=scheduler,
        buffers=buffers,
        )
