import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.embeddings import EmbeddingNet


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def logprobs_from_logits(logits, actions):
    return - F.nll_loss(
        F.log_softmax(logits.flatten(0, 1), dim=-1),
        actions.flatten(0, 1).long(),
        reduction='none'
    ).view_as(actions)

def logprobs_from_mean_and_logstd(mean, log_std, action):
    zs = (mean - action) / torch.exp(log_std)
    return - 0.5 * torch.sum(zs ** 2, dim=-1) + \
           - torch.sum(log_std, dim=-1) + \
           - 0.5 * action.shape[-1] * np.log(2 * np.pi)


class BasePolicy(nn.Module):
    """
    Base policy class. All policies have:
    - [Optional] Perception part (convolutional layers) passed as EmbeddingNet,
    - [Optional] Batch normalization layer,
    - Fully connected part (linear layers with non-linear activations),
    - [Optional] LSTM,
    - Output (linear layer).
    Number of layers, hidden sizes and activations depend on the policy.
    For instance, the discrete policy uses ReLU activations while the continuous
    policy uses Tanh.
    """

    def to(self, device): # Override `to` to keep track of the device.
        self.device = torch.device(device)
        if self.perception is not None:
            self.perception.to(device)
        return super().to(device)

    def initial_state(self, batch_size):
        if self.core is not None:
            return tuple(torch.zeros(self.core.num_layers, batch_size,
                                    self.core.hidden_size) for _ in range(2))
        else:
            return tuple(torch.zeros(1, batch_size, 1) for _ in range(2))

    def forward(self, env_state, core_state=()):
        x = env_state['obs'].to(device=self.device)

        # Original shape -> (unroll_length, batch_size, height, width, n_channels * n_frames)
        T, B, *_ = x.shape

        # Merge time and batch -> (unroll_length * batch_size, width, n_channels * n_frames)
        x = torch.flatten(x, 0, 1).float()

        if self.perception is not None:
            # Split stacked frames, pass them through the embedding one at the time, then stack them back
            x = torch.cat([self.perception(i) for i in torch.split(x, 3, -1)], -1)
            x = x.view(T * B, -1)

        if self.batch_norm is not None:
            is_training = self.batch_norm.training
            if x.shape[0] == 1: # cannot run BatchNorm1d on train() with only 1 sample
                self.batch_norm.eval()
            x = x.view(T, B, -1).permute(1, 2, 0) # BatchNorm1d input shape is (B, size, T) for sequences
            x = self.batch_norm(x)
            x = x.permute(2, 0, 1).flatten(0, 1) # restore shape to (T, B, size)
            if is_training: # restore training if we switched to eval for the 1-sample case
                self.batch_norm.train()

        core_input = self.fc(x)

        # LSTM pass
        if self.core is not None:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (1 - env_state['done'].float()).abs()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd.to(device=self.device) * s.to(device=self.device)
                                            for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
                core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        return core_output, core_state

    @abc.abstractmethod
    def log_probs(policy_output, actions):
        return

    @abc.abstractmethod
    def entropy(policy_output):
        return


class DiscretePolicy(BasePolicy):
    """ Softmax policy for discrete actions. """
    def __init__(self, observation_shape, action_space, embedding_name, use_lstm=True,
                    batch_norm=False, embedding_pretrained=True, embedding_train=False):
        super(DiscretePolicy, self).__init__()

        self.device = torch.device('cpu') # default

        num_actions = action_space.n

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # Perception module
        if embedding_name is not None:
            assert observation_shape[2] % 3 == 0, 'must use RGB observations'
            n_frames = observation_shape[2] // 3
            self.perception = EmbeddingNet(embedding_name, observation_shape,
                pretrained=embedding_pretrained, train=embedding_train)
            conv_out_size = self.perception.out_size
        else:
            n_frames = 1
            self.perception = None
            conv_out_size = observation_shape[0]

        # Add batch norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(conv_out_size * n_frames)
        else:
            self.batch_norm = None

        # Linear layers
        self.fc = nn.Sequential(
            init_(nn.Linear(conv_out_size * n_frames, 1024)),
            nn.ReLU(),
            init_(nn.Linear(1024, 1024)),
            nn.ReLU(),
        )

        # LSTM
        if use_lstm:
            self.core = nn.LSTM(1024, 1024, 2)
        else:
            self.core = None

        # Outputs
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.policy_logits = init_(nn.Linear(1024, num_actions))

    def forward(self, env_state, core_state=(), full_output=False):
        T, B, *_ = env_state['obs'].shape
        core_output, core_state = super().forward(env_state, core_state)

        policy_logits = self.policy_logits(core_output)
        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, -1)
        action = action.view(T, B)
        log_prob = logprobs_from_logits(policy_logits, action).view(T, B)

        # action and log_prob collected by actors and used also for the learner update
        output = dict(action=action, log_prob=log_prob)
        output_optional = {}

        # policy is needed only during the learner update
        if full_output:
            output_optional = dict(
                policy_logits=policy_logits,
            )

        return {**output, **output_optional}, core_state

    def log_probs(self, policy_output, action):
        logits = policy_output['policy_logits']
        return logprobs_from_logits(logits, action)

    def entropy(self, policy_output):
        logits = policy_output['policy_logits']
        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        return torch.sum(-policy * log_policy, dim=-1)


class ContinuousPolicy(BasePolicy):
    """ Gaussian policy with diagonal covariance for continuous actions. """
    def __init__(self, observation_shape, action_space, embedding_name, use_lstm=True,
                    batch_norm=False, embedding_pretrained=True, embedding_train=False,
                    log_std_network=False):
        super(ContinuousPolicy, self).__init__()

        self.device = torch.device('cpu') # default

        num_actions = action_space.shape[0]

        # If the action space is (-a, a), we pass the mean output through tanh
        # and multiply by a
        self.action_multiplier = None
        self.use_tanh = False
        if np.all(action_space.high == - action_space.low):
            self.action_multiplier = torch.from_numpy(action_space.high)
            self.use_tanh = True

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('tanh'))

        # Perception module
        if embedding_name is not None:
            assert observation_shape[2] % 3 == 0, 'must use RGB observations'
            n_frames = observation_shape[2] // 3
            self.perception = EmbeddingNet(embedding_name, observation_shape,
                pretrained=embedding_pretrained, train=embedding_train)
            conv_out_size = self.perception.out_size
        else:
            n_frames = 1
            self.perception = None
            conv_out_size = observation_shape[0]

        # Add batch norm
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(conv_out_size * n_frames)
        else:
            self.batch_norm = None

        # Linear layers
        self.fc = nn.Sequential(
            init_(nn.Linear(conv_out_size * n_frames, 256)),
            nn.Tanh(),
            init_(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        # LSTM
        if use_lstm:
            self.core = nn.LSTM(256, 256, 2)
        else:
            self.core = None

        # Outputs
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.policy_mean = init_(nn.Linear(256, num_actions))
        self.log_std_network = log_std_network
        self.min_log_std = -3.
        if log_std_network:
            self.policy_log_std = init_(nn.Linear(256, num_actions))
        else:
            init_log_std = 0.
            self.policy_log_std = nn.Parameter(torch.ones(num_actions,) * init_log_std)


    def forward(self, env_state, core_state=(), full_output=False):
        T, B, *_ = env_state['obs'].shape
        core_output, core_state = super().forward(env_state, core_state)

        policy_mean = self.policy_mean(core_output)
        if self.use_tanh:
            policy_mean = torch.tanh(policy_mean)
            policy_mean = policy_mean * self.action_multiplier.to(self.device)
        if self.log_std_network:
            policy_log_std = self.policy_log_std(core_output)
            policy_log_std = torch.tanh(policy_log_std)
        else:
            policy_log_std = self.policy_log_std.repeat(T, B, 1)
        policy_log_std = torch.clamp(policy_log_std, self.min_log_std)

        policy_mean = policy_mean.view(T, B, -1)
        policy_log_std = policy_log_std.view(T, B, -1)

        if self.training:
            action = torch.normal(policy_mean, torch.exp(policy_log_std))
        else:
            action = policy_mean

        log_prob = logprobs_from_mean_and_logstd(policy_mean, policy_log_std, action)

        output = dict(action=action, log_prob=log_prob)
        output_optional = {}

        if full_output:
            output_optional = dict(
                policy_mean=policy_mean,
                policy_log_std=policy_log_std,
            )

        return {**output, **output_optional}, core_state

    def log_probs(self, policy_output, action):
        mean = policy_output['policy_mean']
        log_std = policy_output['policy_log_std']
        return logprobs_from_mean_and_logstd(mean, log_std, action)

    def entropy(self, policy_output):
        log_std = policy_output['policy_log_std']
        return log_std.shape[-1] / 2 * np.log(2 * np.pi * np.e) + log_std.sum(-1)
