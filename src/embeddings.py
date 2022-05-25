import numpy as np
import os
import gym
import warnings

import torch
import torch.nn as nn
from torch.nn.modules.linear import Identity
import torchvision.models as models
import torchvision.transforms as T

from src.vision_models.moco import (
    moco_conv3_compressed,
    moco_conv4_compressed,
    moco_conv5,
)
from src.vision_models.resnet import (
    resnet_conv3_compressed,
    resnet_conv4_compressed,
    resnet_conv5,
)
from src.vision_models.generic import generic_convnet


if 'PVR_MODEL_PATH' in os.environ:
    PVR_MODEL_PATH = os.environ['PVR_MODEL_PATH']
else:
    PVR_MODEL_PATH = '/checkpoint/sparisi/pvr_models/'


# ==============================================================================
# GET EMBEDDING
# ==============================================================================

class UberModel(nn.Module):
    def __init__(self, models):
        super(UberModel, self).__init__()
        self.models = models
        assert all(models[0].training == m.training for m in models)
        self.training = models[0].training

    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        return self

    def forward(self, x):
        return torch.cat([m(x) for m in self.models],
            dim=1 if x.ndim > 1 else 0)


def _get_embedding(embedding_name='random', in_channels=3, pretrained=True, train=False):
    """
    See https://pytorch.org/vision/stable/models.html

    Args:
        embedding_name (str, 'random'): the name of the convolution model,
        in_channels (int, 3): number of channels of the input image,
        pretrained (bool, True): if True, the model's weights will be downloaded
            from torchvision (if possible),
        train (bool, False): if True the model will be trained during learning,
            if False its parameters will not change.

    """

    # ResNet default transforms: https://pytorch.org/vision/stable/models.html
    # All pre-trained models expect input images normalized in the same way,
    # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
    # where H and W are expected to be at least 224.
    # The images have to be loaded in to a range of [0, 1] and then
    # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    # MAE has a different transform for testing, see https://github.com/facebookresearch/mae
    transforms = nn.Sequential(
        T.Resize(256, interpolation=3) if 'mae' in embedding_name else T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float), # also divides by 255 if input is uint8
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    assert in_channels == 3, 'Current models accept 3-channel inputs only.'

    def pvr_path(emb_name): # just to make code below more compact
        return os.path.join(PVR_MODEL_PATH, emb_name)

    # GENERIC 5-LAYER CONV
    if embedding_name.startswith('random'):
        n_layers = int(embedding_name[len('random'):])
        model = generic_convnet(n_layers)
        try: # must have saved the model first (run vision_models/generic.py)
            checkpoint = torch.load(pvr_path(f'{embedding_name}.tar'), map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
        except:
            warn = f'WARN: Unable to load embedding model: {embedding_name}. The model will be randomly initialized.'
            yellow_warn = f"\033[33m{warn}\033[m"
            warnings.warn(
                yellow_warn)
        transforms = nn.Sequential(
            T.ConvertImageDtype(torch.float),
        )

    # Make FC layers to be Identity
    # This works for the models below but may not work for any network

    # VANILLA RESNET
    elif embedding_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained, progress=False)
        model.fc = Identity()
    elif embedding_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained, progress=False)
        model.fc = Identity()
    elif embedding_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained, progress=False)
        model.fc = Identity()
    elif embedding_name == 'resnet50_places':
        model = resnet_conv5(checkpoint_path=pvr_path('resnet50_places.pth.tar'))
    elif embedding_name == 'resnet50_l4':
        model = resnet_conv4_compressed(checkpoint_path=pvr_path('resnet50_l4.pth.tar'))
    elif embedding_name == 'resnet50_l3':
        model = resnet_conv3_compressed(checkpoint_path=pvr_path('resnet50_l3.tar'))
    elif embedding_name == 'resnet50_places_l4':
        model = resnet_conv4_compressed(checkpoint_path=pvr_path('resnet50_places_l4.tar'))
    elif embedding_name == 'resnet50_places_l3':
        model = resnet_conv3_compressed(checkpoint_path=pvr_path('resnet50_places_l3.tar'))

    # DEMYSTIFY
    elif embedding_name == 'demy':
        model = moco_conv5(checkpoint_path=pvr_path('demy.pth'))

    # MAE
    elif 'mae' in embedding_name:
        from src.vision_models.mae import (
            mae_vit_base_patch16,
            mae_vit_large_patch16,
            mae_vit_huge_patch14,
        )
        if embedding_name == 'mae_base':
            model = mae_vit_base_patch16()
            checkpoint = torch.load(pvr_path('mae_pretrain_vit_base.pth'), map_location='cpu')
        if embedding_name == 'mae_large':
            model = mae_vit_large_patch16()
            checkpoint = torch.load(pvr_path('mae_pretrain_vit_large.pth'), map_location='cpu')
        if embedding_name == 'mae_huge':
            model = mae_vit_huge_patch14()
            checkpoint = torch.load(pvr_path('mae_pretrain_vit_huge.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)

    # MOCO
    elif embedding_name == 'moco_aug':
        model = moco_conv5(checkpoint_path=pvr_path('moco_aug.pth.tar'))
    elif embedding_name == 'moco_aug_habitat':
        model = moco_conv5(checkpoint_path=pvr_path('moco_aug_habitat.pth'))
    elif embedding_name == 'moco_aug_mujoco':
        model = moco_conv5(checkpoint_path=pvr_path('moco_aug_mujoco.pth'))
    elif embedding_name == 'moco_aug_uber':
        model = moco_conv5(checkpoint_path=pvr_path('moco_aug_uber.pth'))
    elif embedding_name == 'moco_aug_places':
        model = moco_conv5(checkpoint_path=pvr_path('moco_aug_places.pth.tar'))

    elif embedding_name == 'moco_aug_l4':
        model = moco_conv4_compressed(checkpoint_path=pvr_path('moco_aug_l4.pth'))
    elif embedding_name == 'moco_aug_places_l4':
        model = moco_conv4_compressed(checkpoint_path=pvr_path('moco_aug_places_l4.pth'))
    elif embedding_name == 'moco_aug_l3':
        model = moco_conv3_compressed(checkpoint_path=pvr_path('moco_aug_l3.pth'))
    elif embedding_name == 'moco_aug_places_l3':
        model = moco_conv3_compressed(checkpoint_path=pvr_path('moco_aug_places_l3.pth'))

    elif embedding_name == 'moco_croponly':
        model = moco_conv5(checkpoint_path=pvr_path('moco_croponly.pth'))
    elif embedding_name == 'moco_croponly_places':
        model = moco_conv5(checkpoint_path=pvr_path('moco_croponly_places.pth'))
    elif embedding_name == 'moco_croponly_habitat':
        model = moco_conv5(checkpoint_path=pvr_path('moco_croponly_habitat.pth'))
    elif embedding_name == 'moco_croponly_mujoco':
        model = moco_conv5(checkpoint_path=pvr_path('moco_croponly_mujoco.pth'))
    elif embedding_name == 'moco_croponly_uber':
        model = moco_conv5(checkpoint_path=pvr_path('moco_croponly_uber.pth'))

    elif embedding_name == 'moco_croponly_l4':
        model = moco_conv4_compressed(checkpoint_path=pvr_path('moco_croponly_l4.pth'))
    elif embedding_name == 'moco_croponly_l3':
        model = moco_conv3_compressed(checkpoint_path=pvr_path('moco_croponly_l3.pth'))
    elif embedding_name == 'moco_croponly_places_l4':
        model = moco_conv4_compressed(checkpoint_path=pvr_path('moco_croponly_places_l4.pth'))
    elif embedding_name == 'moco_croponly_places_l3':
        model = moco_conv3_compressed(checkpoint_path=pvr_path('moco_croponly_places_l3.pth'))

    elif embedding_name == 'moco_coloronly':
        model = moco_conv5(checkpoint_path=pvr_path('moco_coloronly.pth'))

    # MOCO UBER MODELS (AUG)
    elif embedding_name == 'moco_aug_places_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_345':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_35':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug')[0]
        ])
    elif embedding_name == 'moco_aug_places_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_places_l3')[0],
            _get_embedding('moco_aug_places_l4')[0],
        ])
    elif embedding_name == 'moco_aug_uber_34':
        model = UberModel([
            _get_embedding('moco_aug_l3')[0],
            _get_embedding('moco_aug_l4')[0],
        ])
    elif embedding_name == 'moco_aug_places_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_places_l4')[0],
            _get_embedding('moco_aug_places')[0]
        ])
    elif embedding_name == 'moco_aug_uber_45':
        model = UberModel([
            _get_embedding('moco_aug_l4')[0],
            _get_embedding('moco_aug')[0]
        ])

    # MOCO UBER MODELS (CROP)
    elif embedding_name == 'moco_croponly_places_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_345':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_35':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly')[0]
        ])
    elif embedding_name == 'moco_croponly_places_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_places_l3')[0],
            _get_embedding('moco_croponly_places_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_uber_34':
        model = UberModel([
            _get_embedding('moco_croponly_l3')[0],
            _get_embedding('moco_croponly_l4')[0],
        ])
    elif embedding_name == 'moco_croponly_places_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_places_l4')[0],
            _get_embedding('moco_croponly_places')[0]
        ])
    elif embedding_name == 'moco_croponly_uber_45':
        model = UberModel([
            _get_embedding('moco_croponly_l4')[0],
            _get_embedding('moco_croponly')[0]
        ])

    # MASK
    elif embedding_name == 'maskrcnn_l3':
        from src.vision_models.maskrcnn import mask_rcnn_model
        # Input must be BGR and not normalized in [0, 1] (ie, keep them in [0, 255])
        class _rgb_to_bgr(nn.Module):
            def forward(self, x):
                x[:,:,[0,1,2]] = x[:,:,[2,1,0]]
                return x.float()
        transforms = nn.Sequential(
            _rgb_to_bgr(),
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize([103.530, 116.280, 123.675], [1.0, 1.0, 1.0]),
        )
        model = mask_rcnn_model(checkpoint_path=pvr_path('maskrcnn_l3.pth'))

    # CLIP
    elif 'clip' in embedding_name:
        import clip
        # Custom transforms from
        # https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L76
        # My code avoids PIL and is faster, but works only with antialias=True
        # (see https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
        if embedding_name == 'clip_vit':
            model, _ = clip.load("ViT-B/32", device='cpu')
        elif embedding_name == 'clip_rn50':
            model, _ = clip.load("RN50", device='cpu')
        else:
            raise NotImplementedError(f"Unknown embedding model: {embedding_name}.")
        transforms = nn.Sequential(
            T.Resize(model.visual.input_resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(model.visual.input_resolution),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        )

    else:
        raise NotImplementedError(f"Unknown embedding model: {embedding_name}.")

    if train:
        model.train()
        for p in model.parameters():
            p.requires_grad = True
    else:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model, transforms


# ==============================================================================
# EMBEDDING CLASS
# ==============================================================================

class EmbeddingNet(nn.Module):
    """
    Input shape must be (N, H, W, 3), where N is the number of frames.
    The class will then take care of transforming and normalizing frames.
    The output shape will be (N, O), where O is the embedding size.
    """
    def __init__(self, embedding_name, in_shape, pretrained=True, train=False):
        super(EmbeddingNet, self).__init__()

        self.device = torch.device('cpu') # default

        self.embedding_name = embedding_name
        if self.embedding_name == 'true_state':
            return

        # in_shape = (H, W, channels * n), some envs return n images per obs
        assert in_shape[2] % 3 == 0, 'must use RGB observations'

        self.model, self.transforms = \
            _get_embedding(embedding_name, 3, pretrained, train)

        dummy_in = torch.zeros(1, 3, in_shape[0], in_shape[1])
        dummy_in = self.transforms(dummy_in.to(torch.uint8))
        self.in_shape = dummy_in.shape[1:]
        dummy_out = self._forward(dummy_in)
        self.out_size = np.prod(dummy_out.shape)

        # Cannot call it 'self.training' or it will be set by the eval() and train() functions
        self.freeze = not train


    # Override `to` to keep track of the device. This may not work for other applications,
    # but it is the easiest way for our needs. The alternative is to manually set it
    # outside whenever we call model.to() but it is easy to forget about it.
    # Other solutions found online are incompatible with DistributedDataParallel.
    def to(self, device):
        self.device = torch.device(device)
        try: self.model.to(device)
        except: pass
        return super().to(device)


    def _forward(self, observation):
        if 'clip' in self.embedding_name:
            out = self.model.encode_image(observation)
        elif 'mae' in self.embedding_name:
            out, *_ = self.model.forward_encoder(observation, mask_ratio=0.0)
            out = out[:,0,:]
        else:
            out = self.model(observation)
            if self.embedding_name == 'maskrcnn_l3':
                out = out['res4']
        return out


    def forward(self, observation):
        if self.embedding_name == 'true_state':
            return observation

        # observation.shape -> (N, H, W, 3)
        if observation.ndim == 3: # single sample
            observation = observation.view(1, *observation.shape)
        observation = observation.to(device=self.device)
        observation = observation.transpose(1, 2).transpose(1, 3).contiguous()
        observation = self.transforms(observation.to(torch.uint8)) # must be uint8 for proper normalization
        observation = observation.reshape(-1, *self.in_shape)

        with torch.set_grad_enabled(not self.freeze):
            out = self._forward(observation)
            return out.view(-1, self.out_size).squeeze()


# ==============================================================================
# EMBEDDING WRAPPER
# ==============================================================================

class EmbeddingWrapper(gym.ObservationWrapper):
    """
    This wrapper places a convolution model over the observation.
    The original observation shape must be (H, W, n * 3), where n is the number
    of frames per observation.
    If n > 1, each frame will pass through the convolution separately.
    The outputs will then be stacked.
    Note that the embedding 'true_state' is a special case.
    Args:
        env (gym.Env): the environment,
        embedding (torch.nn.Module): neural network defining the observation
            embedding.
    """
    def __init__(self, env, embedding):
        gym.ObservationWrapper.__init__(self, env)

        self.embedding = embedding

        if self.embedding.embedding_name == 'true_state':
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                dtype=np.float32,
                shape=env._true_state.shape
            )
            return

        in_channels = env.observation_space.shape[2]
        self.n_frames = in_channels // 3
        assert in_channels % 3 == 0,  \
                """ Only RGB images are supported.
                    Be sure that observation shape is (H, W, n * 3),
                    where n is the number of frames per observation. """

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.embedding.out_size * self.n_frames,)
        )

    def observation(self, observation):
        if self.embedding.embedding_name == 'true_state':
            return self.env._true_state # the original env class must implement it

        # if self.n_frames > 1, each frame goes through the embedding one by one
        observation = np.stack(np.split(observation, self.n_frames, axis=-1)) # (H, W, self.n_frames * 3) -> (self.n_frames, H, W, 3)
        return self.embedding(torch.from_numpy(observation)).flatten().cpu().numpy()
