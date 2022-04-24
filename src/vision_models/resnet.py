import torch
import torch.nn as nn
import torchvision.models as models


def resnet_conv3_compressed(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample = nn.Sequential(
        nn.Conv2d(1024,
                  11,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  dilation=1), model._norm_layer(11))

    model.layer3 = nn.Sequential(
        model.layer3,
        models.resnet.BasicBlock(1024,
                                 11,
                                 stride=1,
                                 norm_layer=model._norm_layer,
                                 downsample=downsample))

    # Remove layer 4 and avgpool layer
    model.layer4 = nn.Sequential()
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    # Rename the keys correctly
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = state_dict[k]
        # Delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.' in n or 'layer3.2' in n for n in msg.unexpected_keys])

    return model


def resnet_conv4_compressed(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']

    # Construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample = nn.Sequential(
        nn.Conv2d(2048,
                  42,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  dilation=1), model._norm_layer(42))
    model.layer4 = nn.Sequential(
        model.layer4,
        models.resnet.BasicBlock(2048,
                                 42,
                                 stride=1,
                                 norm_layer=model._norm_layer,
                                 downsample=downsample))

    # Remove avgpool layer
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    # Rename the keys correctly
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = state_dict[k]
        # Delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.2' in n for n in msg.unexpected_keys])

    return model


def resnet_conv5(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    # Construct the model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    model.fc = nn.Sequential()

    # Rename the keys correctly
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len('module.'):]] = state_dict[k]
        # Delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert len(msg.missing_keys) == 0

    return model
