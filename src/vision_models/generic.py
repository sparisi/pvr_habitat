import sys
import torch
import torch.nn as nn

# Some architectures that one could use:
# https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/monobeast.py#L552
# https://github.com/facebookresearch/torchbeast/blob/main/torchbeast/polybeast_learner.py#L147
# https://github.com/Divye02/hand_vil/blob/master/mjrl_mod/policies/gaussian_cnn.py#L285-L292
# https://github.com/facebookresearch/impact-driven-exploration/blob/main/src/models.py#L42

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# This is just a template for a convolutional network.
# You can change batch normalization, pooling, activation, stride, ...
def generic_convnet(n_layers):
    init_ = lambda m: init(m, nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        nn.init.calculate_gain('relu'))
    return nn.Sequential(
        init_(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)),
        nn.ELU(),
        *list([
            init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
            nn.ELU()
        ] * (n_layers - 1)),
    )


# def generic_convnet():
#     init_ = lambda m: init(m, nn.init.orthogonal_,
#         lambda x: nn.init.constant_(x, 0),
#         nn.init.calculate_gain('relu'))
#     return nn.Sequential(
#         init_(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
#         nn.BatchNorm2d(16),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#         #
#         init_(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#         #
#         init_(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#         nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
#         #
#         init_(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
#         nn.BatchNorm2d(32),
#         nn.ReLU(),
#     )


if __name__ == '__main__':
    # Use this to save the network and load it for seed reproducibility, e.g.
    # python generic.py 5
    # This will save the model in the current folder, then you will have to move
    # it to PVR_MODEL_PATH, defined in embeddings.py
    n_layers = int(sys.argv[1])
    model = generic_convnet(n_layers)
    torch.save({
        'state_dict': model.state_dict(),
    }, f'random{n_layers}.tar')
