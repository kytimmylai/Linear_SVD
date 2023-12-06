"""
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠛⠛⠛⠛⠛⠛⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⠿⠛⢉⣠⣴⣶⣾⣿⣿⣿⣿⣿⣿⣿⣷⣶⣤⣌⣉⠙⠻⠿⢿⣿⣿
⣿⣿⡿⠛⣁⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣀⠈⣻
⣿⡟⢁⣾⣿⣿⠿⠛⠻⢿⣭⣿⣿⣿⣿⣿⣿⣟⣥⠟⠛⠛⠻⢿⣿⣿⡟⣿⣧⢸
⡟⢀⣾⣿⣿⡏⠀⠤⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⠃⠠⠶⠄⠀⠀⣽⣿⣿⣦⡉⢸⠇
⢁⣾⣿⣿⣿⣷⣦⣀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣦⣄⡀⣀⣀⣴⣿⣿⣿⣿⣇⠹
⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⠉⠀⠀⠉⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣤
⣿⣟⡻⠿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣿
⣿⣟⣻⣷⣾⣿⣿⣿⣿⣿⣿⣿⡿⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣵⣿⣟⣻
⣿⡿⣿⣿⣿⣿⠿⣿⣿⣿⠟⢋⣠⣤⣤⣀⡙⠛⠻⠿⣿⠿⠿⣩⣿⣿⣿⣟⣻⣿
⣿⣿⣿⣿⣿⣿⣦⣤⣉⣠⣶⣿⣿⣿⣿⣿⣿⣷⣦⣄⡀⠀⠀⣨⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⡿⣿⣿⣿⠿⠿⣭⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣤⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠇
⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠂⠇
"""
import torch
from collections import OrderedDict

def linear_svd_dec(
        layer: torch.nn.Linear, 
        q: int
    ):
    r"""Decompose a linear layer into 2 conti. linear layer w/ SVD, 
    which will produce a similar output w/ reduced dimension within them.

    Layer weight in R^(m x n) ~ U @ diag(S) @ V^t 
    where U in R^(m x q), S in R^(q), V in R^(n x q)

    layer1 = V^t
    layer2 = U @ diag(S)

    Args:
        layer: the target layer project tensor fom 
            n -> m into n -> q -> m
        q: desired channel number between 2 modules
    """
    weight = layer.state_dict()['weight']
    out_ch, in_ch = weight.shape
    u, s, v = torch.svd_lowrank(weight, q=q)
    linear1 = torch.nn.Linear(in_ch, q, bias=False)
    linear1.load_state_dict(OrderedDict({'weight': v.permute(1, 0)}))
    
    linear2 = torch.nn.Linear(q, out_ch, bias=(layer.bias != None))
    linear2_state_dict = OrderedDict({'weight': u @ torch.diag(s)})
    if layer.bias != None:
        linear2_state_dict['bias'] = layer.state_dict()['bias']
    linear2.load_state_dict(linear2_state_dict)
    return torch.nn.Sequential(linear1, linear2)

if __name__ == '__main__':
    test_layer = torch.nn.Linear(48, 192).eval()
    test_input = torch.ones((4, 16, 16, 48))
    test_output = test_layer(test_input)
    trans_layer = linear_svd_dec(test_layer, q=24).eval()
    trans_output = trans_layer(test_input)

    print('Output diff %.4f'%(test_output - trans_output).sum())
    print(
        'Weight diff: %.4f'%(test_layer.state_dict()['weight'] - \
            trans_layer[1].state_dict()['weight'] @ \
            trans_layer[0].state_dict()['weight']
        ).sum()
    )