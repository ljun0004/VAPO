# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as nfunc


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalActNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data.zero_()
        self.init = False

    def forward(self, x, y):
        if self.init:
            scale, bias = self.embed(y).chunk(2, dim=-1)
            return x * scale[:, :, None, None] + bias[:, :, None, None]
        else:
            m, v = torch.mean(x, dim=(0, 2, 3)), torch.var(x, dim=(0, 2, 3))
            std = torch.sqrt(v + 1e-5)
            scale_init = 1. / std
            bias_init = -1. * m / std
            self.embed.weight.data[:, :self.num_features] = scale_init[None].repeat(self.num_classes, 1)
            self.embed.weight.data[:, self.num_features:] = bias_init[None].repeat(self.num_classes, 1)
            self.init = True
            return self(x, y)


logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ContinuousConditionalActNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        del num_classes
        self.num_features = num_features
        self.embed = nn.Sequential(nn.Linear(1, 256),
                                   nn.ELU(inplace=True),
                                   nn.Linear(256, 256),
                                   nn.ELU(inplace=True),
                                   nn.Linear(256, self.num_features*2),
                                   )
        
    def forward(self, x, y):
        scale, bias = self.embed(y.unsqueeze(-1)).chunk(2, dim=-1)
        return x * scale[:, :, None, None] + bias[:, :, None, None]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def call_bn(bn, x, update_batch_stats=True):
    if bn.training is False:
        return bn(x)
    elif not update_batch_stats:
        return nfunc.batch_norm(x, None, None, bn.weight, bn.bias, True, bn.momentum, bn.eps)
    else:
        return bn(x)


def get_norm(n_filters, norm):
    if norm is None or norm.lower() == 'none':
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)
    elif norm == "act":
        return ActNorm(n_filters, False)
    else:
        return Identity()
    
def get_nonl(nonl):
    if nonl == 'leakyrelu':
        return nn.LeakyReLU(leak=0.2, inplace=True)
    elif nonl == 'gelu': 
        return nn.GELU()
    elif nonl == 'silu': 
        # return Lip_swish
        return nn.SiLU()
    elif nonl == None: 
        return Identity()
    else:
        raise NotImplementedError(f'nonl {nonl} not supported yet!')
    
def Lip_swish(x):
	return (x * torch.sigmoid(x))/1.1