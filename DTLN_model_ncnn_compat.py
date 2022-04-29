# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn


class Pytorch_InstantLayerNormalization(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
            Constructor
        """
        super(Pytorch_InstantLayerNormalization, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs

class SeperationBlock_Stateful(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(SeperationBlock_Stateful, self).__init__()
        self.rnn1 = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.rnn2 = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h1_in, c1_in, h2_in, c2_in):
        """

        :param x:  [N, T, input_size]
        :param in_states: [1, 1, 128, 4]
        :return:
        """
        #h1_in, c1_in = in_states[:, :, :, 0], in_states[:, :, :, 1]
        #h2_in, c2_in = in_states[:, :, :, 2], in_states[:, :, :, 3]

        # NCNN not support Gather
        x1, (h1, c1) = self.rnn1(x, (h1_in, c1_in))
        x1 = self.drop(x1)
        x2, (h2, c2) = self.rnn2(x1, (h2_in, c2_in))
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)

        out_states = torch.cat((h1, c1, h2, c2), dim=0)
        return mask, out_states

class Pytorch_DTLN_P1_stateful(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window='rect'):
        super(Pytorch_DTLN_P1_stateful, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        self.sep1 = SeperationBlock_Stateful(input_size=(frame_len // 2 + 1), hidden_size=128, dropout=0.25)

    def forward(self, mag, h1_in, c1_in, h2_in, c2_in):
        """

        :param mag:  [1, 1, 257]
        :param in_state1: [1, 1, 128, 4]
        :return:
        """
        #assert in_state1.shape[0] == 1
        #assert in_state1.shape[-1] == 4
        # N, T, hidden_size
        mask, out_state1 = self.sep1(mag, h1_in, c1_in, h2_in, c2_in)
        estimated_mag = mask * mag
        return estimated_mag, out_state1


class Pytorch_DTLN_P2_stateful(nn.Module):
    def __init__(self, frame_len=512):
        super(Pytorch_DTLN_P2_stateful, self).__init__()
        self.frame_len = frame_len
        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(in_channels=frame_len, out_channels=self.encoder_size,
                                       kernel_size=1, stride=1, bias=False)

        # self.encoder_norm1 = nn.InstanceNorm1d(num_features=self.encoder_size, eps=1e-7, affine=True)
        self.encoder_norm1 = Pytorch_InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock_Stateful(input_size=self.encoder_size, hidden_size=128, dropout=0.25)

        ## TODO with causal padding like in keras,when ksize > 1
        self.decoder_conv1 = nn.Conv1d(in_channels=self.encoder_size, out_channels=frame_len,
                                       kernel_size=1, stride=1, bias=False)

    def forward(self, y1, h1_in, c1_in, h2_in, c2_in):
        """
        :param y1: [1, framelen, 1]
        :param in_state2:  [1, 1, 128, 4]
        :return:
        """
        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2, out_state2 = self.sep2(encoded_f_norm, h1_in, c1_in, h2_in, c2_in)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        return decoded_frame, out_state2
