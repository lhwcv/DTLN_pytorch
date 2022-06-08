# -*- coding: utf-8 -*-
import os
import torch
import tensorflow as tf
import numpy as np
import torch.nn as nn

from audio_io import  wav_read, wav_write
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D

from DTLN_model import Pytorch_DTLN

def tf_seperation_kernel(num_layer, mask_size, x, stateful=False, numUnits=128, dropout=0.0):
    '''
    Method to create a separation kernel.
    !! Important !!: Do not use this layer with a Lambda layer. If used with
    a Lambda layer the gradients are updated correctly.
    Inputs:
        num_layer       Number of LSTM layers
        mask_size       Output size of the mask and size of the Dense layer
    '''

    # creating num_layer number of LSTM layers
    for idx in range(num_layer):
        x = LSTM(numUnits, return_sequences=True, stateful=stateful)(x)
        # using dropout between the LSTM layer for regularization
        if idx < (num_layer - 1):
            x = Dropout(dropout)(x)
    # creating the mask with a Dense and an Activation layer
    mask = Dense(mask_size)(x)
    mask = Activation('sigmoid')(mask)
    # returning the mask
    return mask


def tf_stftLayer(x):
    '''
    Method for an STFT helper layer used with a Lambda layer. The layer
    calculates the STFT on the last dimension and returns the magnitude and
    phase of the STFT.
    '''

    # creating frames from the continuous waveform
    frames = tf.signal.frame(x, 512, 128)
    # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
    stft_dat = tf.signal.rfft(frames)
    # calculating magnitude and phase from the complex signal
    mag = tf.abs(stft_dat)
    phase = tf.math.angle(stft_dat)
    # returning magnitude and phase as list
    return [mag, phase]


def tf_ifftLayer(x):
    '''
    Method for an inverse FFT layer used with an Lambda layer. This layer
    calculates time domain frames from magnitude and phase information.
    As input x a list with [mag,phase] is required.
    '''

    # calculating the complex representation
    s1_stft = (tf.cast(x[0], tf.complex64) *
               tf.exp((1j * tf.cast(x[1], tf.complex64))))
    # returning the time domain frames
    return tf.signal.irfft(s1_stft)


def tf_overlapAddLayer(x):
    '''
    Method for an overlap and add helper layer used with a Lambda layer.
    This layer reconstructs the waveform from a framed signal.
    '''
    block_shift = 128
    # calculating and returning the reconstructed waveform
    return tf.signal.overlap_and_add(x, block_shift)

class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)

        #print('tf mean shape: ', mean.shape)
        #exit()
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean),
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


def build_tf_DTLN_model():
    numLayer = 2
    blockLen = 512
    encoder_size = 256

    # input layer for time signal
    #time_dat = Input(batch_shape=(1, 512 * 2))
    time_dat = Input(batch_shape=(None, None))

    # calculate STFT
    mag, angle = Lambda(tf_stftLayer)(time_dat)
    mag_norm = mag
    # predicting mask with separation kernel
    mask_1 = tf_seperation_kernel(numLayer, (blockLen // 2 + 1), mag_norm)

    #print('mask1 shape: ', mask_1.shape)
    # multiply mask with magnitude
    estimated_mag = Multiply()([mag, mask_1])
    # transform frames back to time domain
    estimated_frames_1 = Lambda(tf_ifftLayer)([estimated_mag, angle])

    #decoded_frame = estimated_frames_1
    #decoded_frame = estimated_mag
    #print('estimated_frames_1 shape: ', estimated_frames_1.shape)
    # encode time domain frames to feature domain
    encoded_frames = Conv1D(encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
    # normalize the input to the separation kernel
    encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
    # # predict mask based on the normalized feature frames
    mask_2 = tf_seperation_kernel(numLayer, encoder_size, encoded_frames_norm)
    # # multiply encoded frames with the mask
    estimated = Multiply()([encoded_frames, mask_2])
    # # decode the frames back to time domain
    decoded_frame = Conv1D(blockLen, 1, padding='causal', use_bias=False)(estimated)
    # create waveform with overlap and add procedure
    estimated_sig = Lambda(tf_overlapAddLayer)(decoded_frame)
    #print('decoded_frame shape: ', decoded_frame.shape)
    model = Model(inputs=time_dat, outputs=[estimated_sig,encoded_frames_norm])
    model.compile()
    print(model.summary())
    return model


def main(weights_file, pytorch_save_file='./model.pth'):

    tf_model = build_tf_DTLN_model()
    tf_model.load_weights(weights_file)
    tf_weigts = tf_model.get_weights()
    # print('weights: ', len(tf_weigts))
    # for i, w in enumerate(tf_weigts):
    #     print(i, w.shape)

    tf_weigts_dict = {}
    tf_weigts_dict['sep1.rnn1.weight_ih_l0'] = torch.from_numpy(tf_weigts[0].T)
    tf_weigts_dict['sep1.rnn1.weight_hh_l0'] = torch.from_numpy(tf_weigts[1].T)
    tf_weigts_dict['sep1.rnn1.bias_hh_l0'] = torch.from_numpy(tf_weigts[2])

    tf_weigts_dict['sep1.rnn2.weight_ih_l0'] = torch.from_numpy(tf_weigts[3].T)
    tf_weigts_dict['sep1.rnn2.weight_hh_l0'] = torch.from_numpy(tf_weigts[4].T)
    tf_weigts_dict['sep1.rnn2.bias_hh_l0'] = torch.from_numpy(tf_weigts[5])

    tf_weigts_dict['sep1.dense.weight'] = torch.from_numpy(tf_weigts[6].T)
    tf_weigts_dict['sep1.dense.bias'] = torch.from_numpy(tf_weigts[7])

    tf_weigts_dict['encoder_conv1.weight'] = torch.from_numpy(tf_weigts[8]).permute(2,1,0)
    tf_weigts_dict['encoder_norm1.gamma'] = torch.from_numpy(tf_weigts[9]).unsqueeze(0).unsqueeze(0)
    tf_weigts_dict['encoder_norm1.beta'] = torch.from_numpy(tf_weigts[10]).unsqueeze(0).unsqueeze(0)

    tf_weigts_dict['sep2.rnn1.weight_ih_l0'] = torch.from_numpy(tf_weigts[11].T)
    tf_weigts_dict['sep2.rnn1.weight_hh_l0'] = torch.from_numpy(tf_weigts[12].T)
    tf_weigts_dict['sep2.rnn1.bias_hh_l0'] = torch.from_numpy(tf_weigts[13])
    tf_weigts_dict['sep2.rnn2.weight_ih_l0'] = torch.from_numpy(tf_weigts[14].T)
    tf_weigts_dict['sep2.rnn2.weight_hh_l0'] = torch.from_numpy(tf_weigts[15].T)
    tf_weigts_dict['sep2.rnn2.bias_hh_l0'] = torch.from_numpy(tf_weigts[16])
    tf_weigts_dict['sep2.dense.weight'] = torch.from_numpy(tf_weigts[17].T)
    tf_weigts_dict['sep2.dense.bias'] = torch.from_numpy(tf_weigts[18])

    tf_weigts_dict['decoder_conv1.weight'] = torch.from_numpy(tf_weigts[19]).permute(2,1,0)

    model = Pytorch_DTLN()
    model.eval()
    torchparam = model.state_dict()
    for k, v in torchparam.items():
        if 'bias_ih_l0' in k:
            #print(k)
            torch.zero_(torchparam[k])
        print("{:20s} {}".format(k, v.shape))
    torchparam.update(tf_weigts_dict)
    model.load_state_dict(torchparam, strict=True)
    print("==> save state_dict() to {}".format(pytorch_save_file))
    torch.save(model.state_dict(), pytorch_save_file)


if __name__ == '__main__':
    # https://github.com/breizhn/DTLN/blob/master/pretrained_model/model.h5
    keras_weights_file = './model.h5'
    
    pytorch_save_file = './model.pth'
    main(keras_weights_file, pytorch_save_file)
