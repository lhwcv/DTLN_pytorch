import os
import torch
import tqdm
import numpy as np
from audio_io import wav_read, wav_write
from DTLN_model import Pytorch_DTLN_stateful
import onnxruntime

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model1_path",
                        type=str,
                        help="model1 path",
                        default=os.path.join(os.path.dirname(__file__), "pretrained/model_p1.onnx"))

    parser.add_argument("--model2_path",
                        type=str,
                        help="model2 path",
                        default=os.path.join(os.path.dirname(__file__), "pretrained/model_p2.onnx"))

    parser.add_argument("--wav_in",
                        type=str,
                        help="wav in",
                        default=os.path.join(os.path.dirname(__file__),
                                             "samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav"))

    parser.add_argument("--wav_out",
                        type=str,
                        help="wav out",
                        default=os.path.join(os.path.dirname(__file__), "samples/enhanced_onnx.wav"))

    args = parser.parse_args()


    print('==> load model1 from: ', args.model1_path)
    # load models
    interpreter_1 = onnxruntime.InferenceSession(args.model1_path)
    model_input_names_1 = [inp.name for inp in interpreter_1.get_inputs()]
    # preallocate input
    model_inputs_1 = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape],
            dtype=np.float32)
        for inp in interpreter_1.get_inputs()}

    for item in model_inputs_1.items():
        print("[ model1 ] input {} , shape: {}".format(item[0], item[1].shape))

    print('==> load model2 from: ', args.model1_path)
    interpreter_2 = onnxruntime.InferenceSession(args.model2_path)
    model_input_names_2 = [inp.name for inp in interpreter_2.get_inputs()]
    # preallocate input
    model_inputs_2 = {
        inp.name: np.zeros(
            [dim if isinstance(dim, int) else 1 for dim in inp.shape],
            dtype=np.float32)
        for inp in interpreter_2.get_inputs()}

    for item in model_inputs_2.items():
        print("[ model2] {} , shape: {}".format(item[0], item[1].shape))

    print('==> read wav from: ', args.wav_in)
    audio, fs = wav_read(args.wav_in, tgt_fs=16000)
    print('==> audio len: {} secs'.format(len(audio) / fs) )
    block_len = 512
    block_shift = 128

    # preallocate output audio
    out = np.zeros((len(audio)))
    # create buffer
    in_buffer = np.zeros((block_len))
    out_buffer = np.zeros((block_len))
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len - block_shift)) // block_shift
    # iterate over the number of blocks

    in_state1 = torch.zeros(2, 1, 128, 2)
    in_state2 = torch.zeros(2, 1, 128, 2)

    for idx in tqdm.tqdm(range(num_blocks)):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx * block_shift:(idx * block_shift) + block_shift]
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')

        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1, 1, -1)).astype('float32')

        # set block to input
        model_inputs_1[model_input_names_1[0]] = in_mag
        # run calculation
        model_outputs_1 = interpreter_1.run(None, model_inputs_1)
        # get the output of the first block
        estimated_mag = model_outputs_1[0]

        # set out states back to input
        model_inputs_1["h1_in"][0] = model_outputs_1[1][0]
        model_inputs_1["c1_in"][0] = model_outputs_1[1][1]
        model_inputs_1["h2_in"][0] = model_outputs_1[1][2]
        model_inputs_1["c2_in"][0] = model_outputs_1[1][3]

        # calculate the ifft
        estimated_complex = estimated_mag * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
        # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1, -1, 1)).astype('float32')
        # set tensors to the second block
        # interpreter_2.set_tensor(input_details_1[1]['index'], states_2)
        model_inputs_2[model_input_names_2[0]] = estimated_block
        # run calculation
        model_outputs_2 = interpreter_2.run(None, model_inputs_2)
        # get output
        out_block = model_outputs_2[0]
        # set out states back to input

        model_inputs_2["h1_in"][0] = model_outputs_2[1][0]
        model_inputs_2["c1_in"][0] = model_outputs_2[1][1]
        model_inputs_2["h2_in"][0] = model_outputs_2[1][2]
        model_inputs_2["c2_in"][0] = model_outputs_2[1][3]

        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)

        # print(idx, np.abs(out_buffer).sum())
        # write block to output file
        out[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]


    print('==> save wav to: ', args.wav_out)
    wav_write(out, 16000, args.wav_out)
