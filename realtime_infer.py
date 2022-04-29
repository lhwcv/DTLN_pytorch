import os
import torch
import tqdm
import numpy as np
from audio_io import wav_read, wav_write
from DTLN_model import Pytorch_DTLN_stateful

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="model dir",
                        default=os.path.dirname(__file__) + "/pretrained/model.pth")
    parser.add_argument("--wav_in",
                        type=str,
                        help="wav in",
                        default=os.path.dirname(__file__) + "/samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav")

    parser.add_argument("--wav_out",
                        type=str,
                        help="wav out",
                        default=os.path.dirname(__file__) + "/samples/enhanced2.wav")

    args = parser.parse_args()

    model = Pytorch_DTLN_stateful()
    print('==> load model from: ', args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print('==> read wav from: ', args.wav_in)
    audio, fs = wav_read(args.wav_in, tgt_fs=16000)
    print('==> audio len: {} secs'.format(len(audio) / fs))

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
        x = torch.from_numpy(in_block)
        with torch.no_grad():
            out_block, in_state1, in_state2 = model(x, in_state1, in_state2)
        out_block = out_block.numpy()
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        out[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]


    print('==> save wav to: ', args.wav_out)
    wav_write(out, 16000, args.wav_out)
