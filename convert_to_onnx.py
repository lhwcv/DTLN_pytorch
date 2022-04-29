import os
import torch
from DTLN_model_ncnn_compat import Pytorch_DTLN_P1_stateful, Pytorch_DTLN_P2_stateful

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        help="model dir",
                        default=os.path.dirname(__file__) + "/pretrained/model.pth")
    parser.add_argument("--model_1",
                        type=str,
                        help="model part 1 save path",
                        default=os.path.dirname(__file__) + "/pretrained/model_p1.onnx")

    parser.add_argument("--model_2",
                        type=str,
                        help="model part 2 save path",
                        default=os.path.dirname(__file__) + "/pretrained/model_p2.onnx")

    args = parser.parse_args()

    model1 = Pytorch_DTLN_P1_stateful()
    print('==> load model from: ', args.model_path)
    model1.load_state_dict(torch.load(args.model_path), strict=False)
    model1.eval()
    model2 = Pytorch_DTLN_P2_stateful()
    model2.load_state_dict(torch.load(args.model_path), strict=False)
    model2.eval()

    block_len = 512
    hidden_size = 128
    # in_state1 = torch.zeros(2, 1, hidden_size, 2)
    # in_state2 = torch.zeros(2, 1, hidden_size, 2)
    in_state1 = torch.zeros(4, 1, 1, 128)
    in_state2 = torch.zeros(4, 1, 1, 128)

    mag = torch.zeros(1, 1, (block_len // 2 + 1))
    phase = torch.zeros(1, 1, (block_len // 2 + 1))
    y1 = torch.zeros(1, block_len, 1)

    # NCNN not support Gather
    input_names = ["mag", "h1_in", "c1_in", "h2_in", "c2_in"]
    output_names = ["y1", "out_state1"]

    print("==> export to: ", args.model_1)
    torch.onnx.export(model1,
                      (mag, in_state1[0], in_state1[1], in_state1[2], in_state1[3]),
                      args.model_1,
                      input_names=input_names, output_names=output_names)

    input_names = ["y1", "h1_in", "c1_in", "h2_in", "c2_in"]
    output_names = ["y", "out_state2"]

    print("==> export to: ", args.model_2)
    torch.onnx.export(model2,
                      (y1, in_state2[0], in_state2[1], in_state2[2], in_state2[3]),
                      args.model_2,
                      input_names=input_names, output_names=output_names)
