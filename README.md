# Dual-signal Transformation LSTM Network

+ This repository provides the pytorch code example

+ official repo:  https://github.com/breizhn/DTLN
---



## 0. prequisites

- pytorch >= 1.11.0
- librosa

## 1. infer demo

```
python DTLN_model.py  --model_path ./pretrained/model.pth  \
   --wav_in ./samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav \
   --wav_out ./out.wav
```

realtime (truck by truck, avg 2ms in pytorch with cpu):

```
python realtime_infer.py  --model_path ./pretrained/model.pth  \
   --wav_in ./samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav \
   --wav_out ./out.wav
```

src wav：[./samples/audioset_realrec_airconditioner_2TE3LoA2OUQ.wav](./samples/)

![](./samples/in.png)


after enhanced: [./samples/enahnced.wav](./samples/)

![](./samples/out.png)


TODO:
- c++ deploy

## Citation
If you find *M-LSD* useful in your project, please consider to cite the following paper.

```
@misc{gu2021realtime,
    title={Towards Real-time and Light-weight Line Segment Detection},
    author={Geonmo Gu and Byungsoo Ko and SeoungHyun Go and Sung-Hyun Lee and Jingeun Lee and Minchul Shin},
    year={2021},
    eprint={2106.00186},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
