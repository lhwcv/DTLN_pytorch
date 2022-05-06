#!/usr/bin/env bash

./build/dtln ./models/dtln/  test_data/test.pcm out_dtln.pcm
./build/nsnet ./models/nsnet2/  test_data/test.pcm out_nsnet2.pcm