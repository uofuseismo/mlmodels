#!/usr/bin/env python3
import numpy as np
import sys
import torch
import time
from fmnet import FMNet

def load_text_file(file_name : str) -> list:
     
    signal1 = []
    signal2 = []
    signal3 = []
    signal4 = []
    signal5 = []
    with open(file_name) as f:
        lines = f.readlines()
        for l in lines:
            s1, s2, s3, s4, s5 = l.split(',')
            signal1.append(float(s1))
            signal2.append(float(s2))
            signal3.append(float(s3))
            signal4.append(float(s4))
            signal5.append(float(s5))
    return [np.asarray(signal1), np.asarray(signal2), np.asarray(signal3),
            np.asarray(signal4), np.asarray(signal5)]

def rescale(v : np.array) -> np.array:
    norm = np.max(np.abs(v))
    if (norm < 1.e-10):
        norm = 0
    else:
        norm = 1/norm
    return v*norm

if __name__ == "__main__":
    num_channels = 1
    signal_length = 400 
    model_file    = 'model_011.pt'
    vertical_file = '../../../testing/data/firstMotionClassifiers/cnnOneComponentP/fmnetTestInputs.txt'
    output_file = '../../../testing/data/firstMotionClassifiers/cnnOneComponentP/fmnetTestOutputs.txt'
    

    print("Loading vertical signals...")
    signals = load_text_file(vertical_file)

    print("Loading torch file...")
    fmnet = FMNet(num_channels = num_channels, num_classes = 3)
    try:
        check_point = torch.load(model_file)
        fmnet.load_state_dict(check_point['model_state_dict'])
        fmnet.eval() # Convert model to evaluation mode
    except Exception as e:
        print("Failed to load model.  Failed with: {}".format(str(e)))
        sys.exit(1)

    # Open output file 
    ofl = open(output_file, 'w')
    # Evaluate the model
    print("Evaluating model for single instance...")
    X_temp = np.zeros([1, signal_length, num_channels])
    for signal in signals:
        assert len(signal) == signal_length, 'invalid signal length'
        X_temp[0, :, 0] = rescale(signal[0:signal_length])
        X = torch.from_numpy(X_temp.transpose(0, 2, 1)).float()
        t0 = time.perf_counter()
        p_torch = fmnet.forward(X)
        t1 = time.perf_counter()
        print("Evaluation time: {} (s)".format(t1 - t0))
        p_work = p_torch.squeeze().to('cpu').detach().numpy()
        ofl.write('{}, {}, {}\n'.format(p_work[0], p_work[1], p_work[2]))
    ofl.close()
