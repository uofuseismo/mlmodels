#!/usr/bin/env python3
import numpy as np
import sys
import torch
import time
from cnnnet import CNNNet

def load_text_file(file_name : str) -> list:
    X = np.zeros([4, 3, 600]) 
    signal1 = []
    signal2 = []
    signal3 = []
    signal4 = []
    i = 0
    with open(file_name) as f:
        lines = f.readlines()
        for l in lines:
            v1, n1, e1, v2, n2, e2, v3, n3, e3, v4, n4, e4  = l.split(',')
            X[0, 0, i] = v1
            X[0, 1, i] = n1
            X[0, 2, i] = e1

            X[1, 0, i] = v2
            X[1, 1, i] = n2
            X[1, 2, i] = e2

            X[2, 0, i] = v3
            X[2, 1, i] = n3
            X[2, 2, i] = e3

            X[3, 0, i] = v4
            X[3, 1, i] = n4
            X[3, 2, i] = e4
            i = i + 1
    return X

def rescale(v : np.array) -> np.array:
    norm = np.max(np.abs(v))
    if (norm < 1.e-10):
        norm = 0
    else:
        norm = 1/norm
    return v*norm

if __name__ == "__main__":
    min_perturbation =-0.85
    max_perturbation = 0.85
    num_channels = 3
    signal_length = 600 
    model_file    = 'sPicker_seed1_model25.pt' #'model_011.pt'
    vertical_file = '../../../testing/data/pickers/cnnThreeComponentS/cnnnetTestInputs.txt'
    output_file = '../../../testing/data/pickers/cnnThreeComponentS/cnnnetTestOutputs.txt'
    

    print("Loading vertical signals...")
    signals = load_text_file(vertical_file)

    print("Loading torch file...")
    cnnnet = CNNNet(num_channels = num_channels,
                    min_lag = min_perturbation,
                    max_lag = max_perturbation)
    try:
        check_point = torch.load(model_file)
        cnnnet.load_state_dict(check_point['state_dict']) # change from model_state_dict to state_dict
        cnnnet.eval() # Convert model to evaluation mode
    except Exception as e:
        print("Failed to load model.  Failed with: {}".format(str(e)))
        sys.exit(1)

    # Open output file 
    ofl = open(output_file, 'w')
    # Evaluate the model
    print("Evaluating model for single instance...")
    X_temp = np.zeros([1, signal_length, num_channels])
    for i_signal in range(signals.shape[0]):
        vertical = signals[i_signal, 0, :]
        north    = signals[i_signal, 1, :]
        east     = signals[i_signal, 2, :]
        assert len(vertical) == signal_length, 'invalid signal length'
        X_temp[0, :, 0] = rescale(east[0:signal_length])
        X_temp[0, :, 1] = rescale(north[0:signal_length])
        X_temp[0, :, 2] = rescale(vertical[0:signal_length])
        X = torch.from_numpy(X_temp.transpose(0, 2, 1)).float()
        t0 = time.perf_counter()
        p_torch = cnnnet.forward(X)
        t1 = time.perf_counter()
        print("Evaluation time: {} (s)".format(t1 - t0))
        perturbation = p_torch.squeeze().to('cpu').detach().numpy()
        ofl.write('{}\n'.format(perturbation))
    ofl.close()
