#!/usr/bin/env python3
import numpy as np
import sys
import torch
import time
from unet import UNet

def load_text_file(file_name : str):
     
    times = []
    signal = []
    with open(file_name) as f:
        lines = f.readlines()
        for l in lines:
            time, value = l.split(',')
            times.append(float(time))
            signal.append(float(value))
    return np.asarray(times), np.asarray(signal)

def rescale(v : np.array) -> np.array:
    norm = np.max(np.abs(v))
    if (norm < 1.e-10):
        norm = 0
    else:
        norm = 1/norm
    return v*norm

def create_windows(n, center = 500, signal_length = 1008):
    i0_window = signal_length//2 - center//2
    i1_window = signal_length//2 + center//2
    i0 = i0_window
    ds = []
    for k in range(n):
        source_start = i0 - i0_window
        source_end   = source_start + signal_length
        if (source_end >= n):
            if (source_end == n):
                do_last = False
            else:
                do_last = True
            break
        if (k == 0):
            destination_start = 0
            destination_end   = i0_window + center
        else:
            destination_start = i0
            destination_end   = destination_start + center
        d = {'source_start' : source_start,
             'source_end'   : source_end,
             'destination_start' : destination_start,
             'destination_end' : destination_end} 
        ds.append(d)
        i0 = i0 + center
    # Loop
    # Get last one
    if (do_last):
       d = {'source_start' : n - signal_length,
            'source_end'  : n,
            'destination_start' : destination_end,
            'destination_end' : n}
       print(d)
       ds.append(d)
    return ds

#def rescale(v : np.array):
#    v_min = np.min(v)
#    v_max = np.max(v)
#    return (v - v_min)/(v_max - v_min)

if __name__ == "__main__":
    #torch.set_num_threads(1)
    num_channels = 1
    signal_length = 1008
    center = 500 # Use middle 500 samples
    model_file    = 'oneCompPDetectorMEW_model_022.pt'
    vertical_file = '../../../testing/data/detectors/uNetOneComponentP/PB.B206.EHZ.PROC.zrunet_p.txt'
    output_example_file = '../../../testing/data/detectors/uNetOneComponentP/PB.B206.P_PROBA.txt'
    output_sliding_file = '../../../testing/data/detectors/uNetOneComponentP/PB.B206.P_PROBA.SLIDING.txt'

    print("Loading vertical signal...")
    times, vertical = load_text_file(vertical_file)

    print("Creating windows...")
    windows = create_windows(len(vertical), center = center, signal_length = signal_length)

    print("Loading torch file...")
    unet = UNet(num_channels = num_channels, num_classes = 1)
    try:
        check_point = torch.load(model_file)
        unet.load_state_dict(check_point['model_state_dict'])
        unet.eval() # Convert model to evaluation mode
    except Exception as e:
        print("Failed to load model.  Failed with: {}".format(str(e)))
        sys.exit(1)

    # Evaluate the model
    print("Evaluating model for single instance...")
    X_temp = np.zeros([1, signal_length, num_channels])
    p = np.zeros(len(vertical))
    X_temp[0, :, 0] = rescale(vertical[0:signal_length])
    X = torch.from_numpy(X_temp.transpose(0, 2, 1)).float()
    t0 = time.perf_counter()
    p_torch = unet.forward(X)
    t1 = time.perf_counter()
    print("Evaluation time: {} (s)".format(t1 - t0))
    p_work = p_torch.squeeze().to('cpu').detach().numpy()
    ofl = open(output_example_file, 'w')
    for i in range(len(p_work)):
        ofl.write('%f,%e\n'%(times[i], p_work[i]))
    ofl.close()

    n_evaluations = 0
    t_total_elapsed = 0
    for iw in range(len(windows)):
        w = windows[iw]
        #i1 = i*signal_length
        #i2 = (i + 1)*signal_length
        #if (i2 > len(vertical)):
        #    break
        i1 = w['source_start']
        i2 = w['source_end']
        j1 = w['destination_start']
        j2 = w['destination_end']
        print(i1, i2, j1, j2)
        # Alysha uses ENZ
        X_temp[0, :, 0] = rescale(vertical[i1:i2])
        X = torch.from_numpy(X_temp.transpose(0, 2, 1)).float()
        t0 = time.perf_counter()
        p_torch = unet.forward(X)
        t1 = time.perf_counter()
        t_elapsed = t1 - t0
        t_total_elapsed = t_total_elapsed + t_elapsed
        
        p_work = p_torch.squeeze().to('cpu').detach().numpy()
        # Just copy the first signal completely
        if (iw == 0):
            p[0:signal_length] = p_work[0:signal_length]
        elif (iw == len(windows) - 1):
            n_copy = j2 - j1
            p[j1:j2] = p_work[signal_length - n_copy:signal_length]
        else:
            # Handle last case
            n_copy = j2 - j1
            p[j1:j2] = p_work[254:254+n_copy] # Overwrite next go-around
        n_evaluations = n_evaluations + 1
    # Loop 
    print("Cumulative evaluation time: {} (s)".format(t_total_elapsed))
    print("Average evaluation time: {} (s)".format(t_total_elapsed/n_evaluations))
    ofl = open(output_sliding_file, 'w') 
    for i in range(len(times)):
        ofl.write('%f,%e\n'%(times[i], p[i]))
    ofl.close()
