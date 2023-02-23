#!/usr/bin/env python3
# Purpose: Converts a PyTorch model to an ONNX model. 
# Prerequisites: torch==1.9.1, onnx
import torch
import os
import sys
import argparse
from cnnnet import CNNNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = ''' 
Converts torch models to either torchscript or ONNX.  For example, we can\n
convert model_10.pt to ONNX with\n
\n
     modelConverter.py --input_file=model_011.pt --onnx_file=pickersClassifiersCNNOneComponentP.onnx\n
''',
     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input_file',
                        default = None,
                        required = True,
                        type = str,
                        help = 'The input torch model from training to convert.')
    parser.add_argument('-o', '--onnx_file',
                        type = str,
                        default = None,
                        help = 'Exports this model to ONNX format with the specified output file name.')
    parser.add_argument('-t', '--torchscript_file',
                        type = str,
                        default = None,
                        help = 'Exports this model to TorchScript for C++ with the specified output file name.')
    parser.add_argument('--hdf5_file',
                        type = str,
                        default = None,
                        help = 'Exports the model weights to an HDF5 file.')
    args = parser.parse_args()

    if (not os.path.exists(args.input_file)):
        print("Input file {} does not exist.".format(args.input_file))
        sys.exit(1)
 
    if (args.onnx_file is None and args.torchscript_file is None):
        print("Nothing to do.")
        sys.exit(0)
    # Prevent overwriting input file
    if (args.onnx_file == args.input_file):
        args.onnx_file = args.onnx_file + '.onnx'
    if (args.torchscript_file == args.input_file):
        args.torchscript_file = args.torch_script + '.pt'

    print("Loading {}...".format(args.input_file))
    batch_size = 16
    num_channels = 1
    signal_length = 400
    cnnnet = CNNNet(num_channels = num_channels,
                    min_lag =-0.75,
                    max_lag = 0.75)
    try:
        check_point = torch.load(args.input_file)
        cnnnet.load_state_dict(check_point['state_dict']) # change from model_state_dict to state_dict
        cnnnet.eval() # Convert model to evaluation mode
    except Exception as e:
        print("Failed to load model.  Failed with: {}".format(str(e)))
        sys.exit(1)

    if (args.torchscript_file is not None):
        print("Converting to TorchScript...")
        script_module = torch.jit.script(cnnnet)
        script_module.save(args.torchscript_file) 
 
    if (args.onnx_file is not None):
        print("Converting to ONNX...")
        dummy_input = torch.randn(1, num_channels, signal_length, requires_grad = False) # Flip/flop length
        torch.onnx.export(cnnnet,
                          (dummy_input, ),
                          f = args.onnx_file,
                          opset_version = 11,
                          input_names = ['input_signal'],
                          output_names = ['output_signal'],
                          dynamic_axes = {'input_signal'  : {0 : 'batch_size'},
                                          'output_signal' : {0 : 'batch_size'}})

    if (args.hdf5_file is not None):
        cnnnet.write_weights_to_hdf5(args.hdf5_file) 
    sys.exit(0) 
