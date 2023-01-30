#!/usr/bin/env python3
# Purpose: Converts a PyTorch model to an ONNX model. 
# Prerequisites: torch==1.9.1, onnx
import torch
import os
import sys
import argparse
from unet import UNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = ''' 
Converts torch models to either torchscript or ONNX.  For example, we can\n
convert model_10.pt to ONNX with\n
\n
     modelConverter.py --input_file=model_10.pt --onnx_file=model_10.onnx\n
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
    signal_length = 1008
    unet = UNet(num_channels = num_channels, num_classes = 1)
    try:
        check_point = torch.load(args.input_file)
        unet.load_state_dict(check_point['model_state_dict'])
        unet.eval() # Convert model to evaluation mode
    except Exception as e:
        print("Failed to load model.  Failed with: {}".format(str(e)))
        sys.exit(1)

    if (args.torchscript_file is not None):
        print("Converting to TorchScript...")
        script_module = torch.jit.script(unet)
        script_module.save(args.torchscript_file) 
 
    if (args.onnx_file is not None):
        print("Converting to ONNX...")
        dummy_input = torch.randn(1, num_channels, signal_length, requires_grad = False) # Flip/flop length
        torch.onnx.export(unet,
                          (dummy_input, ),
                          f = args.onnx_file,
                          opset_version = 11,
                          input_names = ['input_signal'],
                          output_names = ['output_signal'],
                          dynamic_axes = {'input_signal'  : {0 : 'batch_size'},
                                          'output_signal' : {0 : 'batch_size'}})
 

    sys.exit(0) 
