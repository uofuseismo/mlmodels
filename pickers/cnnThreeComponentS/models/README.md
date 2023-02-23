To convert the x'th epoch that was fine-tuned from STEAD to the production model:

    ~/anaconda3/bin/python modelConverter.py -i sPicker_seed1_model25.pt -o pickersCNNThreeComponentS.onnx --hdf5_file pickersCNNThreeComponentS.h5
    git lfs track pickersCNNThreeComponentS.onnx
    git add pickersCNNThreeComponentS.onnx
